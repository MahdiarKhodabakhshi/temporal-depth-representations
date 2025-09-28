#!/usr/bin/env python3
"""
Select (checkpoint, layer) pairs per task using a factorized selector:
1. Rank checkpoints by representation dispersion on unlabeled task texts.
2. Within each top-K checkpoint, choose a layer by existing entropy metrics.
3. Optionally add stability regularization over neighboring checkpoints.
4. Evaluate by looking up the already-computed task main_score at the selected pair.

This script is intentionally analysis-only. It reuses the existing checkpoint/layout
conventions and does not introduce a new MTEB-Harness mode.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments" / "results"
DEFAULT_RERUNS_ROOT = REPO_ROOT / "experiments" / "results_reruns"

import numpy as np
import pandas as pd
import torch


TASK_TO_DATASET = {
    "AmazonCounterfactualClassification": "amazon_counterfactual",
    "AmazonReviewsClassification": "amazon_reviews_multi",
    "ArxivClusteringS2S": "arxiv-clustering-s2s",
    "AskUbuntuDupQuestions": "askubuntudupquestions-reranking",
    "BIOSSES": "biosses-sts",
    "Banking77Classification": "banking77",
    "BiorxivClusteringS2S": "biorxiv-clustering-s2s",
    "EmotionClassification": "emotion",
    "MTOPDomainClassification": "mtop_domain",
    "MTOPIntentClassification": "mtop_intent",
    "MassiveIntentClassification": "amazon_massive_intent",
    "MassiveScenarioClassification": "amazon_massive_scenario",
    "MedrxivClusteringS2S": "medrxiv-clustering-s2s",
    "MindSmallReranking": "mind_small",
    "RedditClustering": "reddit-clustering",
    "SICK-R": "sickr-sts",
    "STS12": "sts12-sts",
    "STS13": "sts13-sts",
    "STS14": "sts14-sts",
    "STS15": "sts15-sts",
    "STS16": "sts16-sts",
    "STS17": "sts17-crosslingual-sts",
    "STSBenchmark": "stsbenchmark-sts",
    "SciDocsRR": "scidocs-reranking",
    "SprintDuplicateQuestions": "sprintduplicatequestions-pairclassification",
    "StackExchangeClustering": "stackexchange-clustering",
    "StackOverflowDupQuestions": "stackoverflowdupquestions-reranking",
    "ToxicConversationsClassification": "toxic_conversations_50k",
    "TweetSentimentExtractionClassification": "tweet_sentiment_extraction",
    "TwentyNewsgroupsClustering": "twentynewsgroups-clustering",
    "TwitterSemEval2015": "twittersemeval2015-pairclassification",
    "TwitterURLCorpus": "twitterurlcorpus-pairclassification",
}



def _load_dataset_helpers():
    from experiments.utils.dataloaders.text_dataloader import _load_mteb_dataset_split, find_data_key_in_examples
    return _load_mteb_dataset_split, find_data_key_in_examples


def _load_model_wrapper_classes():
    from experiments.utils.model_definitions.text_automodel_wrapper import TextLayerwiseAutoModelWrapper, TextModelSpecifications
    return TextLayerwiseAutoModelWrapper, TextModelSpecifications

@dataclass(frozen=True)
class SelectorConfig:
    model_family: str
    model_size: str
    revisions: list[str]
    candidate_layers: list[int]
    dispersion_layer: int
    top_k: int
    layer_score_mode: str
    checkpoint_choice_mode: str
    dispersion_stability_lambda: float
    entropy_stability_mu: float
    pooling_method: str
    dispersion_num_samples: int
    sample_seed: int
    dispersion_num_pairs: int
    batch_size: int
    max_sample_length: int
    baseline_revision: str


class SelectionError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_family", type=str, default="Pythia")
    parser.add_argument("--model_size", type=str, default="410m")
    parser.add_argument(
        "--entropy_root",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT / "Pythia" / "410m"),
    )
    parser.add_argument(
        "--main_runs_root",
        type=str,
        default=str(DEFAULT_RERUNS_ROOT),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DEFAULT_RERUNS_ROOT),
    )
    parser.add_argument(
        "--expected_revisions",
        type=str,
        default="step143000,step135000,step128000,step115000,step105000,step95000,step85000,step75000,step65000,step55000",
    )
    parser.add_argument("--layer_start", type=int, default=15)
    parser.add_argument("--layer_end", type=int, default=24)
    parser.add_argument("--dispersion_layer", type=int, default=-1)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument(
        "--layer_score_mode",
        type=str,
        default="minmax_sum",
        choices=["dataset_entropy", "dime", "infonce", "rank_sum", "zsum", "minmax_sum"],
    )
    parser.add_argument(
        "--checkpoint_choice_mode",
        type=str,
        default="combined_rank",
        choices=["dispersion_only", "combined_rank"],
    )
    parser.add_argument("--dispersion_stability_lambda", type=float, default=0.0)
    parser.add_argument("--entropy_stability_mu", type=float, default=0.0)
    parser.add_argument("--pooling_method", type=str, default="mean")
    parser.add_argument("--dispersion_num_samples", type=int, default=1000)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--dispersion_num_pairs", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_sample_length", type=int, default=512)
    parser.add_argument("--baseline_revision", type=str, default="step143000")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument(
        "--task_names",
        type=str,
        default="",
        help="Optional comma-separated subset of task names to run.",
    )
    parser.add_argument(
        "--cache_task_texts_path",
        type=str,
        default="",
        help="Optional JSON cache path for sampled task texts. If present, load from it; otherwise write to it after sampling.",
    )
    return parser.parse_args()


def _parse_revision_order_key(revision: str) -> tuple[int, str]:
    lowered = revision.lower()
    if lowered == "main":
        return (10**12, revision)
    digits = "".join(ch for ch in revision if ch.isdigit())
    if digits:
        return (int(digits), revision)
    return (-1, revision)


def _revision_seed_value(revision: str) -> int:
    digits = "".join(ch for ch in revision if ch.isdigit())
    if digits:
        return int(digits)
    if revision.lower() == "main":
        return 10**12
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(revision))


def _pick_latest_existing(paths: Sequence[str]) -> str | None:
    existing = [p for p in paths if os.path.isdir(p)]
    if not existing:
        return None
    return max(existing, key=os.path.getmtime)


def find_main_revision_root(main_runs_root: str, revision: str, layer_start: int, layer_end: int) -> str | None:
    direct_path = os.path.join(main_runs_root, revision)
    if os.path.isdir(os.path.join(direct_path, "mteb")):
        return direct_path

    layer_range = f"layers{layer_start}-{layer_end}"
    patterns = [
        os.path.join(
            main_runs_root,
            f"pythia410m_{revision}_{layer_range}_*",
            "Pythia",
            "410m",
            revision,
        ),
        os.path.join(
            main_runs_root,
            f"pythia410m_{revision}_*{layer_range}*",
            "Pythia",
            "410m",
            revision,
        ),
        os.path.join(
            main_runs_root,
            f"pythia410m_{revision}_alllayers_*",
            "Pythia",
            "410m",
            revision,
        ),
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    return _pick_latest_existing(matches)


def _load_pickle(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_main_score(payload: dict[str, Any]) -> float:
    return float(payload["scores"]["test"][0]["main_score"])


def _rank_desc(values: pd.Series) -> pd.Series:
    return values.rank(method="average", ascending=False)


def _rank_asc(values: pd.Series) -> pd.Series:
    return values.rank(method="average", ascending=True)


def _zscore(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0))
    if std == 0.0 or math.isnan(std):
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - float(values.mean())) / std


def _minmax(values: pd.Series) -> pd.Series:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax == vmin or math.isnan(vmax) or math.isnan(vmin):
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - vmin) / (vmax - vmin)


def _flatten_text_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_text_value(item))
        return out
    return []


def _extract_texts_from_dataset(dataset: Any) -> list[str]:
    if len(dataset) == 0:
        return []
    _, find_data_key_in_examples = _load_dataset_helpers()
    first = dataset[0]
    data_key = find_data_key_in_examples(first)
    texts: list[str] = []
    for example in dataset:
        texts.extend(_flatten_text_value(example[data_key]))
    return texts


def load_sampled_task_texts(
    task_to_dataset: dict[str, str],
    num_samples: int,
    cache_path: str | None,
    sample_seed: int,
) -> dict[str, list[str]]:
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict) and "task_texts" in payload:
            meta = payload.get("meta", {})
            cached_seed = meta.get("sample_seed")
            if cached_seed is not None and int(cached_seed) != int(sample_seed):
                raise SelectionError(
                    f"Cached task texts at {cache_path} were built with sample_seed={cached_seed}, not sample_seed={sample_seed}."
                )
            payload = payload["task_texts"]

        return {str(k): [str(x) for x in v] for k, v in payload.items()}

    _load_mteb_dataset_split, _ = _load_dataset_helpers()

    sampled: dict[str, list[str]] = {}
    for task_idx, (task_name, dataset_stub) in enumerate(task_to_dataset.items()):
        dataset_name = f"mteb/{dataset_stub}"
        dataset = _load_mteb_dataset_split(dataset_name, "test")
        texts = _extract_texts_from_dataset(dataset)
        if not texts:
            continue

        sample_count = min(num_samples, len(texts))
        if sample_count < len(texts):
            rng = np.random.default_rng(sample_seed + task_idx)
            chosen_idx = rng.choice(len(texts), size=sample_count, replace=False)
            chosen_idx = np.sort(chosen_idx)
            sampled[task_name] = [texts[int(i)] for i in chosen_idx]
        else:
            sampled[task_name] = list(texts)

    if cache_path:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        payload = {
            "meta": {
                "sample_seed": int(sample_seed),
                "num_samples": int(num_samples),
            },
            "task_texts": sampled,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return sampled


def compute_dispersion(embeddings: np.ndarray, num_pairs: int, seed: int = 0) -> float:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    num_samples = int(embeddings.shape[0])
    if num_samples < 2:
        return float("nan")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = embeddings / norms

    total_pairs = (num_samples * (num_samples - 1)) // 2
    if total_pairs <= num_pairs:
        sims = normalized @ normalized.T
        iu = np.triu_indices(num_samples, k=1)
        dists = 1.0 - sims[iu]
        return float(np.mean(dists))

    rng = np.random.default_rng(seed)
    left = rng.integers(0, num_samples, size=num_pairs, endpoint=False)
    right = rng.integers(0, num_samples - 1, size=num_pairs, endpoint=False)
    right = np.where(right >= left, right + 1, right)
    sims = np.sum(normalized[left] * normalized[right], axis=1)
    dists = 1.0 - sims
    return float(np.mean(dists))


def compute_task_checkpoint_dispersion(
    task_texts: dict[str, list[str]],
    revisions: Sequence[str],
    config: SelectorConfig,
    device_map: str,
) -> pd.DataFrame:
    TextLayerwiseAutoModelWrapper, TextModelSpecifications = _load_model_wrapper_classes()

    rows: list[dict[str, Any]] = []
    for revision in revisions:
        revision_seed_value = _revision_seed_value(revision)
        model_specs = TextModelSpecifications(config.model_family, config.model_size, revision)
        model = TextLayerwiseAutoModelWrapper(
            model_specs,
            device_map=device_map,
            evaluation_layer_idx=config.dispersion_layer,
        )
        model.print_loading_message()
        for task_idx, (task_name, texts) in enumerate(task_texts.items()):
            if not texts:
                continue
            embeddings = model.encode(
                texts,
                batch_size=config.batch_size,
                num_workers=0,
                pooling_method=config.pooling_method,
                max_sample_length=config.max_sample_length,
                verbose=False,
            )
            pair_seed = int(config.sample_seed + 1000 * task_idx + revision_seed_value)
            dispersion = compute_dispersion(
                embeddings,
                num_pairs=config.dispersion_num_pairs,
                seed=pair_seed,
            )
            rows.append(
                {
                    "task_name": task_name,
                    "revision": revision,
                    "dispersion": float(dispersion),
                    "num_samples": int(len(texts)),
                    "embedding_dim": int(embeddings.shape[1]),
                    "dispersion_layer": int(model.evaluation_layer_idx),
                    "dispersion_revision_seed_value": int(revision_seed_value),
                    "dispersion_pair_seed": pair_seed,
                }
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise SelectionError("No dispersion rows were produced.")
    return pd.DataFrame(rows)


def _load_entropy_arrays(
    entropy_root: str,
    revision: str,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    base = os.path.join(entropy_root, revision, "metrics", "mteb", dataset_name, "test")
    entropy_obj = _load_pickle(os.path.join(base, "entropy_dataset.pkl"))
    infonce_obj = _load_pickle(os.path.join(base, "infonce.pkl"))
    dime_obj = _load_pickle(os.path.join(base, "dime.pkl"))

    dataset_entropy = np.asarray(entropy_obj["maxEntropy"], dtype=float)
    if "mi-lower-bound" in infonce_obj:
        infonce = np.asarray(infonce_obj["mi-lower-bound"], dtype=float)
        infonce_for_min = -infonce
        infonce_source = "mi-lower-bound"
    elif "mi_lower_bound" in infonce_obj:
        infonce = np.asarray(infonce_obj["mi_lower_bound"], dtype=float)
        infonce_for_min = -infonce
        infonce_source = "mi_lower_bound"
    elif "raw" in infonce_obj:
        infonce = np.asarray(infonce_obj["raw"], dtype=float)
        infonce_for_min = infonce
        infonce_source = "raw"
    else:
        raise KeyError(f"Unsupported infonce keys for {revision}/{dataset_name}: {sorted(infonce_obj.keys())}")
    dime = np.asarray(dime_obj["maxEntropy"], dtype=float)
    return dataset_entropy, infonce, infonce_for_min, dime, infonce_source


def _load_main_scores_for_layer(main_revision_root: str, layer_idx: int) -> dict[str, float]:
    layer_dir = os.path.join(main_revision_root, "mteb", f"layer_{layer_idx}")
    scores: dict[str, float] = {}
    for json_path in sorted(glob.glob(os.path.join(layer_dir, "*.json"))):
        if json_path.endswith("model_meta.json"):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        scores[str(payload["task_name"])] = _safe_main_score(payload)
    return scores


def build_task_checkpoint_layer_table(
    entropy_root: str,
    main_runs_root: str,
    revisions: Sequence[str],
    task_to_dataset: dict[str, str],
    candidate_layers: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    revision_roots: dict[str, str] = {}
    for revision in revisions:
        main_root = find_main_revision_root(main_runs_root, revision, min(candidate_layers), max(candidate_layers))
        if main_root is None:
            continue
        revision_roots[revision] = main_root
        main_scores_by_layer = {layer_idx: _load_main_scores_for_layer(main_root, layer_idx) for layer_idx in candidate_layers}
        entropy_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = {}
        for task_name, dataset_stub in task_to_dataset.items():
            try:
                if dataset_stub not in entropy_cache:
                    entropy_cache[dataset_stub] = _load_entropy_arrays(entropy_root, revision, dataset_stub)
                dataset_entropy, infonce_raw, infonce_for_min, dime, infonce_source = entropy_cache[dataset_stub]
            except FileNotFoundError:
                continue

            for layer_idx in candidate_layers:
                if layer_idx >= len(dataset_entropy) or layer_idx >= len(infonce_raw) or layer_idx >= len(dime):
                    continue
                main_score = main_scores_by_layer[layer_idx].get(task_name)
                if main_score is None:
                    continue
                rows.append(
                    {
                        "task_name": task_name,
                        "dataset_name": dataset_stub,
                        "revision": revision,
                        "layer_idx": int(layer_idx),
                        "dataset_entropy": float(dataset_entropy[layer_idx]),
                        "infonce_raw": float(infonce_raw[layer_idx]),
                        "infonce_for_min": float(infonce_for_min[layer_idx]),
                        "infonce_source": str(infonce_source),
                        "dime": float(dime[layer_idx]),
                        "main_score": float(main_score),
                    }
                )

    if not rows:
        raise SelectionError("No (task, checkpoint, layer) rows were built from entropy/main-score inputs.")

    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    return df, revision_roots


def score_layers_within_checkpoint(df: pd.DataFrame, layer_score_mode: str) -> pd.DataFrame:
    scored_groups: list[pd.DataFrame] = []
    for (_, _), group in df.groupby(["task_name", "revision"], sort=False):
        group = group.sort_values("layer_idx").copy()
        group["layer_score"] = np.nan

        if layer_score_mode == "dataset_entropy":
            valid = group.dropna(subset=["dataset_entropy"]).copy()
            group.loc[valid.index, "layer_score"] = valid["dataset_entropy"]
        elif layer_score_mode == "dime":
            valid = group.dropna(subset=["dime"]).copy()
            group.loc[valid.index, "layer_score"] = valid["dime"]
        elif layer_score_mode == "infonce":
            valid = group.dropna(subset=["infonce_for_min"]).copy()
            group.loc[valid.index, "layer_score"] = valid["infonce_for_min"]
        elif layer_score_mode == "rank_sum":
            valid = group.dropna(subset=["dataset_entropy", "infonce_for_min", "dime"]).copy()
            if not valid.empty:
                valid["layer_score"] = (
                    _rank_asc(valid["dataset_entropy"]) +
                    _rank_asc(valid["infonce_for_min"]) +
                    _rank_asc(valid["dime"])
                )
                group.loc[valid.index, "layer_score"] = valid["layer_score"]
        elif layer_score_mode == "zsum":
            valid = group.dropna(subset=["dataset_entropy", "infonce_for_min", "dime"]).copy()
            if not valid.empty:
                valid["layer_score"] = (
                    _zscore(valid["dataset_entropy"]) +
                    _zscore(valid["infonce_for_min"]) +
                    _zscore(valid["dime"])
                )
                group.loc[valid.index, "layer_score"] = valid["layer_score"]
        elif layer_score_mode == "minmax_sum":
            valid = group.dropna(subset=["dataset_entropy", "infonce_for_min", "dime"]).copy()
            if not valid.empty:
                valid["layer_score"] = (
                    _minmax(valid["dataset_entropy"]) +
                    _minmax(valid["infonce_for_min"]) +
                    _minmax(valid["dime"])
                )
                group.loc[valid.index, "layer_score"] = valid["layer_score"]
        else:
            raise ValueError(f"Unsupported layer_score_mode: {layer_score_mode}")

        scored_groups.append(group)

    return pd.concat(scored_groups, ignore_index=True)


def add_entropy_stability(df: pd.DataFrame, revisions_in_order: Sequence[str], mu: float) -> pd.DataFrame:
    if mu == 0.0:
        out = df.copy()
        out["layer_score_stable"] = out["layer_score"]
        out["layer_score_delta"] = 0.0
        return out

    out = df.copy()
    revision_to_prev: dict[str, str | None] = {}
    prev: str | None = None
    for revision in revisions_in_order:
        revision_to_prev[revision] = prev
        prev = revision

    deltas: list[float] = []
    stable_scores: list[float] = []
    for _, row in out.iterrows():
        prev_revision = revision_to_prev.get(str(row["revision"]))
        if prev_revision is None:
            delta = 0.0
        else:
            prev_match = out[
                (out["task_name"] == row["task_name"]) &
                (out["revision"] == prev_revision) &
                (out["layer_idx"] == row["layer_idx"])
            ]
            if len(prev_match) == 0 or pd.isna(prev_match.iloc[0]["layer_score"]):
                delta = 0.0
            else:
                delta = abs(float(row["layer_score"]) - float(prev_match.iloc[0]["layer_score"]))
        deltas.append(float(delta))
        stable_scores.append(float(row["layer_score"]) + mu * float(delta))

    out["layer_score_delta"] = deltas
    out["layer_score_stable"] = stable_scores
    return out


def choose_layers_per_checkpoint(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (task_name, revision), group in df.groupby(["task_name", "revision"], sort=False):
        valid = group.dropna(subset=["layer_score_stable"])
        if valid.empty:
            continue
        chosen = valid.sort_values(["layer_score_stable", "layer_idx"], ascending=[True, True]).iloc[0]
        rows.append(
            {
                "task_name": task_name,
                "revision": revision,
                "chosen_layer": int(chosen["layer_idx"]),
                "selected_layer_score": float(chosen["layer_score"]),
                "selected_layer_score_stable": float(chosen["layer_score_stable"]),
                "selected_layer_score_delta": float(chosen["layer_score_delta"]),
                "selected_dataset_entropy": float(chosen["dataset_entropy"]),
                "selected_infonce_raw": float(chosen["infonce_raw"]),
                "selected_infonce_for_min": float(chosen["infonce_for_min"]),
                "selected_infonce_source": str(chosen["infonce_source"]),
                "selected_dime": float(chosen["dime"]),
                "selected_main_score": float(chosen["main_score"]),
            }
        )
    if not rows:
        raise SelectionError("No valid per-checkpoint layer selections were produced.")
    return pd.DataFrame(rows)


def add_dispersion_stability(df: pd.DataFrame, revisions_in_order: Sequence[str], lam: float) -> pd.DataFrame:
    if lam == 0.0:
        out = df.copy()
        out["dispersion_stable"] = out["dispersion"]
        out["dispersion_delta"] = 0.0
        return out

    out = df.copy()
    revision_to_prev: dict[str, str | None] = {}
    prev: str | None = None
    for revision in revisions_in_order:
        revision_to_prev[revision] = prev
        prev = revision

    deltas: list[float] = []
    stable_scores: list[float] = []
    for _, row in out.iterrows():
        prev_revision = revision_to_prev.get(str(row["revision"]))
        if prev_revision is None:
            delta = 0.0
        else:
            prev_match = out[(out["task_name"] == row["task_name"]) & (out["revision"] == prev_revision)]
            if len(prev_match) == 0 or pd.isna(prev_match.iloc[0]["dispersion"]):
                delta = 0.0
            else:
                delta = abs(float(row["dispersion"]) - float(prev_match.iloc[0]["dispersion"]))
        deltas.append(float(delta))
        stable_scores.append(float(row["dispersion"]) - lam * float(delta))

    out["dispersion_delta"] = deltas
    out["dispersion_stable"] = stable_scores
    return out


def choose_pairs(
    dispersion_df: pd.DataFrame,
    chosen_layers_df: pd.DataFrame,
    top_k: int,
    checkpoint_choice_mode: str,
) -> pd.DataFrame:
    merged = dispersion_df.merge(chosen_layers_df, on=["task_name", "revision"], how="inner")
    if merged.empty:
        raise SelectionError("No overlap between dispersion table and per-checkpoint layer selections.")

    rows: list[dict[str, Any]] = []
    for task_name, group in merged.groupby("task_name", sort=False):
        ranked = group.sort_values(["dispersion_stable", "revision"], ascending=[False, True]).copy()
        candidates = ranked.head(top_k).copy()
        if candidates.empty:
            continue

        candidates["dispersion_rank_desc"] = _rank_desc(candidates["dispersion_stable"])
        candidates["entropy_rank_asc"] = _rank_asc(candidates["selected_layer_score_stable"])
        candidates["combined_rank_score"] = candidates["dispersion_rank_desc"] + candidates["entropy_rank_asc"]

        if checkpoint_choice_mode == "dispersion_only":
            selected = candidates.sort_values(["dispersion_stable", "selected_layer_score_stable", "revision"], ascending=[False, True, True]).iloc[0]
        elif checkpoint_choice_mode == "combined_rank":
            selected = candidates.sort_values(["combined_rank_score", "dispersion_stable", "revision"], ascending=[True, False, True]).iloc[0]
        else:
            raise ValueError(f"Unsupported checkpoint_choice_mode: {checkpoint_choice_mode}")

        rows.append(
            {
                "task_name": task_name,
                "selected_revision": str(selected["revision"]),
                "selected_layer": int(selected["chosen_layer"]),
                "selected_main_score": float(selected["selected_main_score"]),
                "selected_dataset_entropy": float(selected["selected_dataset_entropy"]),
                "selected_infonce_raw": float(selected["selected_infonce_raw"]),
                "selected_infonce_source": str(selected["selected_infonce_source"]),
                "selected_dime": float(selected["selected_dime"]),
                "selected_layer_score": float(selected["selected_layer_score"]),
                "selected_layer_score_stable": float(selected["selected_layer_score_stable"]),
                "selected_layer_score_delta": float(selected["selected_layer_score_delta"]),
                "selected_dispersion": float(selected["dispersion"]),
                "selected_dispersion_stable": float(selected["dispersion_stable"]),
                "selected_dispersion_delta": float(selected["dispersion_delta"]),
                "num_candidate_checkpoints": int(len(candidates)),
                "candidate_revisions": ",".join(candidates["revision"].astype(str).tolist()),
                "checkpoint_choice_mode": checkpoint_choice_mode,
            }
        )
    if not rows:
        raise SelectionError("No final task-level pair selections were produced.")
    return pd.DataFrame(rows)


def compute_baselines(
    all_rows: pd.DataFrame,
    baseline_revision: str,
    task_subset: Iterable[str],
    main_runs_root: str,
    candidate_layers: Sequence[int],
) -> dict[str, Any]:
    task_subset = list(task_subset)
    baseline_rows = all_rows[(all_rows["revision"] == baseline_revision) & (all_rows["task_name"].isin(task_subset))].copy()
    result: dict[str, Any] = {
        "same_task_best_layer": None,
        "same_task_best_avg_main": None,
        "same_task_last_layer": None,
        "same_task_last_layer_avg_main": None,
        "full_table_best_layer": None,
        "full_table_best_avg_main": None,
        "full_table_last_layer": None,
        "full_table_last_layer_avg_main": None,
    }
    if not baseline_rows.empty:
        grouped = baseline_rows.groupby("layer_idx", as_index=False)["main_score"].mean()
        grouped = grouped.rename(columns={"main_score": "avg_main_score"}).sort_values("layer_idx")
        best = grouped.sort_values(["avg_main_score", "layer_idx"], ascending=[False, True]).iloc[0]
        last = grouped[grouped["layer_idx"] == max(candidate_layers)].iloc[0]
        result["same_task_best_layer"] = int(best["layer_idx"])
        result["same_task_best_avg_main"] = float(best["avg_main_score"])
        result["same_task_last_layer"] = int(last["layer_idx"])
        result["same_task_last_layer_avg_main"] = float(last["avg_main_score"])

    baseline_root = find_main_revision_root(main_runs_root, baseline_revision, min(candidate_layers), max(candidate_layers))
    if baseline_root is not None:
        csv_path = os.path.join(baseline_root, "average_main_score", "avg_main_score_by_layer.csv")
        if os.path.isfile(csv_path):
            table = pd.read_csv(csv_path)
            if not table.empty:
                table["layer_idx"] = table["layer"].astype(str).str.replace("layer_", "", regex=False).astype(int)
                best = table.sort_values(["avg_main_score", "layer_idx"], ascending=[False, True]).iloc[0]
                last = table.sort_values("layer_idx").iloc[-1]
                result["full_table_best_layer"] = int(best["layer_idx"])
                result["full_table_best_avg_main"] = float(best["avg_main_score"])
                result["full_table_last_layer"] = int(last["layer_idx"])
                result["full_table_last_layer_avg_main"] = float(last["avg_main_score"])

    return result


def write_summary_markdown(
    output_dir: str,
    config: SelectorConfig,
    selected_pairs: pd.DataFrame,
    baselines: dict[str, Any],
    revisions_used: Sequence[str],
    tasks_missing: Sequence[str],
) -> None:
    avg_main = float(selected_pairs["selected_main_score"].mean())
    lines = [
        "# Factorized Dispersion + Entropy Selector",
        "",
        "## Configuration",
        f"- model: `{config.model_family} {config.model_size}`",
        f"- revisions used: `{', '.join(revisions_used)}`",
        f"- candidate layers: `{min(config.candidate_layers)}..{max(config.candidate_layers)}`",
        f"- dispersion layer: `{config.dispersion_layer}`",
        f"- top-k checkpoints: `{config.top_k}`",
        f"- layer score mode: `{config.layer_score_mode}`",
        f"- checkpoint choice mode: `{config.checkpoint_choice_mode}`",
        f"- dispersion stability lambda: `{config.dispersion_stability_lambda}`",
        f"- entropy stability mu: `{config.entropy_stability_mu}`",
        "",
        "## Result",
        f"- tasks selected: `{len(selected_pairs)}`",
        f"- average selected main score: `{avg_main:.10f}`",
    ]
    if baselines.get("same_task_best_avg_main") is not None:
        lines.append(
            f"- same-task baseline best layer at `{config.baseline_revision}`: `layer_{baselines['same_task_best_layer']}` with avg `{baselines['same_task_best_avg_main']:.10f}`"
        )
        lines.append(
            f"- gap vs same-task best: `{avg_main - baselines['same_task_best_avg_main']:+.10f}`"
        )
        lines.append(
            f"- same-task last layer at `{config.baseline_revision}`: `layer_{baselines['same_task_last_layer']}` with avg `{baselines['same_task_last_layer_avg_main']:.10f}`"
        )
        lines.append(
            f"- gap vs same-task last layer: `{avg_main - baselines['same_task_last_layer_avg_main']:+.10f}`"
        )
    if baselines.get("full_table_best_avg_main") is not None:
        lines.append(
            f"- full-table best layer at `{config.baseline_revision}`: `layer_{baselines['full_table_best_layer']}` with avg `{baselines['full_table_best_avg_main']:.10f}`"
        )
        lines.append(
            f"- gap vs full-table best: `{avg_main - baselines['full_table_best_avg_main']:+.10f}`"
        )
        lines.append(
            f"- full-table last layer at `{config.baseline_revision}`: `layer_{baselines['full_table_last_layer']}` with avg `{baselines['full_table_last_layer_avg_main']:.10f}`"
        )
        lines.append(
            f"- gap vs full-table last layer: `{avg_main - baselines['full_table_last_layer_avg_main']:+.10f}`"
        )
    if tasks_missing:
        lines.extend([
            "",
            "## Dropped Tasks",
            f"- `{', '.join(tasks_missing)}`",
        ])
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    revisions = [x.strip() for x in args.expected_revisions.split(",") if x.strip()]
    candidate_layers = list(range(args.layer_start, args.layer_end + 1))
    task_names = [x.strip() for x in args.task_names.split(",") if x.strip()]
    task_to_dataset = TASK_TO_DATASET.copy()
    if task_names:
        missing = sorted(set(task_names) - set(task_to_dataset))
        if missing:
            raise SelectionError(f"Unknown task_names requested: {missing}")
        task_to_dataset = {k: task_to_dataset[k] for k in task_names}

    config = SelectorConfig(
        model_family=args.model_family,
        model_size=args.model_size,
        revisions=revisions,
        candidate_layers=candidate_layers,
        dispersion_layer=args.dispersion_layer,
        top_k=args.top_k,
        layer_score_mode=args.layer_score_mode,
        checkpoint_choice_mode=args.checkpoint_choice_mode,
        dispersion_stability_lambda=args.dispersion_stability_lambda,
        entropy_stability_mu=args.entropy_stability_mu,
        pooling_method=args.pooling_method,
        dispersion_num_samples=args.dispersion_num_samples,
        sample_seed=args.sample_seed,
        dispersion_num_pairs=args.dispersion_num_pairs,
        batch_size=args.batch_size,
        max_sample_length=args.max_sample_length,
        baseline_revision=args.baseline_revision,
    )

    output_dir = os.path.join(
        args.output_root,
        f"pythia410m_factorized_dispersion_entropy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)

    cache_task_texts_path = args.cache_task_texts_path.strip() or os.path.join(output_dir, "sampled_task_texts.json")
    task_texts = load_sampled_task_texts(
        task_to_dataset,
        args.dispersion_num_samples,
        cache_task_texts_path,
        args.sample_seed,
    )
    usable_tasks = {task: texts for task, texts in task_texts.items() if texts}
    if not usable_tasks:
        raise SelectionError("No task texts were available for dispersion computation.")

    dispersion_df = compute_task_checkpoint_dispersion(usable_tasks, revisions, config, args.device_map)
    task_layer_df, revision_roots = build_task_checkpoint_layer_table(
        args.entropy_root,
        args.main_runs_root,
        revisions,
        {task: task_to_dataset[task] for task in usable_tasks},
        candidate_layers,
    )

    usable_revisions = sorted(set(dispersion_df["revision"]).intersection(task_layer_df["revision"]), key=_parse_revision_order_key)
    dispersion_df = dispersion_df[dispersion_df["revision"].isin(usable_revisions)].copy()
    task_layer_df = task_layer_df[task_layer_df["revision"].isin(usable_revisions)].copy()

    dispersion_counts = dispersion_df.groupby("task_name")["revision"].nunique()
    entropy_counts = task_layer_df.groupby("task_name")["revision"].nunique()
    complete_tasks = sorted(
        task for task in usable_tasks
        if dispersion_counts.get(task, 0) == len(usable_revisions) and entropy_counts.get(task, 0) == len(usable_revisions)
    )
    missing_tasks = sorted(set(usable_tasks) - set(complete_tasks))

    dispersion_df = dispersion_df[dispersion_df["task_name"].isin(complete_tasks)].copy()
    task_layer_df = task_layer_df[task_layer_df["task_name"].isin(complete_tasks)].copy()

    dispersion_df = add_dispersion_stability(dispersion_df, usable_revisions, config.dispersion_stability_lambda)
    task_layer_df = score_layers_within_checkpoint(task_layer_df, config.layer_score_mode)
    task_layer_df = add_entropy_stability(task_layer_df, usable_revisions, config.entropy_stability_mu)
    chosen_layers_df = choose_layers_per_checkpoint(task_layer_df)
    selected_pairs = choose_pairs(dispersion_df, chosen_layers_df, config.top_k, config.checkpoint_choice_mode)

    baselines = compute_baselines(task_layer_df, config.baseline_revision, selected_pairs["task_name"], args.main_runs_root, candidate_layers)

    dispersion_df.to_csv(os.path.join(output_dir, "task_checkpoint_dispersion.csv"), index=False)
    task_layer_df.to_csv(os.path.join(output_dir, "task_checkpoint_layer_scores.csv"), index=False)
    chosen_layers_df.to_csv(os.path.join(output_dir, "task_checkpoint_chosen_layers.csv"), index=False)
    selected_pairs.to_csv(os.path.join(output_dir, "selected_pairs.csv"), index=False)

    summary_payload = {
        "config": {
            "model_family": config.model_family,
            "model_size": config.model_size,
            "revisions": usable_revisions,
            "candidate_layers": candidate_layers,
            "dispersion_layer": config.dispersion_layer,
            "top_k": config.top_k,
            "layer_score_mode": config.layer_score_mode,
            "checkpoint_choice_mode": config.checkpoint_choice_mode,
            "dispersion_stability_lambda": config.dispersion_stability_lambda,
            "entropy_stability_mu": config.entropy_stability_mu,
            "pooling_method": config.pooling_method,
            "dispersion_num_samples": config.dispersion_num_samples,
            "sample_seed": config.sample_seed,
            "dispersion_num_pairs": config.dispersion_num_pairs,
            "batch_size": config.batch_size,
            "max_sample_length": config.max_sample_length,
            "baseline_revision": config.baseline_revision,
        },
        "num_tasks_selected": int(len(selected_pairs)),
        "selected_avg_main_score": float(selected_pairs["selected_main_score"].mean()),
        "selected_revision_counts": selected_pairs["selected_revision"].value_counts().to_dict(),
        "selected_layer_counts": selected_pairs["selected_layer"].value_counts().sort_index().to_dict(),
        "tasks_missing": missing_tasks,
        "revision_main_roots": revision_roots,
        "baselines": baselines,
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    write_summary_markdown(output_dir, config, selected_pairs, baselines, usable_revisions, missing_tasks)

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
