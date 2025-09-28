"""
Select a single global (checkpoint, layer) pair for a Pythia model using
label-free geometry computed on a fixed unlabeled text pool.

This script implements the proposal in
"Unsupervised Identification of a Single Best Checkpoint–Layer Pair in Pythia":

1. Build one balanced unlabeled pool from MTEB task texts.
2. For every candidate (revision, layer), compute representation geometry directly
   at that deployed layer.
3. Support several deterministic global selectors:
   - dispersion_only
   - dispersion_entropy_tiebreak
   - alignment_uniformity
   - alignment_uniformity_entropy_tiebreak
   - two_stage
   - two_stage_alignment_uniformity
   - two_stage_dispersion_only
4. Optionally regularize dispersion by temporal stability across neighboring
   checkpoints.
5. Evaluate only by looking up already-computed downstream scores.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments" / "results"
DEFAULT_RERUNS_ROOT = REPO_ROOT / "experiments" / "results_reruns"

import numpy as np
import pandas as pd
import torch

from scripts.analysis.select_taskwise_pairs_factorized import (
    TASK_TO_DATASET,
    SelectionError,
    _parse_revision_order_key,
    _revision_seed_value,
    _rank_asc,
    _rank_desc,
    _load_main_scores_for_layer,
    build_task_checkpoint_layer_table,
    compute_dispersion,
    find_main_revision_root,
    load_sampled_task_texts,
    score_layers_within_checkpoint,
)


SUPPORTED_RULES = [
    "dispersion_only",
    "dispersion_entropy_tiebreak",
    "alignment_uniformity",
    "alignment_uniformity_entropy_tiebreak",
    "pass_metric",
    "two_stage",
    "two_stage_auto",
    "two_stage_alignment_uniformity",
    "two_stage_dispersion_only",
]

TWO_STAGE_RULES = {
    "two_stage",
    "two_stage_auto",
    "two_stage_alignment_uniformity",
    "two_stage_dispersion_only",
}

RERANKER_FEATURE_SPECS = {
    "dispersion_stable": {"ascending": False},
    "alignment": {"ascending": True},
    "uniformity_score": {"ascending": False},
    "entropy_tiebreak_score": {"ascending": True},
    "top_pc_dominance": {"ascending": True},
    "effective_rank": {"ascending": False},
    "rankme": {"ascending": False},
    "spectral_slope": {"ascending": False},
    "participation_ratio": {"ascending": False},
    "knn_aug_stability": {"ascending": False},
    "checkpoint_neighbor_stability": {"ascending": False},
    "layer_neighbor_stability": {"ascending": False},
    "pass_phase_score": {"ascending": False},
    "pass_volatility": {"ascending": True},
    "pass_score": {"ascending": False},
}

DEFAULT_RERANKER_FEATURES = [
    "dispersion_stable",
    "alignment",
    "uniformity_score",
    "entropy_tiebreak_score",
    "top_pc_dominance",
    "effective_rank",
    "knn_aug_stability",
    "checkpoint_neighbor_stability",
    "layer_neighbor_stability",
]

DEFAULT_SELECTION_RULES = [
    "dispersion_only",
    "dispersion_entropy_tiebreak",
    "alignment_uniformity",
    "alignment_uniformity_entropy_tiebreak",
    "pass_metric",
    "two_stage",
]

PASS_RERANKER_FEATURES = {
    "rankme",
    "spectral_slope",
    "pass_phase_score",
    "pass_volatility",
    "pass_score",
}


def _model_slug(model_family: str, model_size: str) -> str:
    family = "".join(ch.lower() for ch in model_family if ch.isalnum())
    size = "".join(ch.lower() for ch in model_size if ch.isalnum())
    return f"{family}{size}"


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
    parser.add_argument("--pooling_method", type=str, default="mean")
    parser.add_argument("--pool_samples_per_task", type=int, default=32)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--dispersion_num_pairs", type=int, default=50000)
    parser.add_argument("--uniformity_num_pairs", type=int, default=50000)
    parser.add_argument("--uniformity_temperature", type=float, default=2.0)
    parser.add_argument("--alignment_quantile", type=float, default=0.30)
    parser.add_argument("--top_m_pairs", type=int, default=10)
    parser.add_argument("--reranker_shortlist_size", type=int, default=20)
    parser.add_argument("--reranker_knn_k", type=int, default=10)
    parser.add_argument("--reranker_analysis_num_texts", type=int, default=256)
    parser.add_argument(
        "--reranker_fusion",
        type=str,
        default="rrf",
        choices=["rrf", "borda"],
    )
    parser.add_argument("--reranker_rrf_k", type=int, default=60)
    parser.add_argument(
        "--reranker_features",
        type=str,
        default=",".join(DEFAULT_RERANKER_FEATURES),
        help="Comma-separated shortlist reranker features drawn from: " + ", ".join(RERANKER_FEATURE_SPECS),
    )
    parser.add_argument("--pass_tau_rank", type=float, default=0.01)
    parser.add_argument("--pass_tau_spectral", type=float, default=0.1)
    parser.add_argument("--pass_weight_dispersion", type=float, default=1.0)
    parser.add_argument("--pass_weight_uniformity", type=float, default=0.75)
    parser.add_argument("--pass_weight_alignment", type=float, default=0.75)
    parser.add_argument("--pass_weight_phase", type=float, default=1.0)
    parser.add_argument("--pass_weight_volatility", type=float, default=0.5)
    parser.add_argument("--stability_lambda", type=float, default=0.0)
    parser.add_argument(
        "--entropy_tiebreak_mode",
        type=str,
        default="minmax_sum",
        choices=["dataset_entropy", "dime", "infonce", "rank_sum", "zsum", "minmax_sum"],
    )
    parser.add_argument(
        "--selection_rules",
        type=str,
        default=",".join(DEFAULT_SELECTION_RULES),
        help="Comma-separated list from: " + ", ".join(SUPPORTED_RULES),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_sample_length", type=int, default=512)
    parser.add_argument("--baseline_revision", type=str, default="step143000")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument(
        "--task_names",
        type=str,
        default="",
        help="Optional comma-separated subset of task names to use in the global pool.",
    )
    parser.add_argument(
        "--cache_task_texts_path",
        type=str,
        default="",
        help="Optional JSON cache for sampled per-task texts.",
    )
    return parser.parse_args()


def _cache_payload_if_compatible(
    cache_path: str,
    requested_tasks: Iterable[str],
    min_samples_per_task: int,
    sample_seed: int,
) -> dict[str, list[str]] | None:
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    meta: dict[str, Any] = {}
    if isinstance(payload, dict) and "task_texts" in payload:
        meta = payload.get("meta", {}) or {}
        payload = payload["task_texts"]

    if not isinstance(payload, dict):
        return None

    cached_seed = meta.get("sample_seed")
    if cached_seed is not None and int(cached_seed) != int(sample_seed):
        return None

    normalized_payload = {str(k): [str(x) for x in v] for k, v in payload.items()}
    requested_tasks = [str(task_name) for task_name in requested_tasks]
    if any(task_name not in normalized_payload for task_name in requested_tasks):
        return None
    if any(len(normalized_payload[task_name]) < int(min_samples_per_task) for task_name in requested_tasks):
        return None
    return normalized_payload


def _discover_task_text_cache(
    search_roots: Iterable[str],
    requested_tasks: Iterable[str],
    min_samples_per_task: int,
    sample_seed: int,
) -> str | None:
    candidate_paths: list[str] = []
    seen: set[str] = set()
    for root in search_roots:
        if not root:
            continue
        abs_root = os.path.abspath(root)
        if not os.path.isdir(abs_root):
            continue
        pattern = os.path.join(abs_root, "**", "*task_texts*.json")
        for candidate in glob.glob(pattern, recursive=True):
            if candidate in seen or not os.path.isfile(candidate):
                continue
            seen.add(candidate)
            candidate_paths.append(candidate)

    candidate_paths.sort(key=os.path.getmtime, reverse=True)
    for candidate in candidate_paths:
        if _cache_payload_if_compatible(
            candidate,
            requested_tasks=requested_tasks,
            min_samples_per_task=min_samples_per_task,
            sample_seed=sample_seed,
        ) is not None:
            return candidate
    return None


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return embeddings / norms


def _compute_alignment(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> float:
    if embeddings_a.shape != embeddings_b.shape:
        raise ValueError(f"Alignment views must have the same shape, got {embeddings_a.shape} vs {embeddings_b.shape}")
    if embeddings_a.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings_a.shape}")
    a = _normalize_rows(embeddings_a)
    b = _normalize_rows(embeddings_b)
    squared_dist = np.sum((a - b) ** 2, axis=1)
    return float(np.mean(squared_dist))


def _compute_uniformity_loss(embeddings: np.ndarray, num_pairs: int, temperature: float, seed: int) -> float:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    num_samples = int(embeddings.shape[0])
    if num_samples < 2:
        return float("nan")

    normalized = _normalize_rows(embeddings)
    total_pairs = (num_samples * (num_samples - 1)) // 2
    if total_pairs <= num_pairs:
        diffs = normalized[:, None, :] - normalized[None, :, :]
        sq_dists = np.sum(diffs * diffs, axis=-1)
        iu = np.triu_indices(num_samples, k=1)
        sampled = sq_dists[iu]
    else:
        rng = np.random.default_rng(seed)
        left = rng.integers(0, num_samples, size=num_pairs, endpoint=False)
        right = rng.integers(0, num_samples - 1, size=num_pairs, endpoint=False)
        right = np.where(right >= left, right + 1, right)
        diffs = normalized[left] - normalized[right]
        sampled = np.sum(diffs * diffs, axis=1)

    return float(np.log(np.mean(np.exp(-temperature * sampled))))


def _choose_analysis_indices(num_texts: int, max_texts: int, seed: int) -> np.ndarray:
    if max_texts <= 0 or max_texts >= num_texts:
        return np.arange(num_texts, dtype=int)
    rng = np.random.default_rng(seed)
    selected = rng.choice(num_texts, size=max_texts, replace=False)
    return np.sort(selected.astype(int))


def _safe_singular_values(matrix: np.ndarray) -> np.ndarray | None:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return None
    if not np.isfinite(matrix).all():
        return None
    try:
        singular_values = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    if singular_values.size == 0 or not np.isfinite(singular_values).all():
        return None
    return singular_values


def _entropy_from_probabilities(probabilities: np.ndarray) -> float:
    positive = probabilities[np.isfinite(probabilities) & (probabilities > 0)]
    if positive.size == 0:
        return float("nan")
    return float(-np.sum(positive * np.log(positive)))


def _fit_loglog_spectral_slope(eigenvalues: np.ndarray) -> float:
    positive = eigenvalues[np.isfinite(eigenvalues) & (eigenvalues > 0)]
    if positive.size < 2:
        return float("nan")
    ranks = np.arange(1, positive.size + 1, dtype=np.float64)
    x = np.log(ranks)
    y = np.log(positive.astype(np.float64))
    try:
        slope, _ = np.polyfit(x, y, deg=1)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return float("nan")
    alpha = -float(slope)
    return alpha if math.isfinite(alpha) else float("nan")


def _compute_spectral_stats(embeddings: np.ndarray) -> dict[str, float]:
    stats = {
        "top_pc_dominance": float("nan"),
        "effective_rank": float("nan"),
        "rankme": float("nan"),
        "spectral_slope": float("nan"),
        "participation_ratio": float("nan"),
    }

    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return stats

    raw_embeddings = np.asarray(embeddings, dtype=np.float64)
    centered_raw = raw_embeddings - raw_embeddings.mean(axis=0, keepdims=True)
    raw_singular_values = _safe_singular_values(centered_raw)
    if raw_singular_values is not None:
        positive_singular_values = raw_singular_values[raw_singular_values > 0]
        total_singular_value = float(np.sum(positive_singular_values))
        if total_singular_value > 0.0 and math.isfinite(total_singular_value):
            rankme_probs = positive_singular_values / total_singular_value
            rankme_entropy = _entropy_from_probabilities(rankme_probs)
            if math.isfinite(rankme_entropy):
                stats["rankme"] = float(np.exp(rankme_entropy))

        raw_eigenvalues = (raw_singular_values ** 2) / max(1, centered_raw.shape[0] - 1)
        stats["spectral_slope"] = _fit_loglog_spectral_slope(raw_eigenvalues)

    normalized_embeddings = _normalize_rows(raw_embeddings)
    centered_normalized = normalized_embeddings - normalized_embeddings.mean(axis=0, keepdims=True)
    normalized_singular_values = _safe_singular_values(centered_normalized)
    if normalized_singular_values is None:
        return stats

    eigenvalues = (normalized_singular_values ** 2) / max(1, centered_normalized.shape[0] - 1)
    total = float(np.sum(eigenvalues))
    if total <= 0.0 or not math.isfinite(total):
        return stats

    stats["top_pc_dominance"] = float(eigenvalues[0] / total)
    probs = eigenvalues / total
    spectral_entropy = _entropy_from_probabilities(probs)
    if math.isfinite(spectral_entropy):
        stats["effective_rank"] = float(np.exp(spectral_entropy))

    denom = float(np.sum(eigenvalues ** 2))
    if denom > 0.0 and math.isfinite(denom):
        stats["participation_ratio"] = float((total ** 2) / denom)
    return stats


def _topk_neighbor_indices(embeddings: np.ndarray, k: int) -> np.ndarray:
    num_samples = int(embeddings.shape[0])
    if num_samples <= 1:
        return np.empty((num_samples, 0), dtype=int)
    k = max(0, min(int(k), num_samples - 1))
    if k == 0:
        return np.empty((num_samples, 0), dtype=int)

    similarities = embeddings @ embeddings.T
    np.fill_diagonal(similarities, -np.inf)
    return np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]


def _compute_knn_augmentation_stability(
    embeddings: np.ndarray,
    augmented_embeddings: np.ndarray,
    k: int,
) -> float:
    if embeddings.shape != augmented_embeddings.shape:
        raise ValueError(f"kNN stability views must have the same shape, got {embeddings.shape} vs {augmented_embeddings.shape}")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    if embeddings.shape[0] < 2:
        return float("nan")

    normalized = _normalize_rows(embeddings)
    normalized_aug = _normalize_rows(augmented_embeddings)
    neighbors = _topk_neighbor_indices(normalized, k)
    neighbors_aug = _topk_neighbor_indices(normalized_aug, k)

    overlaps: list[float] = []
    for base, aug in zip(neighbors, neighbors_aug):
        if base.size == 0 or aug.size == 0:
            continue
        base_set = set(int(x) for x in base.tolist())
        aug_set = set(int(x) for x in aug.tolist())
        union_size = len(base_set | aug_set)
        if union_size == 0:
            continue
        overlaps.append(len(base_set & aug_set) / union_size)

    if not overlaps:
        return float("nan")
    return float(np.mean(overlaps))


def _center_embeddings(embeddings: np.ndarray) -> np.ndarray:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    return np.asarray(centered, dtype=np.float32)


def _compute_linear_cka(centered_x: np.ndarray, centered_y: np.ndarray) -> float:
    if centered_x.shape != centered_y.shape:
        raise ValueError(f"CKA inputs must match, got {centered_x.shape} vs {centered_y.shape}")
    if centered_x.ndim != 2 or centered_x.shape[0] < 2:
        return float("nan")

    x = np.asarray(centered_x, dtype=np.float64)
    y = np.asarray(centered_y, dtype=np.float64)
    cross_cov = x.T @ y
    x_cov = x.T @ x
    y_cov = y.T @ y
    numerator = float(np.sum(cross_cov * cross_cov))
    denom_x = float(np.sqrt(np.sum(x_cov * x_cov)))
    denom_y = float(np.sqrt(np.sum(y_cov * y_cov)))
    if numerator <= 0.0 or denom_x <= 0.0 or denom_y <= 0.0:
        return float("nan")
    return float(numerator / (denom_x * denom_y))


def add_neighbor_stability(
    pair_df: pd.DataFrame,
    pair_embeddings: dict[tuple[str, int], np.ndarray],
    revisions_in_order: list[str],
    candidate_layers: list[int],
) -> pd.DataFrame:
    out = pair_df.copy()
    revision_to_idx = {revision: idx for idx, revision in enumerate(revisions_in_order)}
    layer_to_idx = {int(layer): idx for idx, layer in enumerate(candidate_layers)}

    checkpoint_scores: list[float] = []
    checkpoint_counts: list[int] = []
    layer_scores: list[float] = []
    layer_counts: list[int] = []

    for row in out.itertuples(index=False):
        key = (str(row.revision), int(row.layer_idx))
        centered = pair_embeddings.get(key)

        cp_neighbors: list[float] = []
        layer_neighbors: list[float] = []
        if centered is not None:
            revision_idx = revision_to_idx.get(str(row.revision))
            layer_idx = layer_to_idx.get(int(row.layer_idx))

            if revision_idx is not None:
                for offset in (-1, 1):
                    neighbor_idx = revision_idx + offset
                    if 0 <= neighbor_idx < len(revisions_in_order):
                        neighbor_key = (revisions_in_order[neighbor_idx], int(row.layer_idx))
                        neighbor = pair_embeddings.get(neighbor_key)
                        if neighbor is not None:
                            score = _compute_linear_cka(centered, neighbor)
                            if math.isfinite(score):
                                cp_neighbors.append(float(score))

            if layer_idx is not None:
                for offset in (-1, 1):
                    neighbor_idx = layer_idx + offset
                    if 0 <= neighbor_idx < len(candidate_layers):
                        neighbor_key = (str(row.revision), int(candidate_layers[neighbor_idx]))
                        neighbor = pair_embeddings.get(neighbor_key)
                        if neighbor is not None:
                            score = _compute_linear_cka(centered, neighbor)
                            if math.isfinite(score):
                                layer_neighbors.append(float(score))

        checkpoint_scores.append(float(np.mean(cp_neighbors)) if cp_neighbors else float("nan"))
        checkpoint_counts.append(int(len(cp_neighbors)))
        layer_scores.append(float(np.mean(layer_neighbors)) if layer_neighbors else float("nan"))
        layer_counts.append(int(len(layer_neighbors)))

    out["checkpoint_neighbor_stability"] = checkpoint_scores
    out["checkpoint_neighbor_count"] = checkpoint_counts
    out["layer_neighbor_stability"] = layer_scores
    out["layer_neighbor_count"] = layer_counts
    return out


def _augment_texts(texts: list[str], seed: int) -> list[str]:
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.word as naw
    import nlpaug.flow as naf

    aug = naf.Sequential([
        naw.SplitAug(),
        nac.RandomCharAug(),
        nac.KeyboardAug(),
    ])

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        augmented = [str(aug.augment(x, n=1)) for x in texts]
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
    return augmented


def _cap_task_texts(task_texts: dict[str, list[str]], max_per_task: int, seed: int) -> dict[str, list[str]]:
    capped: dict[str, list[str]] = {}
    for task_idx, (task_name, texts) in enumerate(task_texts.items()):
        if len(texts) <= max_per_task:
            capped[task_name] = list(texts)
            continue
        rng = np.random.default_rng(seed + task_idx)
        chosen_idx = rng.choice(len(texts), size=max_per_task, replace=False)
        chosen_idx = np.sort(chosen_idx)
        capped[task_name] = [texts[int(i)] for i in chosen_idx]
    return capped


def _build_global_pool(task_texts: dict[str, list[str]]) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    texts: list[str] = []
    for task_name, values in task_texts.items():
        for idx, text in enumerate(values):
            texts.append(text)
            rows.append(
                {
                    "task_name": task_name,
                    "task_local_index": int(idx),
                    "text": text,
                }
            )
    if not texts:
        raise SelectionError("The global unlabeled pool is empty.")
    return texts, pd.DataFrame(rows)


def compute_pair_geometry(
    global_texts: list[str],
    revisions: list[str],
    candidate_layers: list[int],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pair_embeddings: dict[tuple[str, int], np.ndarray] = {}
    augmented_texts: list[str] | None = None
    selected_rules = {x.strip() for x in args.selection_rules.split(",") if x.strip()}
    requested_reranker_features = {x.strip() for x in args.reranker_features.split(",") if x.strip()}
    needs_two_stage = any(rule in TWO_STAGE_RULES for rule in selected_rules)
    needs_pass = ("pass_metric" in selected_rules) or (
        needs_two_stage and bool(PASS_RERANKER_FEATURES & requested_reranker_features)
    )
    needs_spectral_analysis = needs_two_stage or needs_pass
    needs_alignment = needs_two_stage or any(rule.startswith("alignment_uniformity") for rule in selected_rules)
    needs_alignment = needs_alignment or needs_pass
    analysis_indices = _choose_analysis_indices(
        len(global_texts),
        args.reranker_analysis_num_texts,
        seed=args.sample_seed + 2026,
    ) if needs_spectral_analysis else None
    if needs_alignment:
        augmented_texts = _augment_texts(global_texts, seed=args.sample_seed + 17)

    from experiments.utils.model_definitions.text_automodel_wrapper import (
        TextLayerwiseAutoModelWrapper,
        TextModelSpecifications,
    )

    max_layer = max(candidate_layers)
    for revision in revisions:
        model_specs = TextModelSpecifications(args.model_family, args.model_size, revision)
        model = TextLayerwiseAutoModelWrapper(
            model_specs,
            device_map=args.device_map,
            evaluation_layer_idx=max_layer,
        )
        model.print_loading_message()
        _, layerwise_embeddings = model.encode(
            global_texts,
            batch_size=args.batch_size,
            num_workers=0,
            pooling_method=args.pooling_method,
            max_sample_length=args.max_sample_length,
            verbose=False,
            return_layerwise_encodings=True,
        )

        if needs_alignment:
            _, layerwise_aug_embeddings = model.encode(
                augmented_texts,
                batch_size=args.batch_size,
                num_workers=0,
                pooling_method=args.pooling_method,
                max_sample_length=args.max_sample_length,
                verbose=False,
                return_layerwise_encodings=True,
            )
        else:
            layerwise_aug_embeddings = None

        revision_seed_value = _revision_seed_value(revision)
        for layer_idx in candidate_layers:
            if layer_idx >= layerwise_embeddings.shape[0]:
                continue
            layer_embeddings = np.asarray(layerwise_embeddings[layer_idx], dtype=float)
            pair_seed = int(args.sample_seed + revision_seed_value + 10000 * layer_idx)
            dispersion = compute_dispersion(layer_embeddings, num_pairs=args.dispersion_num_pairs, seed=pair_seed)

            row = {
                "revision": revision,
                "layer_idx": int(layer_idx),
                "num_pool_texts": int(layer_embeddings.shape[0]),
                "embedding_dim": int(layer_embeddings.shape[1]),
                "dispersion": float(dispersion),
                "dispersion_pair_seed": pair_seed,
            }

            if layerwise_aug_embeddings is not None:
                augmented_embeddings = np.asarray(layerwise_aug_embeddings[layer_idx], dtype=float)
                alignment = _compute_alignment(layer_embeddings, augmented_embeddings)
                uniformity_seed = int(pair_seed + 1_000_000)
                uniformity_loss = _compute_uniformity_loss(
                    layer_embeddings,
                    num_pairs=args.uniformity_num_pairs,
                    temperature=args.uniformity_temperature,
                    seed=uniformity_seed,
                )
                row["alignment"] = float(alignment)
                row["uniformity_loss"] = float(uniformity_loss)
                row["uniformity_score"] = float(-uniformity_loss)
                row["uniformity_pair_seed"] = uniformity_seed
            else:
                row["alignment"] = np.nan
                row["uniformity_loss"] = np.nan
                row["uniformity_score"] = np.nan
                row["uniformity_pair_seed"] = np.nan

            if analysis_indices is not None:
                analysis_embeddings = layer_embeddings[analysis_indices]
                spectral_stats = _compute_spectral_stats(analysis_embeddings)
                row["top_pc_dominance"] = float(spectral_stats["top_pc_dominance"]) if math.isfinite(spectral_stats["top_pc_dominance"]) else np.nan
                row["effective_rank"] = float(spectral_stats["effective_rank"]) if math.isfinite(spectral_stats["effective_rank"]) else np.nan
                row["rankme"] = float(spectral_stats["rankme"]) if math.isfinite(spectral_stats["rankme"]) else np.nan
                row["spectral_slope"] = float(spectral_stats["spectral_slope"]) if math.isfinite(spectral_stats["spectral_slope"]) else np.nan
                row["participation_ratio"] = float(spectral_stats["participation_ratio"]) if math.isfinite(spectral_stats["participation_ratio"]) else np.nan

                if needs_two_stage:
                    pair_embeddings[(revision, int(layer_idx))] = _center_embeddings(analysis_embeddings)

                if needs_two_stage and layerwise_aug_embeddings is not None:
                    knn_aug_stability = _compute_knn_augmentation_stability(
                        layer_embeddings,
                        augmented_embeddings,
                        k=args.reranker_knn_k,
                    )
                    row["knn_aug_stability"] = float(knn_aug_stability) if math.isfinite(knn_aug_stability) else np.nan
                else:
                    row["knn_aug_stability"] = np.nan
            else:
                row["top_pc_dominance"] = np.nan
                row["effective_rank"] = np.nan
                row["rankme"] = np.nan
                row["spectral_slope"] = np.nan
                row["participation_ratio"] = np.nan
                row["knn_aug_stability"] = np.nan

            rows.append(row)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise SelectionError("No pair geometry rows were produced.")
    pair_geometry_df = pd.DataFrame(rows)
    if needs_two_stage and pair_embeddings:
        pair_geometry_df = add_neighbor_stability(pair_geometry_df, pair_embeddings, revisions, candidate_layers)
    else:
        pair_geometry_df["checkpoint_neighbor_stability"] = np.nan
        pair_geometry_df["checkpoint_neighbor_count"] = 0
        pair_geometry_df["layer_neighbor_stability"] = np.nan
        pair_geometry_df["layer_neighbor_count"] = 0
    return pair_geometry_df


def load_pair_avg_main_table(
    main_runs_root: str,
    revisions: Iterable[str],
    candidate_layers: Iterable[int],
) -> pd.DataFrame:
    candidate_layers = list(candidate_layers)
    rows: list[dict[str, Any]] = []
    for revision in revisions:
        main_root = find_main_revision_root(main_runs_root, revision, min(candidate_layers), max(candidate_layers))
        if main_root is None:
            continue

        csv_path = os.path.join(main_root, "average_main_score", "avg_main_score_by_layer.csv")
        if os.path.isfile(csv_path):
            table = pd.read_csv(csv_path)
            if table.empty:
                continue
            table["layer_idx"] = table["layer"].astype(str).str.replace("layer_", "", regex=False).astype(int)
            table = table[table["layer_idx"].isin(candidate_layers)].copy()
            for _, row in table.iterrows():
                rows.append(
                    {
                        "revision": revision,
                        "layer_idx": int(row["layer_idx"]),
                        "avg_main_score": float(row["avg_main_score"]),
                        "num_tasks_eval": int(row.get("num_tasks", 0)),
                        "avg_main_source": csv_path,
                    }
                )
            continue

        for layer_idx in candidate_layers:
            scores = _load_main_scores_for_layer(main_root, layer_idx)
            if not scores:
                continue
            rows.append(
                {
                    "revision": revision,
                    "layer_idx": int(layer_idx),
                    "avg_main_score": float(np.mean(list(scores.values()))),
                    "num_tasks_eval": int(len(scores)),
                    "avg_main_source": os.path.join(main_root, "mteb", f"layer_{layer_idx}"),
                }
            )

    if not rows:
        raise SelectionError("No average-main-score rows were found for the requested revisions/layers.")
    return pd.DataFrame(rows)


def build_pair_entropy_tiebreak_table(
    args: argparse.Namespace,
    revisions: list[str],
    task_to_dataset: dict[str, str],
    candidate_layers: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    task_layer_df, _ = build_task_checkpoint_layer_table(
        args.entropy_root,
        args.main_runs_root,
        revisions,
        task_to_dataset,
        candidate_layers,
    )
    task_layer_df = task_layer_df[task_layer_df["revision"].isin(revisions)].copy()
    if task_layer_df.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    task_layer_df = score_layers_within_checkpoint(task_layer_df, args.entropy_tiebreak_mode)
    valid_rows = task_layer_df.dropna(subset=["layer_score"]).copy()
    if valid_rows.empty:
        return pd.DataFrame(), task_layer_df, []

    counts = valid_rows.groupby("task_name")["revision"].nunique()
    complete_tasks = sorted(
        task for task in task_to_dataset
        if counts.get(task, 0) == len(revisions)
    )
    valid_rows = valid_rows[valid_rows["task_name"].isin(complete_tasks)].copy()
    if valid_rows.empty:
        return pd.DataFrame(), task_layer_df, complete_tasks

    pair_scores = (
        valid_rows.groupby(["revision", "layer_idx"], as_index=False)
        .agg(
            entropy_tiebreak_score=("layer_score", "mean"),
            entropy_tiebreak_score_median=("layer_score", "median"),
            entropy_tiebreak_support_tasks=("task_name", "nunique"),
            dataset_entropy_mean=("dataset_entropy", "mean"),
            infonce_for_min_mean=("infonce_for_min", "mean"),
            dime_mean=("dime", "mean"),
        )
    )
    return pair_scores, valid_rows, complete_tasks


def add_temporal_stability(pair_df: pd.DataFrame, revisions_in_order: list[str], lam: float) -> pd.DataFrame:
    out = pair_df.copy()
    prev_by_revision: dict[str, str | None] = {}
    previous: str | None = None
    for revision in revisions_in_order:
        prev_by_revision[revision] = previous
        previous = revision

    deltas: list[float] = []
    stable_scores: list[float] = []
    for _, row in out.iterrows():
        prev_revision = prev_by_revision.get(str(row["revision"]))
        if prev_revision is None:
            delta = 0.0
        else:
            prev_match = out[
                (out["revision"] == prev_revision) &
                (out["layer_idx"] == row["layer_idx"])
            ]
            if prev_match.empty or pd.isna(prev_match.iloc[0]["dispersion"]):
                delta = 0.0
            else:
                delta = abs(float(row["dispersion"]) - float(prev_match.iloc[0]["dispersion"]))
        deltas.append(float(delta))
        stable_scores.append(float(row["dispersion"]) - lam * float(delta))

    out["dispersion_delta_prev"] = deltas
    out["dispersion_stable"] = stable_scores
    return out


def _stable_sigmoid(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    if value >= 0.0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def _robust_standardize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float))
    finite_values = numeric[finite_mask]
    if finite_values.empty:
        return pd.Series(np.zeros(len(numeric), dtype=float), index=numeric.index)

    median = float(finite_values.median())
    mad = float(np.median(np.abs(finite_values.to_numpy(dtype=float) - median)))
    if not math.isfinite(mad) or mad <= 1e-12:
        standardized = pd.Series(np.zeros(len(numeric), dtype=float), index=numeric.index)
    else:
        standardized = (numeric - median) / (mad + 1e-12)
    return standardized.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def add_pass_metric(pair_df: pd.DataFrame, revisions_in_order: list[str], args: argparse.Namespace) -> pd.DataFrame:
    out = pair_df.copy()
    prev_by_revision: dict[str, str | None] = {}
    previous: str | None = None
    for revision in revisions_in_order:
        prev_by_revision[revision] = previous
        previous = revision

    value_lookup: dict[tuple[str, int], dict[str, float]] = {}
    for row in out.itertuples(index=False):
        value_lookup[(str(row.revision), int(row.layer_idx))] = {
            "rankme": float(row.rankme) if pd.notna(row.rankme) else float("nan"),
            "spectral_slope": float(row.spectral_slope) if pd.notna(row.spectral_slope) else float("nan"),
        }

    tau_rank = max(float(args.pass_tau_rank), 1e-12)
    tau_spectral = max(float(args.pass_tau_spectral), 1e-12)

    rankme_deltas: list[float] = []
    spectral_deltas: list[float] = []
    phase_scores: list[float] = []
    volatilities: list[float] = []
    availability: list[bool] = []

    for row in out.itertuples(index=False):
        key = (str(row.revision), int(row.layer_idx))
        current_values = value_lookup.get(key, {})
        current_rankme = float(current_values.get("rankme", float("nan")))
        current_spectral = float(current_values.get("spectral_slope", float("nan")))

        delta_rankme = 0.0
        delta_spectral = 0.0
        prev_revision = prev_by_revision.get(str(row.revision))
        if prev_revision is not None:
            prev_values = value_lookup.get((prev_revision, int(row.layer_idx)), {})
            prev_rankme = float(prev_values.get("rankme", float("nan")))
            prev_spectral = float(prev_values.get("spectral_slope", float("nan")))
            if math.isfinite(current_rankme) and math.isfinite(prev_rankme):
                delta_rankme = current_rankme - prev_rankme
            if math.isfinite(current_spectral) and math.isfinite(prev_spectral):
                delta_spectral = current_spectral - prev_spectral

        dispersion_delta = float(row.dispersion_delta_prev) if pd.notna(row.dispersion_delta_prev) else 0.0
        phase_score = _stable_sigmoid(delta_spectral / tau_spectral) * _stable_sigmoid((-delta_rankme) / tau_rank)
        volatility = abs(dispersion_delta) + abs(delta_rankme) + abs(delta_spectral)

        rankme_deltas.append(float(delta_rankme))
        spectral_deltas.append(float(delta_spectral))
        phase_scores.append(float(phase_score))
        volatilities.append(float(volatility))
        availability.append(
            bool(
                pd.notna(row.dispersion)
                and pd.notna(row.uniformity_score)
                and pd.notna(row.alignment)
                and math.isfinite(current_rankme)
                and math.isfinite(current_spectral)
            )
        )

    out["rankme_delta_prev"] = rankme_deltas
    out["spectral_slope_delta_prev"] = spectral_deltas
    out["pass_phase_score"] = phase_scores
    out["pass_volatility"] = volatilities
    out["pass_available"] = availability

    out["pass_dispersion_norm"] = _robust_standardize(out["dispersion"])
    out["pass_uniformity_norm"] = _robust_standardize(out["uniformity_score"])
    out["pass_alignment_norm"] = _robust_standardize(out["alignment"])
    out["pass_phase_norm"] = _robust_standardize(out["pass_phase_score"])
    out["pass_volatility_norm"] = _robust_standardize(out["pass_volatility"])

    out["pass_score"] = (
        float(args.pass_weight_dispersion) * out["pass_dispersion_norm"]
        + float(args.pass_weight_uniformity) * out["pass_uniformity_norm"]
        - float(args.pass_weight_alignment) * out["pass_alignment_norm"]
        + float(args.pass_weight_phase) * out["pass_phase_norm"]
        - float(args.pass_weight_volatility) * out["pass_volatility_norm"]
    )
    out.loc[~out["pass_available"], "pass_score"] = np.nan
    return out


def add_metric_ranks(pair_df: pd.DataFrame) -> pd.DataFrame:
    out = pair_df.copy()
    out["revision_step"] = out["revision"].map(lambda x: _parse_revision_order_key(str(x))[0])
    out["dispersion_rank_desc"] = _rank_desc(out["dispersion"])
    out["dispersion_stable_rank_desc"] = _rank_desc(out["dispersion_stable"])
    if out["uniformity_score"].notna().any():
        out["uniformity_rank_desc"] = _rank_desc(out["uniformity_score"].fillna(-np.inf))
    else:
        out["uniformity_rank_desc"] = np.nan
    if out["alignment"].notna().any():
        finite_alignment = out["alignment"].replace([np.inf, -np.inf], np.nan)
        out["alignment_rank_asc"] = _rank_asc(finite_alignment.fillna(np.inf))
    else:
        out["alignment_rank_asc"] = np.nan
    if out["entropy_tiebreak_score"].notna().any():
        finite_entropy = out["entropy_tiebreak_score"].replace([np.inf, -np.inf], np.nan)
        out["entropy_tiebreak_rank_asc"] = _rank_asc(finite_entropy.fillna(np.inf))
    else:
        out["entropy_tiebreak_rank_asc"] = np.nan
    if out["top_pc_dominance"].notna().any():
        finite_top_pc = out["top_pc_dominance"].replace([np.inf, -np.inf], np.nan)
        out["top_pc_dominance_rank_asc"] = _rank_asc(finite_top_pc.fillna(np.inf))
    else:
        out["top_pc_dominance_rank_asc"] = np.nan
    if out["effective_rank"].notna().any():
        finite_effective_rank = out["effective_rank"].replace([np.inf, -np.inf], np.nan)
        out["effective_rank_rank_desc"] = _rank_desc(finite_effective_rank.fillna(-np.inf))
    else:
        out["effective_rank_rank_desc"] = np.nan
    if out["rankme"].notna().any():
        finite_rankme = out["rankme"].replace([np.inf, -np.inf], np.nan)
        out["rankme_rank_desc"] = _rank_desc(finite_rankme.fillna(-np.inf))
    else:
        out["rankme_rank_desc"] = np.nan
    if out["spectral_slope"].notna().any():
        finite_spectral = out["spectral_slope"].replace([np.inf, -np.inf], np.nan)
        out["spectral_slope_rank_desc"] = _rank_desc(finite_spectral.fillna(-np.inf))
    else:
        out["spectral_slope_rank_desc"] = np.nan
    if out["participation_ratio"].notna().any():
        finite_participation = out["participation_ratio"].replace([np.inf, -np.inf], np.nan)
        out["participation_ratio_rank_desc"] = _rank_desc(finite_participation.fillna(-np.inf))
    else:
        out["participation_ratio_rank_desc"] = np.nan
    if out["knn_aug_stability"].notna().any():
        finite_knn = out["knn_aug_stability"].replace([np.inf, -np.inf], np.nan)
        out["knn_aug_stability_rank_desc"] = _rank_desc(finite_knn.fillna(-np.inf))
    else:
        out["knn_aug_stability_rank_desc"] = np.nan
    if out["checkpoint_neighbor_stability"].notna().any():
        finite_checkpoint = out["checkpoint_neighbor_stability"].replace([np.inf, -np.inf], np.nan)
        out["checkpoint_neighbor_stability_rank_desc"] = _rank_desc(finite_checkpoint.fillna(-np.inf))
    else:
        out["checkpoint_neighbor_stability_rank_desc"] = np.nan
    if out["layer_neighbor_stability"].notna().any():
        finite_layer = out["layer_neighbor_stability"].replace([np.inf, -np.inf], np.nan)
        out["layer_neighbor_stability_rank_desc"] = _rank_desc(finite_layer.fillna(-np.inf))
    else:
        out["layer_neighbor_stability_rank_desc"] = np.nan
    if out["pass_phase_score"].notna().any():
        finite_pass_phase = out["pass_phase_score"].replace([np.inf, -np.inf], np.nan)
        out["pass_phase_score_rank_desc"] = _rank_desc(finite_pass_phase.fillna(-np.inf))
    else:
        out["pass_phase_score_rank_desc"] = np.nan
    if out["pass_volatility"].notna().any():
        finite_pass_volatility = out["pass_volatility"].replace([np.inf, -np.inf], np.nan)
        out["pass_volatility_rank_asc"] = _rank_asc(finite_pass_volatility.fillna(np.inf))
    else:
        out["pass_volatility_rank_asc"] = np.nan
    if out["pass_score"].notna().any():
        finite_pass_score = out["pass_score"].replace([np.inf, -np.inf], np.nan)
        out["pass_score_rank_desc"] = _rank_desc(finite_pass_score.fillna(-np.inf))
    else:
        out["pass_score_rank_desc"] = np.nan
    if out["avg_main_score"].notna().any():
        out["avg_main_rank_desc"] = _rank_desc(out["avg_main_score"].fillna(-np.inf))
    else:
        out["avg_main_rank_desc"] = np.nan
    return out


def _resolve_two_stage_shortlist_rule(rule: str, model_size: str) -> str:
    if rule in {"two_stage_alignment_uniformity"}:
        return "alignment_uniformity"
    if rule in {"two_stage_dispersion_only"}:
        return "dispersion_only"
    normalized_size = "".join(ch.lower() for ch in model_size if ch.isalnum())
    return "dispersion_only" if normalized_size == "70m" else "alignment_uniformity"


def _build_shortlist_from_rule(
    pair_df: pd.DataFrame,
    shortlist_rule: str,
    shortlist_size: int,
    alignment_quantile: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {"shortlist_rule": shortlist_rule}

    if shortlist_rule == "dispersion_only":
        working = pair_df.dropna(subset=["dispersion_stable"]).copy()
        if working.empty:
            raise SelectionError("two-stage shortlist rule dispersion_only had no eligible candidates.")
        shortlist = working.sort_values(
            ["dispersion_stable", "dispersion", "revision_step", "layer_idx"],
            ascending=[False, False, True, True],
        ).head(shortlist_size).reset_index(drop=True)
        meta["selection_score_name"] = "dispersion_stable"
        meta["num_shortlist_candidates"] = int(len(shortlist))
        return shortlist, meta

    if shortlist_rule == "alignment_uniformity":
        working = pair_df.dropna(subset=["alignment", "uniformity_score", "dispersion_stable"]).copy()
        if working.empty:
            raise SelectionError("two-stage shortlist rule alignment_uniformity had no eligible candidates.")
        threshold = float(working["alignment"].quantile(alignment_quantile))
        feasible = working[working["alignment"] <= threshold].copy()
        if feasible.empty:
            feasible = working.nsmallest(max(1, shortlist_size), columns=["alignment"]).copy()
        shortlist = feasible.sort_values(
            ["uniformity_score", "dispersion_stable", "revision_step", "layer_idx"],
            ascending=[False, False, True, True],
        ).head(shortlist_size).reset_index(drop=True)
        meta["selection_score_name"] = "uniformity_score"
        meta["alignment_quantile"] = float(alignment_quantile)
        meta["alignment_threshold"] = threshold
        meta["num_feasible_candidates"] = int(len(feasible))
        meta["num_shortlist_candidates"] = int(len(shortlist))
        return shortlist, meta

    raise ValueError(f"Unsupported shortlist rule: {shortlist_rule}")


def _rank_feature_within_shortlist(values: pd.Series, ascending: bool) -> pd.Series:
    finite = values.replace([np.inf, -np.inf], np.nan)
    fill_value = np.inf if ascending else -np.inf
    return finite.fillna(fill_value).rank(method="average", ascending=ascending)


def _rerank_shortlist(
    shortlist: pd.DataFrame,
    reranker_features: list[str],
    reranker_fusion: str,
    reranker_rrf_k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reranked = shortlist.copy().reset_index(drop=True)
    if reranked.empty:
        raise SelectionError("The two-stage reranker received an empty shortlist.")
    reranked["shortlist_rank"] = np.arange(1, len(reranked) + 1)

    available_features: list[str] = []
    rank_cols: list[str] = []
    for feature_name in reranker_features:
        if feature_name not in reranked.columns:
            continue
        values = reranked[feature_name]
        if not values.notna().any():
            continue
        feature_rank_col = f"{feature_name}_shortlist_rank"
        reranked[feature_rank_col] = _rank_feature_within_shortlist(
            values,
            ascending=bool(RERANKER_FEATURE_SPECS[feature_name]["ascending"]),
        )
        available_features.append(feature_name)
        rank_cols.append(feature_rank_col)

    if not available_features:
        reranked["reranker_score"] = np.nan
        reranked["reranker_rank"] = reranked["shortlist_rank"]
        return reranked, {
            "reranker_feature_names": [],
            "reranker_fusion_method": reranker_fusion,
            "num_reranker_features_used": 0,
        }

    if reranker_fusion == "rrf":
        reranked["reranker_score"] = 0.0
        for rank_col in rank_cols:
            reranked["reranker_score"] += 1.0 / (float(reranker_rrf_k) + reranked[rank_col])
    elif reranker_fusion == "borda":
        shortlist_len = float(len(reranked))
        reranked["reranker_score"] = 0.0
        for rank_col in rank_cols:
            reranked["reranker_score"] += (shortlist_len + 1.0) - reranked[rank_col]
    else:
        raise ValueError(f"Unsupported reranker_fusion: {reranker_fusion}")

    reranked = reranked.sort_values(
        ["reranker_score", "shortlist_rank", "revision_step", "layer_idx"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    reranked["reranker_rank"] = np.arange(1, len(reranked) + 1)
    return reranked, {
        "reranker_feature_names": available_features,
        "reranker_fusion_method": reranker_fusion,
        "num_reranker_features_used": int(len(available_features)),
    }


def _select_rule(
    pair_df: pd.DataFrame,
    rule: str,
    model_size: str,
    top_m_pairs: int,
    reranker_shortlist_size: int,
    alignment_quantile: float,
    reranker_features: list[str],
    reranker_fusion: str,
    reranker_rrf_k: int,
) -> tuple[pd.Series, dict[str, Any], pd.DataFrame]:
    meta: dict[str, Any] = {}

    if rule == "dispersion_only":
        working = pair_df.dropna(subset=["dispersion_stable"]).copy()
        selected = working.sort_values(
            ["dispersion_stable", "dispersion", "revision_step", "layer_idx"],
            ascending=[False, False, True, True],
        ).iloc[0]
        meta["selection_score_name"] = "dispersion_stable"
        return selected, meta, pd.DataFrame()

    if rule == "dispersion_entropy_tiebreak":
        working = pair_df.dropna(subset=["dispersion_stable", "entropy_tiebreak_score"]).copy()
        candidates = working.sort_values(
            ["dispersion_stable", "dispersion", "revision_step", "layer_idx"],
            ascending=[False, False, True, True],
        ).head(top_m_pairs)
        selected = candidates.sort_values(
            ["entropy_tiebreak_score", "dispersion_stable", "revision_step", "layer_idx"],
            ascending=[True, False, True, True],
        ).iloc[0]
        meta["selection_score_name"] = "entropy_tiebreak_score"
        meta["num_top_m_candidates"] = int(len(candidates))
        return selected, meta, pd.DataFrame()

    if rule == "alignment_uniformity":
        working = pair_df.dropna(subset=["alignment", "uniformity_score", "dispersion_stable"]).copy()
        threshold = float(working["alignment"].quantile(alignment_quantile))
        feasible = working[working["alignment"] <= threshold].copy()
        if feasible.empty:
            feasible = working.nsmallest(max(1, top_m_pairs), columns=["alignment"]).copy()
        selected = feasible.sort_values(
            ["uniformity_score", "dispersion_stable", "revision_step", "layer_idx"],
            ascending=[False, False, True, True],
        ).iloc[0]
        meta["alignment_quantile"] = float(alignment_quantile)
        meta["alignment_threshold"] = threshold
        meta["num_feasible_candidates"] = int(len(feasible))
        meta["selection_score_name"] = "uniformity_score"
        return selected, meta, pd.DataFrame()

    if rule == "alignment_uniformity_entropy_tiebreak":
        working = pair_df.dropna(subset=["alignment", "uniformity_score", "entropy_tiebreak_score", "dispersion_stable"]).copy()
        threshold = float(working["alignment"].quantile(alignment_quantile))
        feasible = working[working["alignment"] <= threshold].copy()
        if feasible.empty:
            feasible = working.nsmallest(max(1, top_m_pairs), columns=["alignment"]).copy()
        candidates = feasible.sort_values(
            ["uniformity_score", "dispersion_stable", "revision_step", "layer_idx"],
            ascending=[False, False, True, True],
        ).head(top_m_pairs)
        selected = candidates.sort_values(
            ["entropy_tiebreak_score", "uniformity_score", "revision_step", "layer_idx"],
            ascending=[True, False, True, True],
        ).iloc[0]
        meta["alignment_quantile"] = float(alignment_quantile)
        meta["alignment_threshold"] = threshold
        meta["num_feasible_candidates"] = int(len(feasible))
        meta["num_top_m_candidates"] = int(len(candidates))
        meta["selection_score_name"] = "entropy_tiebreak_score"
        return selected, meta, pd.DataFrame()

    if rule == "pass_metric":
        working = pair_df.dropna(subset=["pass_score"]).copy()
        if working.empty:
            raise SelectionError("PASS metric had no eligible candidates. Ensure PASS components were computed.")
        selected = working.sort_values(
            ["pass_score", "dispersion", "revision_step", "layer_idx"],
            ascending=[False, False, True, True],
        ).iloc[0]
        meta["selection_score_name"] = "pass_score"
        return selected, meta, pd.DataFrame()

    if rule in TWO_STAGE_RULES:
        shortlist_rule = _resolve_two_stage_shortlist_rule(rule, model_size)
        shortlist, shortlist_meta = _build_shortlist_from_rule(
            pair_df,
            shortlist_rule=shortlist_rule,
            shortlist_size=reranker_shortlist_size,
            alignment_quantile=alignment_quantile,
        )
        reranked, reranker_meta = _rerank_shortlist(
            shortlist,
            reranker_features=reranker_features,
            reranker_fusion=reranker_fusion,
            reranker_rrf_k=reranker_rrf_k,
        )
        selected = reranked.iloc[0]
        meta.update(shortlist_meta)
        meta.update(reranker_meta)
        meta["selection_score_name"] = "reranker_score"
        meta["selected_shortlist_rank"] = int(selected["shortlist_rank"]) if "shortlist_rank" in selected else None
        meta["selected_reranker_rank"] = int(selected["reranker_rank"]) if "reranker_rank" in selected else None
        meta["selected_reranker_score"] = float(selected["reranker_score"]) if "reranker_score" in selected and not pd.isna(selected["reranker_score"]) else None
        meta["reranker_shortlist_size"] = int(reranker_shortlist_size)
        reranked = reranked.copy()
        reranked["rule"] = rule
        return selected, meta, reranked

    raise ValueError(f"Unsupported rule: {rule}")


def select_pairs_for_rules(
    pair_df: pd.DataFrame,
    selection_rules: list[str],
    model_size: str,
    top_m_pairs: int,
    reranker_shortlist_size: int,
    alignment_quantile: float,
    reranker_features: list[str],
    reranker_fusion: str,
    reranker_rrf_k: int,
    baseline_revision: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    eligible = pair_df.dropna(subset=["avg_main_score"]).copy()
    if eligible.empty:
        raise SelectionError("No evaluable pairs are available after merging geometry with average-main scores.")

    oracle = eligible.sort_values(
        ["avg_main_score", "revision_step", "layer_idx"],
        ascending=[False, True, True],
    ).iloc[0]

    baseline_rows = eligible[eligible["revision"] == baseline_revision].copy()
    if baseline_rows.empty:
        raise SelectionError(f"No evaluable rows were found for baseline_revision={baseline_revision}")
    baseline_best = baseline_rows.sort_values(
        ["avg_main_score", "layer_idx"],
        ascending=[False, True],
    ).iloc[0]
    baseline_last = baseline_rows[baseline_rows["layer_idx"] == baseline_rows["layer_idx"].max()].iloc[0]

    results: list[dict[str, Any]] = []
    shortlist_rows: list[pd.DataFrame] = []
    for rule in selection_rules:
        selected, meta, shortlist_df = _select_rule(
            eligible,
            rule,
            model_size=model_size,
            top_m_pairs=top_m_pairs,
            reranker_shortlist_size=reranker_shortlist_size,
            alignment_quantile=alignment_quantile,
            reranker_features=reranker_features,
            reranker_fusion=reranker_fusion,
            reranker_rrf_k=reranker_rrf_k,
        )
        result = {
            "rule": rule,
            "selected_revision": str(selected["revision"]),
            "selected_layer": int(selected["layer_idx"]),
            "selected_avg_main_score": float(selected["avg_main_score"]),
            "selected_num_tasks_eval": int(selected["num_tasks_eval"]),
            "selected_dispersion": float(selected["dispersion"]),
            "selected_dispersion_stable": float(selected["dispersion_stable"]),
            "selected_dispersion_delta_prev": float(selected["dispersion_delta_prev"]),
            "selected_alignment": float(selected["alignment"]) if not pd.isna(selected["alignment"]) else None,
            "selected_uniformity_score": float(selected["uniformity_score"]) if not pd.isna(selected["uniformity_score"]) else None,
            "selected_entropy_tiebreak_score": float(selected["entropy_tiebreak_score"]) if not pd.isna(selected["entropy_tiebreak_score"]) else None,
            "selected_top_pc_dominance": float(selected["top_pc_dominance"]) if not pd.isna(selected["top_pc_dominance"]) else None,
            "selected_effective_rank": float(selected["effective_rank"]) if not pd.isna(selected["effective_rank"]) else None,
            "selected_rankme": float(selected["rankme"]) if not pd.isna(selected["rankme"]) else None,
            "selected_spectral_slope": float(selected["spectral_slope"]) if not pd.isna(selected["spectral_slope"]) else None,
            "selected_participation_ratio": float(selected["participation_ratio"]) if not pd.isna(selected["participation_ratio"]) else None,
            "selected_knn_aug_stability": float(selected["knn_aug_stability"]) if not pd.isna(selected["knn_aug_stability"]) else None,
            "selected_checkpoint_neighbor_stability": float(selected["checkpoint_neighbor_stability"]) if not pd.isna(selected["checkpoint_neighbor_stability"]) else None,
            "selected_layer_neighbor_stability": float(selected["layer_neighbor_stability"]) if not pd.isna(selected["layer_neighbor_stability"]) else None,
            "selected_rankme_delta_prev": float(selected["rankme_delta_prev"]) if not pd.isna(selected["rankme_delta_prev"]) else None,
            "selected_spectral_slope_delta_prev": float(selected["spectral_slope_delta_prev"]) if not pd.isna(selected["spectral_slope_delta_prev"]) else None,
            "selected_pass_phase_score": float(selected["pass_phase_score"]) if not pd.isna(selected["pass_phase_score"]) else None,
            "selected_pass_volatility": float(selected["pass_volatility"]) if not pd.isna(selected["pass_volatility"]) else None,
            "selected_pass_score": float(selected["pass_score"]) if not pd.isna(selected["pass_score"]) else None,
            "selected_avg_main_rank_desc": float(selected["avg_main_rank_desc"]) if not pd.isna(selected["avg_main_rank_desc"]) else None,
            "selected_dispersion_rank_desc": float(selected["dispersion_rank_desc"]) if not pd.isna(selected["dispersion_rank_desc"]) else None,
            "selected_dispersion_stable_rank_desc": float(selected["dispersion_stable_rank_desc"]) if not pd.isna(selected["dispersion_stable_rank_desc"]) else None,
            "selected_alignment_rank_asc": float(selected["alignment_rank_asc"]) if not pd.isna(selected["alignment_rank_asc"]) else None,
            "selected_uniformity_rank_desc": float(selected["uniformity_rank_desc"]) if not pd.isna(selected["uniformity_rank_desc"]) else None,
            "selected_entropy_tiebreak_rank_asc": float(selected["entropy_tiebreak_rank_asc"]) if not pd.isna(selected["entropy_tiebreak_rank_asc"]) else None,
            "selected_rankme_rank_desc": float(selected["rankme_rank_desc"]) if not pd.isna(selected["rankme_rank_desc"]) else None,
            "selected_spectral_slope_rank_desc": float(selected["spectral_slope_rank_desc"]) if not pd.isna(selected["spectral_slope_rank_desc"]) else None,
            "selected_pass_phase_score_rank_desc": float(selected["pass_phase_score_rank_desc"]) if not pd.isna(selected["pass_phase_score_rank_desc"]) else None,
            "selected_pass_volatility_rank_asc": float(selected["pass_volatility_rank_asc"]) if not pd.isna(selected["pass_volatility_rank_asc"]) else None,
            "selected_pass_score_rank_desc": float(selected["pass_score_rank_desc"]) if not pd.isna(selected["pass_score_rank_desc"]) else None,
            "delta_vs_oracle": float(selected["avg_main_score"] - oracle["avg_main_score"]),
            "delta_vs_baseline_best": float(selected["avg_main_score"] - baseline_best["avg_main_score"]),
            "delta_vs_baseline_last": float(selected["avg_main_score"] - baseline_last["avg_main_score"]),
        }
        result.update(meta)
        results.append(result)
        if not shortlist_df.empty:
            shortlist_export = shortlist_df.copy()
            shortlist_export["selected_by_reranker"] = (
                (shortlist_export["revision"].astype(str) == str(selected["revision"])) &
                (shortlist_export["layer_idx"].astype(int) == int(selected["layer_idx"]))
            )
            shortlist_rows.append(shortlist_export)

    baseline_payload = {
        "oracle_best_pair": {
            "revision": str(oracle["revision"]),
            "layer": int(oracle["layer_idx"]),
            "avg_main_score": float(oracle["avg_main_score"]),
            "avg_main_rank_desc": float(oracle["avg_main_rank_desc"]) if not pd.isna(oracle["avg_main_rank_desc"]) else None,
            "dispersion_rank_desc": float(oracle["dispersion_rank_desc"]) if not pd.isna(oracle["dispersion_rank_desc"]) else None,
            "dispersion_stable_rank_desc": float(oracle["dispersion_stable_rank_desc"]) if not pd.isna(oracle["dispersion_stable_rank_desc"]) else None,
            "alignment_rank_asc": float(oracle["alignment_rank_asc"]) if not pd.isna(oracle["alignment_rank_asc"]) else None,
            "uniformity_rank_desc": float(oracle["uniformity_rank_desc"]) if not pd.isna(oracle["uniformity_rank_desc"]) else None,
            "entropy_tiebreak_rank_asc": float(oracle["entropy_tiebreak_rank_asc"]) if not pd.isna(oracle["entropy_tiebreak_rank_asc"]) else None,
        },
        "baseline_revision_best_layer": {
            "revision": str(baseline_best["revision"]),
            "layer": int(baseline_best["layer_idx"]),
            "avg_main_score": float(baseline_best["avg_main_score"]),
        },
        "baseline_revision_last_layer": {
            "revision": str(baseline_last["revision"]),
            "layer": int(baseline_last["layer_idx"]),
            "avg_main_score": float(baseline_last["avg_main_score"]),
        },
    }
    shortlist_payload = pd.concat(shortlist_rows, ignore_index=True) if shortlist_rows else pd.DataFrame()
    return pd.DataFrame(results), baseline_payload, shortlist_payload


def write_summary_markdown(
    output_dir: str,
    args: argparse.Namespace,
    usable_revisions: list[str],
    task_texts: dict[str, list[str]],
    global_pool_df: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    selected_rules_df: pd.DataFrame,
    baselines: dict[str, Any],
    entropy_tasks_used: list[str],
) -> None:
    lines = [
        "# Global Unsupervised Best-Pair Selector",
        "",
        "## Configuration",
        f"- model: `{args.model_family} {args.model_size}`",
        f"- usable revisions: `{', '.join(usable_revisions)}`",
        f"- candidate layers: `{args.layer_start}..{args.layer_end}`",
        f"- per-task texts in global pool: `{args.pool_samples_per_task}`",
        f"- total global pool size: `{len(global_pool_df)}`",
        f"- dispersion pairs: `{args.dispersion_num_pairs}`",
        f"- uniformity pairs: `{args.uniformity_num_pairs}`",
        f"- alignment quantile: `{args.alignment_quantile}`",
        f"- entropy tie-break mode: `{args.entropy_tiebreak_mode}`",
        f"- stability lambda: `{args.stability_lambda}`",
        f"- PASS tau rank: `{args.pass_tau_rank}`",
        f"- PASS tau spectral: `{args.pass_tau_spectral}`",
        f"- PASS weights (D,U,A,P,V): `({args.pass_weight_dispersion}, {args.pass_weight_uniformity}, {args.pass_weight_alignment}, {args.pass_weight_phase}, {args.pass_weight_volatility})`",
        f"- reranker shortlist size: `{args.reranker_shortlist_size}`",
        f"- reranker fusion: `{args.reranker_fusion}`",
        f"- reranker RRF k: `{args.reranker_rrf_k}`",
        f"- reranker kNN k: `{args.reranker_knn_k}`",
        f"- reranker analysis texts: `{args.reranker_analysis_num_texts}`",
        f"- reranker features: `{args.reranker_features}`",
        f"- selection rules: `{args.selection_rules}`",
        "",
        "## Baselines",
        f"- oracle best static pair: `{baselines['oracle_best_pair']['revision']}/layer_{baselines['oracle_best_pair']['layer']}` -> `{baselines['oracle_best_pair']['avg_main_score']:.10f}`",
        f"- `{args.baseline_revision}` best layer: `layer_{baselines['baseline_revision_best_layer']['layer']}` -> `{baselines['baseline_revision_best_layer']['avg_main_score']:.10f}`",
        f"- `{args.baseline_revision}` last layer: `layer_{baselines['baseline_revision_last_layer']['layer']}` -> `{baselines['baseline_revision_last_layer']['avg_main_score']:.10f}`",
        "",
        "## Rule Results",
    ]

    for _, row in selected_rules_df.sort_values("selected_avg_main_score", ascending=False).iterrows():
        detail_bits: list[str] = []
        if pd.notna(row.get("shortlist_rule")):
            detail_bits.append(f"shortlist `{row['shortlist_rule']}`")
        if pd.notna(row.get("num_shortlist_candidates")):
            detail_bits.append(f"K={int(row['num_shortlist_candidates'])}")
        if pd.notna(row.get("selected_reranker_rank")):
            detail_bits.append(f"reranker rank `{int(row['selected_reranker_rank'])}`")
        detail_text = f" [{' / '.join(detail_bits)}]" if detail_bits else ""
        lines.append(
            f"- `{row['rule']}`{detail_text}: `{row['selected_revision']}/layer_{int(row['selected_layer'])}` -> `{row['selected_avg_main_score']:.10f}` "
            f"(vs `{args.baseline_revision}` best: `{row['delta_vs_baseline_best']:+.10f}`, "
            f"vs `{args.baseline_revision}` last: `{row['delta_vs_baseline_last']:+.10f}`, "
            f"vs oracle: `{row['delta_vs_oracle']:+.10f}`)"
        )

    if entropy_tasks_used:
        lines.extend([
            "",
            "## Entropy Tie-Break Coverage",
            f"- tasks contributing to entropy tie-break aggregation: `{len(entropy_tasks_used)}`",
            f"- task names: `{', '.join(entropy_tasks_used)}`",
        ])

    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    revisions = [x.strip() for x in args.expected_revisions.split(",") if x.strip()]
    candidate_layers = list(range(args.layer_start, args.layer_end + 1))
    requested_rules = [x.strip() for x in args.selection_rules.split(",") if x.strip()]
    reranker_features = [x.strip() for x in args.reranker_features.split(",") if x.strip()]
    invalid_rules = sorted(set(requested_rules) - set(SUPPORTED_RULES))
    if invalid_rules:
        raise SelectionError(f"Unsupported selection rules requested: {invalid_rules}")
    invalid_reranker_features = sorted(set(reranker_features) - set(RERANKER_FEATURE_SPECS))
    if invalid_reranker_features:
        raise SelectionError(f"Unsupported reranker features requested: {invalid_reranker_features}")
    if args.reranker_shortlist_size < 1:
        raise SelectionError("--reranker_shortlist_size must be >= 1")
    if args.top_m_pairs < 1:
        raise SelectionError("--top_m_pairs must be >= 1")
    if args.pass_tau_rank <= 0.0:
        raise SelectionError("--pass_tau_rank must be > 0")
    if args.pass_tau_spectral <= 0.0:
        raise SelectionError("--pass_tau_spectral must be > 0")

    task_names = [x.strip() for x in args.task_names.split(",") if x.strip()]
    task_to_dataset = TASK_TO_DATASET.copy()
    if task_names:
        missing = sorted(set(task_names) - set(task_to_dataset))
        if missing:
            raise SelectionError(f"Unknown task_names requested: {missing}")
        task_to_dataset = {k: task_to_dataset[k] for k in task_names}

    output_dir = os.path.join(
        args.output_root,
        f"{_model_slug(args.model_family, args.model_size)}_global_pair_selector_unsupervised_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)

    cache_task_texts_path = args.cache_task_texts_path.strip()
    cache_search_roots = [
        args.output_root,
        args.main_runs_root,
        os.path.join(REPO_ROOT, "experiments", "results_reruns"),
    ]
    if cache_task_texts_path and not os.path.isfile(cache_task_texts_path):
        discovered_cache = _discover_task_text_cache(
            cache_search_roots,
            requested_tasks=task_to_dataset.keys(),
            min_samples_per_task=args.pool_samples_per_task,
            sample_seed=args.sample_seed,
        )
        if discovered_cache is not None:
            print(
                f"Requested cache_task_texts_path was not found at {cache_task_texts_path}; "
                f"reusing compatible cache {discovered_cache}"
            )
            cache_task_texts_path = discovered_cache

    if not cache_task_texts_path:
        discovered_cache = _discover_task_text_cache(
            cache_search_roots,
            requested_tasks=task_to_dataset.keys(),
            min_samples_per_task=args.pool_samples_per_task,
            sample_seed=args.sample_seed,
        )
        if discovered_cache is not None:
            print(f"Reusing compatible task-text cache {discovered_cache}")
            cache_task_texts_path = discovered_cache
        else:
            cache_task_texts_path = os.path.join(output_dir, "sampled_task_texts.json")

    task_texts = load_sampled_task_texts(
        task_to_dataset,
        args.pool_samples_per_task,
        cache_task_texts_path,
        args.sample_seed,
    )
    task_texts = {task: texts for task, texts in task_texts.items() if task in task_to_dataset}
    task_texts = _cap_task_texts(task_texts, args.pool_samples_per_task, args.sample_seed)
    usable_task_texts = {task: texts for task, texts in task_texts.items() if texts}
    if not usable_task_texts:
        raise SelectionError("No task texts were available for the global unlabeled pool.")

    global_texts, global_pool_df = _build_global_pool(usable_task_texts)
    global_pool_df.to_csv(os.path.join(output_dir, "global_pool_texts.csv"), index=False)
    with open(os.path.join(output_dir, "global_pool_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "pool_samples_per_task": int(args.pool_samples_per_task),
                "sample_seed": int(args.sample_seed),
                "num_tasks": int(len(usable_task_texts)),
                "num_texts": int(len(global_texts)),
                "task_counts": {k: len(v) for k, v in usable_task_texts.items()},
                "cache_task_texts_path": cache_task_texts_path,
            },
            f,
            indent=2,
        )

    pair_geometry_df = compute_pair_geometry(global_texts, revisions, candidate_layers, args)
    pair_eval_df = load_pair_avg_main_table(args.main_runs_root, revisions, candidate_layers)

    usable_revisions = sorted(
        set(pair_geometry_df["revision"]).intersection(pair_eval_df["revision"]),
        key=_parse_revision_order_key,
    )
    if not usable_revisions:
        raise SelectionError("No overlap between geometry results and evaluable average-main-score revisions.")

    pair_geometry_df = pair_geometry_df[pair_geometry_df["revision"].isin(usable_revisions)].copy()
    pair_eval_df = pair_eval_df[pair_eval_df["revision"].isin(usable_revisions)].copy()

    pair_metrics = pair_geometry_df.merge(pair_eval_df, on=["revision", "layer_idx"], how="left")
    pair_metrics = add_temporal_stability(pair_metrics, usable_revisions, args.stability_lambda)
    pair_metrics = add_pass_metric(pair_metrics, usable_revisions, args)

    entropy_pair_scores, task_layer_entropy_scores, entropy_tasks_used = build_pair_entropy_tiebreak_table(
        args,
        usable_revisions,
        {task: task_to_dataset[task] for task in usable_task_texts},
        candidate_layers,
    )
    if not entropy_pair_scores.empty:
        pair_metrics = pair_metrics.merge(entropy_pair_scores, on=["revision", "layer_idx"], how="left")
        entropy_pair_scores.to_csv(os.path.join(output_dir, "pair_entropy_tiebreak_scores.csv"), index=False)
        task_layer_entropy_scores.to_csv(os.path.join(output_dir, "task_layer_entropy_scores.csv"), index=False)
    else:
        pair_metrics["entropy_tiebreak_score"] = np.nan
        pair_metrics["entropy_tiebreak_score_median"] = np.nan
        pair_metrics["entropy_tiebreak_support_tasks"] = np.nan
        pair_metrics["dataset_entropy_mean"] = np.nan
        pair_metrics["infonce_for_min_mean"] = np.nan
        pair_metrics["dime_mean"] = np.nan

    pair_metrics = add_metric_ranks(pair_metrics)
    pair_metrics = pair_metrics.sort_values(["revision_step", "layer_idx"], ascending=[True, True]).reset_index(drop=True)
    pair_metrics.to_csv(os.path.join(output_dir, "pair_metrics.csv"), index=False)

    selected_rules_df, baselines, reranker_shortlists_df = select_pairs_for_rules(
        pair_metrics,
        selection_rules=requested_rules,
        model_size=args.model_size,
        top_m_pairs=args.top_m_pairs,
        reranker_shortlist_size=args.reranker_shortlist_size,
        alignment_quantile=args.alignment_quantile,
        reranker_features=reranker_features,
        reranker_fusion=args.reranker_fusion,
        reranker_rrf_k=args.reranker_rrf_k,
        baseline_revision=args.baseline_revision,
    )
    selected_rules_df.to_csv(os.path.join(output_dir, "selected_rule_pairs.csv"), index=False)
    if not reranker_shortlists_df.empty:
        reranker_shortlists_df.to_csv(os.path.join(output_dir, "two_stage_shortlists.csv"), index=False)

    summary_payload = {
        "config": {
            "model_family": args.model_family,
            "model_size": args.model_size,
            "requested_revisions": revisions,
            "usable_revisions": usable_revisions,
            "candidate_layers": candidate_layers,
            "pooling_method": args.pooling_method,
            "pool_samples_per_task": args.pool_samples_per_task,
            "sample_seed": args.sample_seed,
            "dispersion_num_pairs": args.dispersion_num_pairs,
            "uniformity_num_pairs": args.uniformity_num_pairs,
            "uniformity_temperature": args.uniformity_temperature,
            "alignment_quantile": args.alignment_quantile,
            "top_m_pairs": args.top_m_pairs,
            "reranker_shortlist_size": args.reranker_shortlist_size,
            "reranker_knn_k": args.reranker_knn_k,
            "reranker_analysis_num_texts": args.reranker_analysis_num_texts,
            "reranker_fusion": args.reranker_fusion,
            "reranker_rrf_k": args.reranker_rrf_k,
            "reranker_features": reranker_features,
            "pass_tau_rank": args.pass_tau_rank,
            "pass_tau_spectral": args.pass_tau_spectral,
            "pass_weight_dispersion": args.pass_weight_dispersion,
            "pass_weight_uniformity": args.pass_weight_uniformity,
            "pass_weight_alignment": args.pass_weight_alignment,
            "pass_weight_phase": args.pass_weight_phase,
            "pass_weight_volatility": args.pass_weight_volatility,
            "stability_lambda": args.stability_lambda,
            "entropy_tiebreak_mode": args.entropy_tiebreak_mode,
            "selection_rules": requested_rules,
            "baseline_revision": args.baseline_revision,
        },
        "global_pool": {
            "num_tasks": int(len(usable_task_texts)),
            "num_texts": int(len(global_texts)),
            "task_counts": {k: len(v) for k, v in usable_task_texts.items()},
        },
        "num_pair_rows": int(len(pair_metrics)),
        "num_evaluable_pairs": int(pair_metrics["avg_main_score"].notna().sum()),
        "entropy_tiebreak_tasks_used": entropy_tasks_used,
        "baselines": baselines,
        "rule_results": selected_rules_df.to_dict(orient="records"),
        "num_two_stage_shortlist_rows": int(len(reranker_shortlists_df)),
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    write_summary_markdown(
        output_dir,
        args,
        usable_revisions,
        usable_task_texts,
        global_pool_df,
        pair_metrics,
        selected_rules_df,
        baselines,
        entropy_tasks_used,
    )

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
