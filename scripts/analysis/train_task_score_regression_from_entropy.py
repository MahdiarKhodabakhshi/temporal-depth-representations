"""
Train/test a linear regressor on task-level data:
X = [dataset_entropy_maxEntropy, infonce_mi_lower_bound, dime_maxEntropy]
y = task main_score

Evaluation uses Leave-One-Checkpoint-Out (9 train checkpoints, 1 test checkpoint).
For each held-out checkpoint, we select a layer per task by predicted main score
and report how close that is to each task's oracle best layer score.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments" / "results"
DEFAULT_RERUNS_ROOT = REPO_ROOT / "experiments" / "results_reruns"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


def _pick_latest_existing(paths: list[str]) -> str | None:
    existing = [p for p in paths if os.path.isdir(p)]
    if not existing:
        return None
    return max(existing, key=os.path.getmtime)


def find_main_revision_root(main_runs_root: str, revision: str) -> str | None:
    patterns = [
        os.path.join(
            main_runs_root,
            f"pythia410m_{revision}_layers15-24_*",
            "Pythia",
            "410m",
            revision,
        ),
        os.path.join(
            main_runs_root,
            f"pythia410m_{revision}_*layers15-24*",
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


def _load_entropy_arrays(entropy_root: str, revision: str, dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = os.path.join(entropy_root, revision, "metrics", "mteb", dataset_name, "test")
    entropy_obj = _load_pickle(os.path.join(base, "entropy_dataset.pkl"))
    infonce_obj = _load_pickle(os.path.join(base, "infonce.pkl"))
    dime_obj = _load_pickle(os.path.join(base, "dime.pkl"))

    e = np.asarray(entropy_obj["maxEntropy"], dtype=float)
    if "mi-lower-bound" in infonce_obj:
        i = np.asarray(infonce_obj["mi-lower-bound"], dtype=float)
    elif "mi_lower_bound" in infonce_obj:
        i = np.asarray(infonce_obj["mi_lower_bound"], dtype=float)
    else:
        i = np.asarray(infonce_obj["raw"], dtype=float)
    d = np.asarray(dime_obj["maxEntropy"], dtype=float)
    return e, i, d


def _safe_main_score(payload: dict[str, Any]) -> float:
    return float(payload["scores"]["test"][0]["main_score"])


def _list_layer_indices(main_revision_root: str) -> list[int]:
    mteb_dir = os.path.join(main_revision_root, "mteb")
    layers: list[int] = []
    if not os.path.isdir(mteb_dir):
        return layers
    for name in os.listdir(mteb_dir):
        if not name.startswith("layer_"):
            continue
        try:
            layers.append(int(name.split("_")[-1]))
        except ValueError:
            continue
    return sorted(layers)


def _load_main_scores(main_revision_root: str, layer_idx: int) -> dict[str, float]:
    layer_dir = os.path.join(main_revision_root, "mteb", f"layer_{layer_idx}")
    out: dict[str, float] = {}
    for json_path in sorted(glob.glob(os.path.join(layer_dir, "*.json"))):
        if json_path.endswith("model_meta.json"):
            continue
        with open(json_path, "r") as f:
            payload = json.load(f)
        task_name = str(payload["task_name"])
        out[task_name] = _safe_main_score(payload)
    return out


def build_task_layer_dataset(
    entropy_root: str,
    main_runs_root: str,
    revisions: list[str],
    layer_start: int,
    layer_end: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {
        "revisions_requested": revisions,
        "revisions_used": [],
        "revisions_skipped_missing_main": [],
        "revision_main_roots": {},
    }
    rows: list[dict[str, Any]] = []

    for revision in revisions:
        main_root = find_main_revision_root(main_runs_root, revision)
        if main_root is None:
            meta["revisions_skipped_missing_main"].append(revision)
            continue
        meta["revision_main_roots"][revision] = main_root

        available_layers = [x for x in _list_layer_indices(main_root) if layer_start <= x <= layer_end]
        if not available_layers:
            meta["revisions_skipped_missing_main"].append(revision)
            continue

        task_layer_scores: dict[int, dict[str, float]] = {}
        for layer_idx in available_layers:
            task_layer_scores[layer_idx] = _load_main_scores(main_root, layer_idx)

        entropy_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        revision_rows_before = len(rows)
        for task_name, dataset_name in TASK_TO_DATASET.items():
            try:
                if dataset_name not in entropy_cache:
                    entropy_cache[dataset_name] = _load_entropy_arrays(entropy_root, revision, dataset_name)
                e, i, d = entropy_cache[dataset_name]
            except FileNotFoundError:
                continue

            for layer_idx in available_layers:
                main_score = task_layer_scores[layer_idx].get(task_name)
                if main_score is None:
                    continue
                if layer_idx >= len(e) or layer_idx >= len(i) or layer_idx >= len(d):
                    continue
                rows.append(
                    {
                        "revision": revision,
                        "task_name": task_name,
                        "dataset_name": dataset_name,
                        "layer_idx": layer_idx,
                        "dataset_entropy_maxEntropy": float(e[layer_idx]),
                        "infonce_mi_lower_bound": float(i[layer_idx]),
                        "dime_maxEntropy": float(d[layer_idx]),
                        "main_score": float(main_score),
                    }
                )
        if len(rows) > revision_rows_before:
            meta["revisions_used"].append(revision)
        else:
            meta["revisions_skipped_missing_main"].append(revision)

    if not rows:
        raise RuntimeError("No rows built for training data.")

    data = pd.DataFrame(rows)
    # Some metric files contain NaNs for specific task/layer pairs.
    # LinearRegression cannot train with NaNs, so keep only finite rows.
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(
        subset=[
            "dataset_entropy_maxEntropy",
            "infonce_mi_lower_bound",
            "dime_maxEntropy",
            "main_score",
        ]
    )
    # Keep the same candidate layer set for every checkpoint.
    layer_sets = [
        set(group["layer_idx"].unique().tolist())
        for _, group in data.groupby("revision", sort=True)
    ]
    common_layers = sorted(set.intersection(*layer_sets)) if layer_sets else []
    if common_layers:
        data = data[data["layer_idx"].isin(common_layers)].copy()
    meta["common_layers_used"] = common_layers
    data.sort_values(["revision", "task_name", "layer_idx"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data, meta


def _eval_checkpoint_taskwise(test_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    task_rows: list[dict[str, Any]] = []

    for task_name, g in test_df.groupby("task_name", sort=True):
        g2 = g.sort_values("layer_idx")
        pred_idx = int(g2["pred_main_score"].values.argmax())
        true_idx = int(g2["main_score"].values.argmax())
        pred_row = g2.iloc[pred_idx]
        true_row = g2.iloc[true_idx]

        chosen_layer = int(pred_row["layer_idx"])
        oracle_layer = int(true_row["layer_idx"])
        chosen_true_score = float(pred_row["main_score"])
        oracle_score = float(true_row["main_score"])
        regret = oracle_score - chosen_true_score

        task_rows.append(
            {
                "revision": str(g2["revision"].iloc[0]),
                "task_name": task_name,
                "chosen_layer": chosen_layer,
                "oracle_layer": oracle_layer,
                "layer_abs_error": abs(chosen_layer - oracle_layer),
                "chosen_true_score": chosen_true_score,
                "oracle_score": oracle_score,
                "regret": regret,
                "within_0_002": int(chosen_true_score >= oracle_score - 0.002),
                "within_0_005": int(chosen_true_score >= oracle_score - 0.005),
                "within_0_010": int(chosen_true_score >= oracle_score - 0.010),
            }
        )

    task_df = pd.DataFrame(task_rows)
    mean_by_layer = test_df.groupby("layer_idx", as_index=False)["main_score"].mean()
    best_single_layer_row = mean_by_layer.sort_values("main_score", ascending=False).iloc[0]

    summary = {
        "revision": str(test_df["revision"].iloc[0]),
        "num_tasks": int(task_df.shape[0]),
        "selected_avg_main": float(task_df["chosen_true_score"].mean()),
        "oracle_avg_main": float(task_df["oracle_score"].mean()),
        "avg_regret": float(task_df["regret"].mean()),
        "median_regret": float(task_df["regret"].median()),
        "mean_abs_layer_error": float(task_df["layer_abs_error"].mean()),
        "within_0_002_rate": float(task_df["within_0_002"].mean()),
        "within_0_005_rate": float(task_df["within_0_005"].mean()),
        "within_0_010_rate": float(task_df["within_0_010"].mean()),
        "best_single_layer": int(best_single_layer_row["layer_idx"]),
        "best_single_layer_avg_main": float(best_single_layer_row["main_score"]),
    }
    summary["selected_minus_best_single_layer_avg"] = (
        summary["selected_avg_main"] - summary["best_single_layer_avg_main"]
    )
    summary["selected_over_oracle_ratio"] = (
        summary["selected_avg_main"] / summary["oracle_avg_main"]
        if summary["oracle_avg_main"] != 0
        else float("nan")
    )
    return summary, task_df


def run_logo(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feature_cols = [
        "dataset_entropy_maxEntropy",
        "infonce_mi_lower_bound",
        "dime_maxEntropy",
    ]
    target_col = "main_score"
    group_col = "revision"

    X = data[feature_cols].values
    y = data[target_col].values
    groups = data[group_col].values

    model = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])
    logo = LeaveOneGroupOut()

    oof_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    task_eval_rows: list[pd.DataFrame] = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
        train_df = data.iloc[train_idx]
        test_df = data.iloc[test_idx].copy()

        model.fit(train_df[feature_cols].values, train_df[target_col].values)
        preds = model.predict(test_df[feature_cols].values)
        test_df["pred_main_score"] = preds
        test_df["fold"] = fold_idx
        oof_parts.append(test_df)

        rmse = float(np.sqrt(mean_squared_error(test_df[target_col], preds)))
        mae = float(mean_absolute_error(test_df[target_col], preds))
        r2 = float(r2_score(test_df[target_col], preds))

        cp_summary, cp_task_df = _eval_checkpoint_taskwise(test_df)
        cp_summary["fold"] = fold_idx
        cp_summary["rmse_rows"] = rmse
        cp_summary["mae_rows"] = mae
        cp_summary["r2_rows"] = r2
        fold_rows.append(cp_summary)

        cp_task_df["fold"] = fold_idx
        task_eval_rows.append(cp_task_df)

    oof_df = pd.concat(oof_parts, ignore_index=True)
    fold_df = pd.DataFrame(fold_rows).sort_values("revision").reset_index(drop=True)
    task_eval_df = pd.concat(task_eval_rows, ignore_index=True).sort_values(["revision", "task_name"])

    overall = {
        "num_rows": int(data.shape[0]),
        "num_checkpoints": int(data["revision"].nunique()),
        "num_tasks_total": int(data["task_name"].nunique()),
        "num_layers_total": int(data["layer_idx"].nunique()),
        "rmse_oof_rows": float(np.sqrt(mean_squared_error(oof_df[target_col], oof_df["pred_main_score"]))),
        "mae_oof_rows": float(mean_absolute_error(oof_df[target_col], oof_df["pred_main_score"])),
        "r2_oof_rows": float(r2_score(oof_df[target_col], oof_df["pred_main_score"])),
        "mean_selected_avg_main": float(fold_df["selected_avg_main"].mean()),
        "mean_oracle_avg_main": float(fold_df["oracle_avg_main"].mean()),
        "mean_avg_regret": float(fold_df["avg_regret"].mean()),
        "mean_within_0_002_rate": float(fold_df["within_0_002_rate"].mean()),
        "mean_within_0_005_rate": float(fold_df["within_0_005_rate"].mean()),
        "mean_within_0_010_rate": float(fold_df["within_0_010_rate"].mean()),
        "mean_abs_layer_error": float(fold_df["mean_abs_layer_error"].mean()),
    }

    model.fit(X, y)
    reg = model.named_steps["regressor"]
    scaler = model.named_steps["scaler"]
    coef = reg.coef_ / scaler.scale_
    intercept = float(reg.intercept_ - np.sum((reg.coef_ * scaler.mean_) / scaler.scale_))
    overall["linear_coefficients"] = {
        "dataset_entropy_maxEntropy": float(coef[0]),
        "infonce_mi_lower_bound": float(coef[1]),
        "dime_maxEntropy": float(coef[2]),
        "intercept": intercept,
    }
    return oof_df, fold_df, task_eval_df, overall


def write_outputs(
    out_dir: str,
    data: pd.DataFrame,
    oof: pd.DataFrame,
    folds: pd.DataFrame,
    task_eval: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    dataset_path = os.path.join(out_dir, "dataset_task_layer.csv")
    oof_path = os.path.join(out_dir, "oof_row_predictions.csv")
    folds_path = os.path.join(out_dir, "fold_checkpoint_metrics.csv")
    task_eval_path = os.path.join(out_dir, "fold_task_selection_metrics.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    summary_md = os.path.join(out_dir, "summary.md")

    data.to_csv(dataset_path, index=False)
    oof.to_csv(oof_path, index=False)
    folds.to_csv(folds_path, index=False)
    task_eval.to_csv(task_eval_path, index=False)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    with open(summary_md, "w") as f:
        ov = summary["overall"]
        f.write("# Taskwise Entropy-to-Main Linear Regression (LOGO)\n\n")
        f.write("## Setup\n\n")
        f.write(f"- rows: {ov['num_rows']}\n")
        f.write(f"- checkpoints: {ov['num_checkpoints']}\n")
        f.write(f"- tasks: {ov['num_tasks_total']}\n")
        f.write(f"- layers: {ov['num_layers_total']}\n")
        f.write("\n## Row-Level OOF Metrics\n\n")
        f.write(f"- rmse_oof_rows: {ov['rmse_oof_rows']:.6f}\n")
        f.write(f"- mae_oof_rows: {ov['mae_oof_rows']:.6f}\n")
        f.write(f"- r2_oof_rows: {ov['r2_oof_rows']:.6f}\n")
        f.write("\n## Taskwise Selection Quality (Held-Out Checkpoints)\n\n")
        f.write(f"- mean_selected_avg_main: {ov['mean_selected_avg_main']:.6f}\n")
        f.write(f"- mean_oracle_avg_main: {ov['mean_oracle_avg_main']:.6f}\n")
        f.write(f"- mean_avg_regret: {ov['mean_avg_regret']:.6f}\n")
        f.write(f"- mean_within_0_002_rate: {ov['mean_within_0_002_rate']:.4f}\n")
        f.write(f"- mean_within_0_005_rate: {ov['mean_within_0_005_rate']:.4f}\n")
        f.write(f"- mean_within_0_010_rate: {ov['mean_within_0_010_rate']:.4f}\n")
        f.write(f"- mean_abs_layer_error: {ov['mean_abs_layer_error']:.4f}\n")
        f.write("\n## Fold Checkpoint Metrics\n\n")
        f.write(folds.to_markdown(index=False))
        f.write("\n\n## Linear Coefficients\n\n")
        coef_df = pd.DataFrame(
            [{"feature": k, "coefficient": v} for k, v in ov["linear_coefficients"].items()]
        )
        f.write(coef_df.to_markdown(index=False))
        f.write("\n")

    print(out_dir)
    print(summary_md)
    print(summary_json)


def main() -> None:
    args = parse_args()
    revisions = [x.strip() for x in args.expected_revisions.split(",") if x.strip()]
    data, meta = build_task_layer_dataset(
        entropy_root=args.entropy_root,
        main_runs_root=args.main_runs_root,
        revisions=revisions,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
    )
    oof, folds, task_eval, overall = run_logo(data)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_root, f"pythia410m_entropy3_taskwise_linear_logo_{ts}")
    summary = {
        "entropy_root": args.entropy_root,
        "main_runs_root": args.main_runs_root,
        "expected_revisions": revisions,
        "layer_range": [args.layer_start, args.layer_end],
        "meta": meta,
        "overall": overall,
    }
    write_outputs(out_dir, data, oof, folds, task_eval, summary)


if __name__ == "__main__":
    main()
