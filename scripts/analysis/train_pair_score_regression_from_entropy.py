#!/usr/bin/env python3
"""
Train/test a linear regressor that predicts avg main score from 3 entropy features
for (checkpoint, layer) pairs.

Data expected:
- Entropy tables per checkpoint: layer_metric_table.csv
- Avg-main tables per checkpoint: avg_main_score_by_layer.csv

Evaluation:
- Leave-One-Checkpoint-Out cross-validation
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RERUNS_ROOT = REPO_ROOT / "experiments" / "results_reruns"


@dataclass
class CheckpointPaths:
    revision: str
    entropy_csv: str
    avg_main_csv: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entropy_tables_root",
        type=str,
        default=str(DEFAULT_RERUNS_ROOT),
    )
    parser.add_argument(
        "--avg_main_root",
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
    return parser.parse_args()


def _layer_to_idx(layer_value: str) -> int:
    return int(str(layer_value).split("_")[-1])


def find_checkpoint_paths(entropy_tables_root: str, avg_main_root: str, revisions: list[str]) -> list[CheckpointPaths]:
    out: list[CheckpointPaths] = []
    for rev in revisions:
        entropy_matches = sorted(
            glob.glob(os.path.join(entropy_tables_root, rev, "layer_metric_table.csv"))
            + glob.glob(os.path.join(entropy_tables_root, "**", rev, "layer_metric_table.csv"), recursive=True)
        )
        if not entropy_matches:
            continue
        entropy_csv = max(entropy_matches, key=os.path.getmtime)

        avg_glob = os.path.join(
            avg_main_root,
            f"pythia410m_{rev}_layers15-24_h100_*",
            "Pythia",
            "410m",
            rev,
            "average_main_score",
            "avg_main_score_by_layer.csv",
        )
        matches = sorted(glob.glob(avg_glob))
        avg_main_csv = matches[0] if matches else None
        out.append(CheckpointPaths(revision=rev, entropy_csv=entropy_csv, avg_main_csv=avg_main_csv))
    return out


def build_dataset(paths: list[CheckpointPaths]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for cp in paths:
        if cp.avg_main_csv is None:
            skipped.append(cp.revision)
            continue

        ent = pd.read_csv(cp.entropy_csv).copy()
        ent["layer_idx"] = ent["layer"].map(_layer_to_idx)
        ent = ent[
            [
                "layer_idx",
                "dataset_entropy_maxEntropy",
                "infonce_mi_lower_bound",
                "dime_maxEntropy",
            ]
        ]

        avg = pd.read_csv(cp.avg_main_csv).copy()
        avg["layer_idx"] = avg["layer"].map(_layer_to_idx)
        avg = avg[["layer_idx", "avg_main_score"]]

        merged = ent.merge(avg, on="layer_idx", how="inner")
        merged["revision"] = cp.revision
        merged["pair_id"] = merged["revision"] + "::layer_" + merged["layer_idx"].astype(str)
        frames.append(merged)

    if not frames:
        raise RuntimeError("No checkpoint had both entropy and avg-main-score files.")
    data = pd.concat(frames, ignore_index=True)
    return data, skipped


def run_logo_linear_regression(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    feature_cols = [
        "dataset_entropy_maxEntropy",
        "infonce_mi_lower_bound",
        "dime_maxEntropy",
    ]
    target_col = "avg_main_score"
    group_col = "revision"

    X = data[feature_cols].values
    y = data[target_col].values
    groups = data[group_col].values

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    logo = LeaveOneGroupOut()
    fold_rows: list[dict] = []
    oof_parts: list[pd.DataFrame] = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        test_df = data.iloc[test_idx].copy()
        test_group = str(test_df["revision"].iloc[0])

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        true_best_idx = int(np.argmax(y_test))
        pred_best_idx = int(np.argmax(preds))
        true_best_layer = int(test_df.iloc[true_best_idx]["layer_idx"])
        pred_best_layer = int(test_df.iloc[pred_best_idx]["layer_idx"])
        best_layer_hit = int(true_best_layer == pred_best_layer)

        fold_rows.append(
            {
                "fold": fold_idx,
                "held_out_revision": test_group,
                "num_test_rows": int(len(test_df)),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "true_best_layer": true_best_layer,
                "pred_best_layer": pred_best_layer,
                "best_layer_hit": best_layer_hit,
                "true_best_score": float(y_test[true_best_idx]),
                "pred_score_at_pred_best": float(preds[pred_best_idx]),
            }
        )

        test_df["pred_avg_main_score"] = preds
        test_df["fold"] = fold_idx
        oof_parts.append(test_df)

    oof = pd.concat(oof_parts, ignore_index=True)
    folds = pd.DataFrame(fold_rows)

    overall = {
        "num_rows": int(len(data)),
        "num_checkpoints": int(data["revision"].nunique()),
        "rmse_oof": float(np.sqrt(mean_squared_error(oof[target_col], oof["pred_avg_main_score"]))),
        "mae_oof": float(mean_absolute_error(oof[target_col], oof["pred_avg_main_score"])),
        "r2_oof": float(r2_score(oof[target_col], oof["pred_avg_main_score"])),
        "best_layer_hit_rate": float(folds["best_layer_hit"].mean()),
    }

    # OOF-based global-best pair estimate
    oof_pred_best = oof.sort_values("pred_avg_main_score", ascending=False).iloc[0]
    oof_true_best = oof.sort_values("avg_main_score", ascending=False).iloc[0]
    overall["oof_pred_global_best_pair"] = f"{oof_pred_best['revision']}::layer_{int(oof_pred_best['layer_idx'])}"
    overall["oof_pred_global_best_score_pred"] = float(oof_pred_best["pred_avg_main_score"])
    overall["oof_true_global_best_pair"] = f"{oof_true_best['revision']}::layer_{int(oof_true_best['layer_idx'])}"
    overall["oof_true_global_best_score"] = float(oof_true_best["avg_main_score"])

    # Fit on full data for final deployable model summary
    model.fit(X, y)
    full_pred = model.predict(X)
    full_df = data.copy()
    full_df["pred_avg_main_score"] = full_pred
    final_pred_best = full_df.sort_values("pred_avg_main_score", ascending=False).iloc[0]
    overall["final_fit_pred_best_pair"] = f"{final_pred_best['revision']}::layer_{int(final_pred_best['layer_idx'])}"
    overall["final_fit_pred_best_score"] = float(final_pred_best["pred_avg_main_score"])

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

    return oof, folds, overall


def main() -> None:
    args = parse_args()
    revisions = [x.strip() for x in args.expected_revisions.split(",") if x.strip()]
    cp_paths = find_checkpoint_paths(args.entropy_tables_root, args.avg_main_root, revisions)
    data, skipped = build_dataset(cp_paths)
    oof, folds, overall = run_logo_linear_regression(data)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_root, f"pythia410m_entropy3_linear_regression_logo_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    data_path = os.path.join(out_dir, "dataset_used.csv")
    oof_path = os.path.join(out_dir, "oof_predictions.csv")
    folds_path = os.path.join(out_dir, "fold_metrics.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    summary_md = os.path.join(out_dir, "summary.md")

    data.sort_values(["revision", "layer_idx"]).to_csv(data_path, index=False)
    oof.sort_values(["revision", "layer_idx"]).to_csv(oof_path, index=False)
    folds.to_csv(folds_path, index=False)

    summary = {
        "entropy_tables_root": args.entropy_tables_root,
        "avg_main_root": args.avg_main_root,
        "expected_revisions": revisions,
        "used_revisions": sorted(data["revision"].unique().tolist()),
        "skipped_revisions_missing_avg_main": sorted(skipped),
        "overall": overall,
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    with open(summary_md, "w") as f:
        f.write("# Entropy-to-Main Linear Regression (Leave-One-Checkpoint-Out)\n\n")
        f.write(f"- num_rows: {overall['num_rows']}\n")
        f.write(f"- num_checkpoints: {overall['num_checkpoints']}\n")
        f.write(f"- skipped_revisions_missing_avg_main: {sorted(skipped)}\n")
        f.write(f"- rmse_oof: {overall['rmse_oof']:.6f}\n")
        f.write(f"- mae_oof: {overall['mae_oof']:.6f}\n")
        f.write(f"- r2_oof: {overall['r2_oof']:.6f}\n")
        f.write(f"- best_layer_hit_rate: {overall['best_layer_hit_rate']:.4f}\n")
        f.write(f"- oof_pred_global_best_pair: {overall['oof_pred_global_best_pair']}\n")
        f.write(f"- oof_true_global_best_pair: {overall['oof_true_global_best_pair']}\n")
        f.write(f"- final_fit_pred_best_pair: {overall['final_fit_pred_best_pair']}\n\n")
        f.write("## Fold Metrics\n\n")
        f.write(folds.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Linear Coefficients\n\n")
        coef_df = pd.DataFrame(
            [{"feature": k, "coefficient": v} for k, v in overall["linear_coefficients"].items()]
        )
        f.write(coef_df.to_markdown(index=False))
        f.write("\n")

    print(out_dir)
    print(summary_md)
    print(summary_json)


if __name__ == "__main__":
    main()
