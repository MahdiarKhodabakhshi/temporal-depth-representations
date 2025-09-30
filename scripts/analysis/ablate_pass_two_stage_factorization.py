from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.pass_ablation_common import (
    build_component_table,
    checkpoint_axis_evaluation,
    default_output_dir,
    layer_axis_evaluation,
    load_pair_metrics,
    load_selector_summary,
    resolve_pass_run_artifacts,
    safe_kendall,
    safe_spearman,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PASS two-stage checkpoint and layer factorization ablations.")
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Completed PASS selector output dir or containing job root. If omitted, resolve the latest PASS run.",
    )
    parser.add_argument("--pair_metrics_csv", "--pair_metrics", dest="pair_metrics_csv", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument(
        "--model_size",
        type=str,
        default="410m",
        help="Model size hint used only when resolving the latest PASS run automatically.",
    )
    parser.add_argument("--checkpoint_aggregators", type=str, default="max,median,mean")
    parser.add_argument("--time_norm", type=str, default="global", choices=["global", "per_layer", "per_checkpoint", "two_way"])
    parser.add_argument("--layer_norm", type=str, default="per_checkpoint", choices=["global", "per_layer", "per_checkpoint", "two_way"])
    return parser.parse_args()


def choose_checkpoint(df: pd.DataFrame, agg: str) -> pd.DataFrame:
    if agg == "max":
        out = df.groupby("revision", as_index=False).agg(checkpoint_score=("checkpoint_score", "max"))
    elif agg == "mean":
        out = df.groupby("revision", as_index=False).agg(checkpoint_score=("checkpoint_score", "mean"))
    elif agg == "median":
        out = df.groupby("revision", as_index=False).agg(checkpoint_score=("checkpoint_score", "median"))
    else:
        raise ValueError(agg)
    return out.sort_values(["checkpoint_score", "revision"], ascending=[False, True]).reset_index(drop=True)


def run_two_stage_ablation(args: argparse.Namespace) -> dict[str, object]:
    artifacts = resolve_pass_run_artifacts(
        run_dir=args.run_dir,
        pair_metrics_path=args.pair_metrics_csv,
        model_size=args.model_size,
    )
    out_dir = default_output_dir(artifacts, "two_stage_ablation", output_dir=args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_pair_metrics(artifacts.pair_metrics_path)
    checkpoint_aggs = [x.strip() for x in args.checkpoint_aggregators.split(",") if x.strip()]

    # time-only part for checkpoint choice
    time_df = build_component_table(base_df, normalization_mode=args.time_norm)
    time_df["checkpoint_score"] = time_df["norm_P"] - 0.5 * time_df["norm_V"]

    # static-only part for layer choice
    layer_df = build_component_table(base_df, normalization_mode=args.layer_norm)
    layer_df["layer_score"] = layer_df["norm_D"] + 0.75 * layer_df["norm_U"] - 0.75 * layer_df["norm_A"]

    checkpoint_rows = []
    selected_rows = []

    oracle_checkpoint_scores = base_df.groupby("revision", as_index=False).agg(oracle_checkpoint_score=("avg_main_score", "max"))

    for agg in checkpoint_aggs:
        ckpt_ranked = choose_checkpoint(time_df, agg)
        chosen_revision = str(ckpt_ranked.iloc[0]["revision"])

        checkpoint_eval_table = ckpt_ranked.merge(oracle_checkpoint_scores, on="revision", how="inner")
        checkpoint_rows.append({
            "checkpoint_aggregator": agg,
            "time_norm": args.time_norm,
            "layer_norm": args.layer_norm,
            "chosen_revision": chosen_revision,
            "checkpoint_spearman": safe_spearman(checkpoint_eval_table["checkpoint_score"], checkpoint_eval_table["oracle_checkpoint_score"]),
            "checkpoint_kendall": safe_kendall(checkpoint_eval_table["checkpoint_score"], checkpoint_eval_table["oracle_checkpoint_score"]),
        })

        subset = layer_df[layer_df["revision"].astype(str) == chosen_revision].copy()
        subset = subset.dropna(subset=["avg_main_score", "layer_score"])
        if subset.empty:
            continue
        selected_layer_row = subset.sort_values(["layer_score", "layer_idx"], ascending=[False, True]).iloc[0]
        oracle_layer_row = subset.sort_values(["avg_main_score", "layer_idx"], ascending=[False, True]).iloc[0]

        selected_rows.append({
            "checkpoint_aggregator": agg,
            "time_norm": args.time_norm,
            "layer_norm": args.layer_norm,
            "chosen_revision": chosen_revision,
            "selected_layer": int(selected_layer_row["layer_idx"]),
            "oracle_layer_in_chosen_checkpoint": int(oracle_layer_row["layer_idx"]),
            "selected_avg_main_score": float(selected_layer_row["avg_main_score"]),
            "oracle_avg_main_score_in_chosen_checkpoint": float(oracle_layer_row["avg_main_score"]),
            "layer_gap_in_chosen_checkpoint": float(selected_layer_row["avg_main_score"] - oracle_layer_row["avg_main_score"]),
            "layer_hit_in_chosen_checkpoint": int(int(selected_layer_row["layer_idx"]) == int(oracle_layer_row["layer_idx"])),
            "layer_spearman_in_chosen_checkpoint": safe_spearman(subset["layer_score"], subset["avg_main_score"]),
            "layer_kendall_in_chosen_checkpoint": safe_kendall(subset["layer_score"], subset["avg_main_score"]),
        })

    checkpoint_df = pd.DataFrame(checkpoint_rows)
    selected_df = pd.DataFrame(selected_rows)
    checkpoint_df.to_csv(out_dir / "two_stage_checkpoint_eval.csv", index=False)
    selected_df.to_csv(out_dir / "two_stage_selected_pairs.csv", index=False)

    # Optional direct layer-axis evaluation for static-only metric across all checkpoints.
    static_global = layer_df.copy()
    static_global["static_only_score"] = static_global["layer_score"]
    layer_axis = layer_axis_evaluation(static_global, "static_only_score")
    layer_axis_df = pd.DataFrame([layer_axis])
    layer_axis_df.to_csv(out_dir / "two_stage_static_layer_axis.csv", index=False)

    # Optional direct checkpoint-axis evaluation for time-only metric across all checkpoints.
    checkpoint_axis_rows = []
    for agg in checkpoint_aggs:
        checkpoint_axis = checkpoint_axis_evaluation(time_df.rename(columns={"checkpoint_score": "time_only_score"}), "time_only_score", agg)
        checkpoint_axis["time_norm"] = args.time_norm
        checkpoint_axis["layer_norm"] = args.layer_norm
        checkpoint_axis_rows.append(checkpoint_axis)
    pd.DataFrame(checkpoint_axis_rows).to_csv(out_dir / "two_stage_time_checkpoint_axis.csv", index=False)

    selector_summary = load_selector_summary(artifacts.summary_path)
    summary = {
        "run_dir": str(artifacts.run_dir),
        "pair_metrics_csv": str(artifacts.pair_metrics_path),
        "summary_json": str(artifacts.summary_path) if artifacts.summary_path else None,
        "selector_config": selector_summary.get("config", {}),
        "time_norm": args.time_norm,
        "layer_norm": args.layer_norm,
        "checkpoint_aggregators": checkpoint_aggs,
        "outputs": {
            "two_stage_checkpoint_eval": str(out_dir / "two_stage_checkpoint_eval.csv"),
            "two_stage_selected_pairs": str(out_dir / "two_stage_selected_pairs.csv"),
            "two_stage_static_layer_axis": str(out_dir / "two_stage_static_layer_axis.csv"),
            "two_stage_time_checkpoint_axis": str(out_dir / "two_stage_time_checkpoint_axis.csv"),
        },
    }
    write_summary_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run_two_stage_ablation(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
