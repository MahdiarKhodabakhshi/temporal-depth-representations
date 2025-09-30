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
    add_variant_score,
    checkpoint_axis_evaluation,
    default_output_dir,
    evaluate_pair_selection,
    layer_axis_evaluation,
    load_pair_metrics,
    load_selector_summary,
    resolve_pass_run_artifacts,
    per_revision_layer_table,
    make_variant_weights,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PASS normalization ablations from a completed PASS selector run.")
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
    parser.add_argument("--variants", type=str, default="full_pass,static_only,time_only")
    parser.add_argument("--normalization_modes", type=str, default="global,per_layer,per_checkpoint,two_way")
    parser.add_argument("--checkpoint_aggregators", type=str, default="max,median,mean")
    return parser.parse_args()


def run_normalization_ablation(args: argparse.Namespace) -> dict[str, object]:
    artifacts = resolve_pass_run_artifacts(
        run_dir=args.run_dir,
        pair_metrics_path=args.pair_metrics_csv,
        model_size=args.model_size,
    )
    out_dir = default_output_dir(artifacts, "normalization_ablation", output_dir=args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_pair_metrics(artifacts.pair_metrics_path)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    norm_modes = [x.strip() for x in args.normalization_modes.split(",") if x.strip()]
    aggregators = [x.strip() for x in args.checkpoint_aggregators.split(",") if x.strip()]

    pair_rows = []
    checkpoint_rows = []
    layer_rows = []
    per_revision_rows = []

    for variant in variants:
        weights = make_variant_weights(variant)
        for norm_mode in norm_modes:
            scored = add_variant_score(base_df, variant_name=variant, weights=weights, normalization_mode=norm_mode)
            score_col = "ablation_score"

            pair_summary = evaluate_pair_selection(scored, score_col)
            pair_summary["variant_name"] = variant
            pair_summary["normalization_mode"] = norm_mode
            pair_rows.append(pair_summary)

            layer_summary = layer_axis_evaluation(scored, score_col)
            layer_summary["variant_name"] = variant
            layer_summary["normalization_mode"] = norm_mode
            layer_rows.append(layer_summary)

            pr = per_revision_layer_table(scored, score_col)
            if not pr.empty:
                pr.insert(0, "variant_name", variant)
                pr.insert(1, "normalization_mode", norm_mode)
                per_revision_rows.append(pr)

            for agg in aggregators:
                ckpt_summary = checkpoint_axis_evaluation(scored, score_col, aggregator=agg)
                ckpt_summary["variant_name"] = variant
                ckpt_summary["normalization_mode"] = norm_mode
                checkpoint_rows.append(ckpt_summary)

    pair_df = pd.DataFrame(pair_rows)
    checkpoint_df = pd.DataFrame(checkpoint_rows)
    layer_df = pd.DataFrame(layer_rows)
    per_revision_df = pd.concat(per_revision_rows, ignore_index=True) if per_revision_rows else pd.DataFrame()

    pair_df.to_csv(out_dir / "normalization_ablation_pair_selection.csv", index=False)
    checkpoint_df.to_csv(out_dir / "normalization_ablation_checkpoint_axis.csv", index=False)
    layer_df.to_csv(out_dir / "normalization_ablation_layer_axis.csv", index=False)
    per_revision_df.to_csv(out_dir / "normalization_ablation_per_revision_layers.csv", index=False)

    selector_summary = load_selector_summary(artifacts.summary_path)
    summary = {
        "run_dir": str(artifacts.run_dir),
        "pair_metrics_csv": str(artifacts.pair_metrics_path),
        "summary_json": str(artifacts.summary_path) if artifacts.summary_path else None,
        "selector_config": selector_summary.get("config", {}),
        "variants": variants,
        "normalization_modes": norm_modes,
        "outputs": {
            "pair_selection": str(out_dir / "normalization_ablation_pair_selection.csv"),
            "checkpoint_axis": str(out_dir / "normalization_ablation_checkpoint_axis.csv"),
            "layer_axis": str(out_dir / "normalization_ablation_layer_axis.csv"),
            "per_revision_layers": str(out_dir / "normalization_ablation_per_revision_layers.csv"),
        },
    }
    write_summary_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run_normalization_ablation(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
