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
    build_component_table,
    component_bias_table,
    default_output_dir,
    evaluate_pair_selection,
    checkpoint_axis_evaluation,
    layer_axis_evaluation,
    load_pair_metrics,
    load_selector_summary,
    resolve_pass_run_artifacts,
    per_revision_layer_table,
    variance_decomposition,
    variant_kinds,
    make_variant_weights,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PASS component ablations from a completed PASS selector pair_metrics.csv."
    )
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
    parser.add_argument(
        "--normalization_mode",
        type=str,
        default="global",
        choices=["global", "per_layer", "per_checkpoint", "two_way"],
    )
    parser.add_argument("--checkpoint_aggregators", type=str, default="max,median,mean")
    return parser.parse_args()


def run_component_ablation(args: argparse.Namespace) -> dict[str, object]:
    artifacts = resolve_pass_run_artifacts(
        run_dir=args.run_dir,
        pair_metrics_path=args.pair_metrics_csv,
        model_size=args.model_size,
    )
    output_dir = default_output_dir(artifacts, "component_ablation", output_dir=args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_pair_metrics(artifacts.pair_metrics_path)
    checkpoint_aggs = [x.strip() for x in args.checkpoint_aggregators.split(",") if x.strip()]

    component_rows = []
    checkpoint_rows = []
    layer_rows = []
    per_revision_rows = []

    for variant in variant_kinds():
        weights = make_variant_weights(variant)
        scored = add_variant_score(base_df, variant_name=variant, weights=weights, normalization_mode=args.normalization_mode)
        score_col = "ablation_score"

        pair_summary = evaluate_pair_selection(scored, score_col)
        pair_summary["variant_name"] = variant
        pair_summary["normalization_mode"] = args.normalization_mode
        component_rows.append(pair_summary)

        layer_summary = layer_axis_evaluation(scored, score_col)
        layer_summary["variant_name"] = variant
        layer_summary["normalization_mode"] = args.normalization_mode
        layer_rows.append(layer_summary)

        pr = per_revision_layer_table(scored, score_col)
        if not pr.empty:
            pr.insert(0, "variant_name", variant)
            pr.insert(1, "normalization_mode", args.normalization_mode)
            per_revision_rows.append(pr)

        for agg in checkpoint_aggs:
            ckpt_summary = checkpoint_axis_evaluation(scored, score_col, aggregator=agg)
            ckpt_summary["variant_name"] = variant
            ckpt_summary["normalization_mode"] = args.normalization_mode
            checkpoint_rows.append(ckpt_summary)

    component_df = pd.DataFrame(component_rows)
    checkpoint_df = pd.DataFrame(checkpoint_rows)
    layer_df = pd.DataFrame(layer_rows)
    per_revision_df = pd.concat(per_revision_rows, ignore_index=True) if per_revision_rows else pd.DataFrame()

    component_df.to_csv(output_dir / "component_ablation_pair_selection.csv", index=False)
    checkpoint_df.to_csv(output_dir / "component_ablation_checkpoint_axis.csv", index=False)
    layer_df.to_csv(output_dir / "component_ablation_layer_axis.csv", index=False)
    per_revision_df.to_csv(output_dir / "component_ablation_per_revision_layers.csv", index=False)

    scored_global = build_component_table(base_df, normalization_mode=args.normalization_mode)
    variance_rows = []
    for col in ["dispersion", "uniformity_score", "alignment", "pass_phase_score", "pass_volatility", "rankme", "spectral_slope", "pass_score"]:
        if col in scored_global.columns:
            variance_rows.append(variance_decomposition(scored_global, col))
    variance_df = pd.DataFrame(variance_rows)
    variance_df.to_csv(output_dir / "component_variance_decomposition.csv", index=False)

    bias_cols = [
        "dispersion", "uniformity_score", "alignment", "pass_phase_score", "pass_volatility",
        "rankme", "spectral_slope", "top_pc_dominance", "effective_rank", "participation_ratio", "pass_score",
    ]
    bias_df = component_bias_table(scored_global, bias_cols)
    bias_df.to_csv(output_dir / "component_axis_bias.csv", index=False)

    selector_summary = load_selector_summary(artifacts.summary_path)
    summary = {
        "run_dir": str(artifacts.run_dir),
        "pair_metrics_csv": str(artifacts.pair_metrics_path),
        "summary_json": str(artifacts.summary_path) if artifacts.summary_path else None,
        "selector_config": selector_summary.get("config", {}),
        "normalization_mode": args.normalization_mode,
        "variants": variant_kinds(),
        "outputs": {
            "pair_selection": str(output_dir / "component_ablation_pair_selection.csv"),
            "checkpoint_axis": str(output_dir / "component_ablation_checkpoint_axis.csv"),
            "layer_axis": str(output_dir / "component_ablation_layer_axis.csv"),
            "per_revision_layers": str(output_dir / "component_ablation_per_revision_layers.csv"),
            "variance_decomposition": str(output_dir / "component_variance_decomposition.csv"),
            "component_axis_bias": str(output_dir / "component_axis_bias.csv"),
        },
    }
    write_summary_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run_component_ablation(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
