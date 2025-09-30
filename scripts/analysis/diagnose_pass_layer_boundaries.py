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
    component_bias_table,
    default_output_dir,
    evaluate_pair_selection,
    layer_axis_evaluation,
    load_pair_metrics,
    load_selector_summary,
    make_variant_weights,
    resolve_pass_run_artifacts,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PASS boundary and layer-range diagnostics.")
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
    parser.add_argument("--normalization_mode", type=str, default="global", choices=["global", "per_layer", "per_checkpoint", "two_way"])
    parser.add_argument("--variants", type=str, default="full_pass,static_only,time_only")
    parser.add_argument(
        "--layer_windows",
        type=str,
        default="",
        help="Comma-separated min:max windows, e.g. 0:5,0:10,15:24,17:21. Empty means use automatic windows.",
    )
    return parser.parse_args()


def parse_windows(df: pd.DataFrame, text: str) -> list[tuple[int, int]]:
    if text.strip():
        windows = []
        for part in text.split(","):
            lo, hi = part.strip().split(":")
            windows.append((int(lo), int(hi)))
        return windows
    lo = int(df["layer_idx"].min())
    hi = int(df["layer_idx"].max())
    mid = (lo + hi) // 2
    return [
        (lo, hi),
        (lo, min(lo + 5, hi)),
        (max(lo, hi - 5), hi),
        (max(lo, mid - 2), min(hi, mid + 2)),
    ]


def run_boundary_diagnostics(args: argparse.Namespace) -> dict[str, object]:
    artifacts = resolve_pass_run_artifacts(
        run_dir=args.run_dir,
        pair_metrics_path=args.pair_metrics_csv,
        model_size=args.model_size,
    )
    out_dir = default_output_dir(artifacts, "boundary_diagnostics", output_dir=args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_pair_metrics(artifacts.pair_metrics_path)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    windows = parse_windows(base_df, args.layer_windows)

    # Raw bias diagnostics.
    raw_cols = [
        "dispersion", "uniformity_score", "alignment", "pass_phase_score", "pass_volatility",
        "rankme", "spectral_slope", "top_pc_dominance", "effective_rank", "participation_ratio", "pass_score",
    ]
    bias_df = component_bias_table(base_df, raw_cols)
    bias_df.to_csv(out_dir / "boundary_component_bias.csv", index=False)

    rows = []
    layer_axis_rows = []
    for variant in variants:
        weights = make_variant_weights(variant)
        for lo, hi in windows:
            subset = base_df[(base_df["layer_idx"] >= lo) & (base_df["layer_idx"] <= hi)].copy()
            if subset.empty:
                continue
            scored = add_variant_score(subset, variant_name=variant, weights=weights, normalization_mode=args.normalization_mode)
            pair_summary = evaluate_pair_selection(scored, "ablation_score")
            pair_summary.update({
                "variant_name": variant,
                "normalization_mode": args.normalization_mode,
                "layer_min": int(lo),
                "layer_max": int(hi),
                "selected_is_window_min": int(pair_summary.get("selected_layer") == lo) if pair_summary.get("selected_layer") is not None else None,
                "selected_is_window_max": int(pair_summary.get("selected_layer") == hi) if pair_summary.get("selected_layer") is not None else None,
            })
            rows.append(pair_summary)

            layer_summary = layer_axis_evaluation(scored, "ablation_score")
            layer_summary.update({
                "variant_name": variant,
                "normalization_mode": args.normalization_mode,
                "layer_min": int(lo),
                "layer_max": int(hi),
            })
            layer_axis_rows.append(layer_summary)

    pd.DataFrame(rows).to_csv(out_dir / "boundary_window_pair_selection.csv", index=False)
    pd.DataFrame(layer_axis_rows).to_csv(out_dir / "boundary_window_layer_axis.csv", index=False)

    selector_summary = load_selector_summary(artifacts.summary_path)
    summary = {
        "run_dir": str(artifacts.run_dir),
        "pair_metrics_csv": str(artifacts.pair_metrics_path),
        "summary_json": str(artifacts.summary_path) if artifacts.summary_path else None,
        "selector_config": selector_summary.get("config", {}),
        "normalization_mode": args.normalization_mode,
        "variants": variants,
        "windows": windows,
        "outputs": {
            "boundary_component_bias": str(out_dir / "boundary_component_bias.csv"),
            "boundary_window_pair_selection": str(out_dir / "boundary_window_pair_selection.csv"),
            "boundary_window_layer_axis": str(out_dir / "boundary_window_layer_axis.csv"),
        },
    }
    write_summary_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run_boundary_diagnostics(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
