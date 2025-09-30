from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.ablate_pass_components import run_component_ablation
from scripts.analysis.ablate_pass_normalization import run_normalization_ablation
from scripts.analysis.ablate_pass_two_stage_factorization import run_two_stage_ablation
from scripts.analysis.diagnose_pass_layer_boundaries import run_boundary_diagnostics
from scripts.analysis.pass_ablation_common import (
    default_output_root,
    resolve_pass_run_artifacts,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full offline PASS ablation suite from a completed PASS selector run."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Completed PASS selector output dir or containing job root. If omitted, resolve the latest PASS run.",
    )
    parser.add_argument("--pair_metrics_csv", "--pair_metrics", dest="pair_metrics_csv", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument(
        "--model_size",
        type=str,
        default="410m",
        help="Model size hint used only when resolving the latest PASS run automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = resolve_pass_run_artifacts(
        run_dir=args.run_dir,
        pair_metrics_path=args.pair_metrics_csv,
        model_size=args.model_size,
    )
    output_root = default_output_root(artifacts, output_root=args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    common = {
        "run_dir": str(artifacts.run_dir),
        "pair_metrics_csv": str(artifacts.pair_metrics_path),
        "model_size": args.model_size,
    }

    component_summary = run_component_ablation(
        SimpleNamespace(
            **common,
            output_dir=str(output_root / "component_ablation"),
            normalization_mode="global",
            checkpoint_aggregators="max,median,mean",
        )
    )
    normalization_summary = run_normalization_ablation(
        SimpleNamespace(
            **common,
            output_dir=str(output_root / "normalization_ablation"),
            variants="full_pass,static_only,time_only",
            normalization_modes="global,per_layer,per_checkpoint,two_way",
            checkpoint_aggregators="max,median,mean",
        )
    )
    two_stage_summary = run_two_stage_ablation(
        SimpleNamespace(
            **common,
            output_dir=str(output_root / "two_stage_ablation"),
            checkpoint_aggregators="max,median,mean",
            time_norm="global",
            layer_norm="per_checkpoint",
        )
    )
    boundary_summary = run_boundary_diagnostics(
        SimpleNamespace(
            **common,
            output_dir=str(output_root / "boundary_diagnostics"),
            normalization_mode="global",
            variants="full_pass,static_only,time_only",
            layer_windows="",
        )
    )

    summary = {
        "run_dir": str(artifacts.run_dir),
        "pair_metrics_csv": str(artifacts.pair_metrics_path),
        "summary_json": str(artifacts.summary_path) if artifacts.summary_path else None,
        "output_root": str(output_root),
        "execution_order": [
            "component_ablation",
            "normalization_ablation",
            "two_stage_ablation",
            "boundary_diagnostics",
        ],
        "studies": {
            "component_ablation": component_summary,
            "normalization_ablation": normalization_summary,
            "two_stage_ablation": two_stage_summary,
            "boundary_diagnostics": boundary_summary,
        },
    }
    write_summary_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
