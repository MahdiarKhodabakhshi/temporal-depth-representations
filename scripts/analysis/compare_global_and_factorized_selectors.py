#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments" / "results_reruns"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a completed global selector run against a factorized selector summary."
    )
    parser.add_argument("--global-summary", type=str, required=True, help="Path to global selector summary.json")
    parser.add_argument("--factorized-summary", type=str, required=True, help="Path to factorized selector summary.json")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory under which the comparison output folder will be created.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    global_summary = Path(args.global_summary).expanduser().resolve()
    factorized_summary = Path(args.factorized_summary).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    global_obj = load_json(global_summary)
    factorized_obj = load_json(factorized_summary)

    baselines = global_obj["baselines"]
    oracle = baselines["oracle_best_pair"]
    baseline_best = baselines["baseline_revision_best_layer"]
    baseline_last = baselines["baseline_revision_last_layer"]

    rows: list[dict[str, object]] = []

    def add_static_row(method: str, revision: str, layer: int, score: float, notes: str = "") -> None:
        rows.append(
            {
                "group": "static_global_32_tasks",
                "method": method,
                "revision": revision,
                "layer": int(layer),
                "avg_main_score": float(score),
                "num_tasks": 32,
                "delta_vs_oracle_static": float(score - oracle["avg_main_score"]),
                "delta_vs_step143000_best": float(score - baseline_best["avg_main_score"]),
                "delta_vs_step143000_last": float(score - baseline_last["avg_main_score"]),
                "notes": notes,
            }
        )

    add_static_row(
        "oracle_static_best_pair",
        oracle["revision"],
        oracle["layer"],
        oracle["avg_main_score"],
        "Best static (checkpoint, layer) from already-computed average-main tables.",
    )
    add_static_row(
        "baseline_revision_best_layer",
        baseline_best["revision"],
        baseline_best["layer"],
        baseline_best["avg_main_score"],
        "Best fixed layer in the baseline checkpoint.",
    )
    add_static_row(
        "baseline_revision_last_layer",
        baseline_last["revision"],
        baseline_last["layer"],
        baseline_last["avg_main_score"],
        "Last layer in the baseline checkpoint.",
    )

    for row in global_obj["rule_results"]:
        add_static_row(
            f"global_selector::{row['rule']}",
            row["selected_revision"],
            int(row["selected_layer"]),
            float(row["selected_avg_main_score"]),
            "Global unsupervised selector result.",
        )

    rows.append(
        {
            "group": "taskwise_factorized_summary",
            "method": "factorized_selector::best_ablation",
            "revision": "mixed_per_task",
            "layer": "mixed_per_task",
            "avg_main_score": float(factorized_obj["selected_avg_main_score"]),
            "num_tasks": int(factorized_obj["num_tasks_selected"]),
            "delta_vs_oracle_static": None,
            "delta_vs_step143000_best": float(
                factorized_obj["selected_avg_main_score"] - factorized_obj["baselines"]["same_task_best_avg_main"]
            ),
            "delta_vs_step143000_last": float(
                factorized_obj["selected_avg_main_score"] - factorized_obj["baselines"]["same_task_last_layer_avg_main"]
            ),
            "notes": (
                "Not directly comparable to the static-global rows if task coverage differs. "
                "Interpret this as a taskwise factorized-selector summary."
            ),
        }
    )

    df = pd.DataFrame(rows)
    out_root = output_root / f"pythia410m_global_selector_comparison_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / "comparison_table.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    static_df = df[df["group"] == "static_global_32_tasks"].sort_values("avg_main_score", ascending=False)
    best_rule = max(global_obj["rule_results"], key=lambda row: float(row["selected_avg_main_score"]))
    fact_row = df[df["group"] == "taskwise_factorized_summary"].iloc[0]

    md_lines = [
        "# Pythia-410m Global Selector Comparison",
        "",
        "## Static Global Results",
        "",
        "| Method | Revision | Layer | Avg Main Score | Delta vs Oracle Static | Delta vs Baseline Best | Delta vs Baseline Last |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for _, row in static_df.iterrows():
        md_lines.append(
            f"| {row['method']} | {row['revision']} | {row['layer']} | {row['avg_main_score']:.10f} | "
            f"{row['delta_vs_oracle_static']:+.10f} | {row['delta_vs_step143000_best']:+.10f} | "
            f"{row['delta_vs_step143000_last']:+.10f} |"
        )

    md_lines.extend(
        [
            "",
            "## Factorized Summary",
            "",
            "| Method | Avg Main Score | Num Tasks | Delta vs Baseline Best | Delta vs Baseline Last | Notes |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            (
                f"| {fact_row['method']} | {fact_row['avg_main_score']:.10f} | {int(fact_row['num_tasks'])} | "
                f"{fact_row['delta_vs_step143000_best']:+.10f} | {fact_row['delta_vs_step143000_last']:+.10f} | "
                f"{fact_row['notes']} |"
            ),
            "",
            "## Conclusion",
            "",
            (
                f"- Best global-selector rule in this run: `{best_rule['rule']}` -> "
                f"`{best_rule['selected_revision']}/layer_{int(best_rule['selected_layer'])}` with "
                f"`{best_rule['selected_avg_main_score']:.10f}`."
            ),
            (
                f"- Delta vs oracle static best pair: `{best_rule['delta_vs_oracle']:+.10f}`. "
                f"Delta vs baseline best layer: `{best_rule['delta_vs_baseline_best']:+.10f}`."
            ),
            "- Interpret the factorized row separately if its task set differs from the static-global run.",
        ]
    )

    (out_root / "comparison_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    with open(out_root / "comparison_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "global_summary": str(global_summary),
                "factorized_summary": str(factorized_summary),
                "output_dir": str(out_root),
            },
            f,
            indent=2,
        )

    print(str(out_root))


if __name__ == "__main__":
    main()
