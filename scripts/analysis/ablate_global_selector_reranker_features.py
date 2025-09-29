from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.select_global_pair_unsupervised import (
    DEFAULT_RERANKER_FEATURES,
    RERANKER_FEATURE_SPECS,
    _build_shortlist_from_rule,
    _rerank_shortlist,
    _resolve_two_stage_shortlist_rule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline reranker ablations for a completed 410m global-selector run. "
            "This reuses pair_metrics.csv and summary.json, so it does not require a GPU."
        )
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help=(
            "Completed global-selector run directory. If omitted, resolve the latest run "
            "under the latest global-selector rule-comparison job root."
        ),
    )
    parser.add_argument("--pair_metrics", type=str, default="", help="Optional override for pair_metrics.csv.")
    parser.add_argument("--summary", type=str, default="", help="Optional override for summary.json.")
    parser.add_argument(
        "--shortlist_rule",
        type=str,
        default="auto",
        choices=["auto", "alignment_uniformity", "dispersion_only"],
        help="Shortlist generator to rerank offline. `auto` follows the model-size heuristic used by two_stage.",
    )
    parser.add_argument(
        "--reranker_shortlist_size",
        type=int,
        default=0,
        help="Override shortlist size. 0 means reuse the completed run config.",
    )
    parser.add_argument(
        "--alignment_quantile",
        type=float,
        default=-1.0,
        help="Override shortlist alignment quantile. Negative means reuse the completed run config.",
    )
    parser.add_argument(
        "--reranker_fusion",
        type=str,
        default="",
        choices=["", "rrf", "borda"],
        help="Override reranker fusion method. Empty means reuse the completed run config.",
    )
    parser.add_argument(
        "--reranker_rrf_k",
        type=int,
        default=0,
        help="Override RRF k. 0 means reuse the completed run config.",
    )
    parser.add_argument(
        "--feature_sets",
        type=str,
        default="current_full;geometry_only;geometry_plus_entropy;recommended_pruned;stability_only;spectral_local_only;pass_only;pass_plus_geometry;pass_phase_bundle",
        help=(
            "Semicolon-separated reranker feature sets. Each item is either a preset name or "
            "`name=feat1+feat2+...`. Presets are documented in the output meta.json."
        ),
    )
    return parser.parse_args()


def _resolve_latest_run_dir() -> Path:
    candidate_root_markers = [
        ROOT / "slurm_logs/latest_p410m_global_selector_rule_comparison_root.txt",
        ROOT / "slurm_logs/latest_p410m_two_stage_compare_root.txt",
    ]
    latest_root_path = next((path for path in candidate_root_markers if path.is_file()), None)
    if latest_root_path is None:
        raise FileNotFoundError(
            "Could not resolve a default run_dir because "
            "no latest global-selector run marker exists."
        )

    run_root = Path(latest_root_path.read_text(encoding="utf-8").strip())
    if not run_root.is_dir():
        raise FileNotFoundError(f"Latest run root does not exist: {run_root}")

    candidate_dirs: dict[str, Path] = {}
    for pattern in (
        "pythia410m_global_pair_selector_unsupervised_*",
        "pythia410m_global_best_pair_unsup_*",
    ):
        for path in run_root.glob(pattern):
            candidate_dirs[str(path)] = path

    candidates = sorted(candidate_dirs.values(), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No completed run directories found under {run_root}")
    return candidates[0]


def _build_feature_set_presets(summary_payload: dict[str, Any]) -> dict[str, list[str]]:
    current_full = list(summary_payload.get("config", {}).get("reranker_features", DEFAULT_RERANKER_FEATURES))
    return {
        "current_full": current_full,
        "geometry_only": ["dispersion_stable", "alignment", "uniformity_score"],
        "geometry_plus_entropy": ["dispersion_stable", "alignment", "uniformity_score", "entropy_tiebreak_score"],
        "recommended_pruned": [
            "dispersion_stable",
            "alignment",
            "uniformity_score",
            "checkpoint_neighbor_stability",
            "layer_neighbor_stability",
        ],
        "stability_only": ["checkpoint_neighbor_stability", "layer_neighbor_stability"],
        "spectral_local_only": ["top_pc_dominance", "effective_rank", "knn_aug_stability"],
        "pass_only": ["pass_score"],
        "pass_plus_geometry": ["pass_score", "dispersion_stable", "alignment", "uniformity_score"],
        "pass_phase_bundle": ["rankme", "spectral_slope", "pass_phase_score", "pass_volatility"],
    }


def _parse_feature_sets(spec: str, presets: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    parsed: list[tuple[str, list[str]]] = []
    seen_names: set[str] = set()

    for chunk in [x.strip() for x in spec.split(";") if x.strip()]:
        if "=" in chunk:
            name, feature_expr = chunk.split("=", 1)
            name = name.strip()
            features = [x.strip() for x in feature_expr.split("+") if x.strip()]
        else:
            name = chunk
            if name not in presets:
                raise ValueError(f"Unknown feature set preset: {name}")
            features = list(presets[name])

        if not name:
            raise ValueError(f"Invalid feature set specification: {chunk}")
        if name in seen_names:
            raise ValueError(f"Duplicate feature set name: {name}")
        invalid = sorted(set(features) - set(RERANKER_FEATURE_SPECS))
        if invalid:
            raise ValueError(f"Feature set {name} contains unsupported features: {invalid}")

        seen_names.add(name)
        parsed.append((name, features))

    if not parsed:
        raise ValueError("No feature sets were requested.")
    return parsed


def _load_reference_results(run_dir: Path, summary_payload: dict[str, Any]) -> pd.DataFrame:
    selected_path = run_dir / "selected_rule_pairs.csv"
    if selected_path.is_file():
        return pd.read_csv(selected_path)

    rule_results = summary_payload.get("rule_results", [])
    if not rule_results:
        return pd.DataFrame()
    return pd.DataFrame(rule_results)


def _format_pair(revision: str, layer_idx: int) -> str:
    return f"{revision}/layer_{int(layer_idx)}"


def _spearman_summary(shortlist: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_name in features:
        if feature_name not in shortlist.columns:
            continue
        values = shortlist[feature_name]
        if not values.notna().any():
            continue
        corr = values.corr(shortlist["avg_main_score"], method="spearman")
        rows.append(
            {
                "feature_name": feature_name,
                "expected_direction": "lower_better"
                if RERANKER_FEATURE_SPECS[feature_name]["ascending"]
                else "higher_better",
                "spearman_with_avg_main": float(corr) if pd.notna(corr) else None,
                "feature_min": float(values.min()),
                "feature_max": float(values.max()),
                "feature_std": float(values.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).expanduser() if args.run_dir.strip() else _resolve_latest_run_dir()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    pair_metrics_path = Path(args.pair_metrics).expanduser() if args.pair_metrics.strip() else run_dir / "pair_metrics.csv"
    summary_path = Path(args.summary).expanduser() if args.summary.strip() else run_dir / "summary.json"
    if not pair_metrics_path.is_file():
        raise FileNotFoundError(f"pair_metrics.csv not found: {pair_metrics_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")

    pair_df = pd.read_csv(pair_metrics_path)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_payload = json.load(f)

    config = summary_payload["config"]
    model_size = str(config["model_size"])
    alignment_quantile = float(config["alignment_quantile"]) if args.alignment_quantile < 0 else float(args.alignment_quantile)
    reranker_shortlist_size = int(config.get("reranker_shortlist_size", config.get("top_m_pairs", 20))) if args.reranker_shortlist_size <= 0 else int(args.reranker_shortlist_size)
    reranker_fusion = str(config.get("reranker_fusion", "rrf")) if not args.reranker_fusion else str(args.reranker_fusion)
    reranker_rrf_k = int(config.get("reranker_rrf_k", 60)) if args.reranker_rrf_k <= 0 else int(args.reranker_rrf_k)
    shortlist_rule = (
        _resolve_two_stage_shortlist_rule("two_stage", model_size)
        if args.shortlist_rule == "auto"
        else args.shortlist_rule
    )

    presets = _build_feature_set_presets(summary_payload)
    feature_sets = _parse_feature_sets(args.feature_sets, presets)
    reference_results_df = _load_reference_results(run_dir, summary_payload)

    evaluable = pair_df.dropna(subset=["avg_main_score"]).copy()
    if evaluable.empty:
        raise ValueError("No evaluable rows were found in pair_metrics.csv.")

    oracle = evaluable.sort_values(
        ["avg_main_score", "revision_step", "layer_idx"],
        ascending=[False, True, True],
    ).iloc[0]
    baseline_best = summary_payload["baselines"]["baseline_revision_best_layer"]
    baseline_last = summary_payload["baselines"]["baseline_revision_last_layer"]

    shortlist, shortlist_meta = _build_shortlist_from_rule(
        evaluable,
        shortlist_rule=shortlist_rule,
        shortlist_size=reranker_shortlist_size,
        alignment_quantile=alignment_quantile,
    )
    shortlist = shortlist.copy().reset_index(drop=True)
    shortlist["shortlist_rank"] = range(1, len(shortlist) + 1)
    shortlist_best = shortlist.sort_values(
        ["avg_main_score", "revision_step", "layer_idx"],
        ascending=[False, True, True],
    ).iloc[0]
    feature_spearman_df = _spearman_summary(shortlist, list(RERANKER_FEATURE_SPECS))

    reference_rule = "alignment_uniformity"
    reference_row = reference_results_df[reference_results_df["rule"] == reference_rule]
    reference_selected = reference_row.iloc[0] if not reference_row.empty else None

    rows: list[dict[str, Any]] = []
    reranked_rows: list[pd.DataFrame] = []
    for feature_set_name, features in feature_sets:
        reranked, rerank_meta = _rerank_shortlist(
            shortlist,
            reranker_features=features,
            reranker_fusion=reranker_fusion,
            reranker_rrf_k=reranker_rrf_k,
        )
        selected = reranked.iloc[0]

        row = {
            "feature_set": feature_set_name,
            "feature_names": features,
            "selected_revision": str(selected["revision"]),
            "selected_layer": int(selected["layer_idx"]),
            "selected_pair": _format_pair(str(selected["revision"]), int(selected["layer_idx"])),
            "selected_avg_main_score": float(selected["avg_main_score"]),
            "delta_vs_oracle": float(selected["avg_main_score"] - oracle["avg_main_score"]),
            "delta_vs_baseline_best": float(selected["avg_main_score"] - baseline_best["avg_main_score"]),
            "delta_vs_baseline_last": float(selected["avg_main_score"] - baseline_last["avg_main_score"]),
            "delta_vs_shortlist_best": float(selected["avg_main_score"] - shortlist_best["avg_main_score"]),
            "selected_shortlist_rank": int(selected["shortlist_rank"]),
            "selected_reranker_rank": int(selected["reranker_rank"]),
            "selected_reranker_score": float(selected["reranker_score"]) if pd.notna(selected["reranker_score"]) else None,
            "same_as_oracle": bool(
                str(selected["revision"]) == str(oracle["revision"]) and int(selected["layer_idx"]) == int(oracle["layer_idx"])
            ),
            "same_as_shortlist_best": bool(
                str(selected["revision"]) == str(shortlist_best["revision"]) and int(selected["layer_idx"]) == int(shortlist_best["layer_idx"])
            ),
            "num_features": int(len(features)),
        }
        if reference_selected is not None:
            row["delta_vs_alignment_uniformity"] = float(selected["avg_main_score"] - float(reference_selected["selected_avg_main_score"]))
            row["same_as_alignment_uniformity"] = bool(
                str(selected["revision"]) == str(reference_selected["selected_revision"])
                and int(selected["layer_idx"]) == int(reference_selected["selected_layer"])
            )
        else:
            row["delta_vs_alignment_uniformity"] = None
            row["same_as_alignment_uniformity"] = None
        rows.append(row)

        reranked_export = reranked.copy()
        reranked_export["feature_set"] = feature_set_name
        reranked_export["feature_names"] = json.dumps(features)
        reranked_export["selected_by_feature_set"] = (
            (reranked_export["revision"].astype(str) == str(selected["revision"]))
            & (reranked_export["layer_idx"].astype(int) == int(selected["layer_idx"]))
        )
        reranked_rows.append(reranked_export)

    results_df = pd.DataFrame(rows).sort_values(
        ["selected_avg_main_score", "selected_shortlist_rank", "feature_set"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    reranked_df = pd.concat(reranked_rows, ignore_index=True)

    out_dir = ROOT / "experiments/results_reruns" / f"pythia410m_offline_reranker_ablation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(out_dir / "offline_reranker_results.csv", index=False)
    reranked_df.to_csv(out_dir / "offline_reranker_shortlists.csv", index=False)
    shortlist.to_csv(out_dir / "base_shortlist.csv", index=False)
    feature_spearman_df.to_csv(out_dir / "shortlist_feature_spearman.csv", index=False)

    lines = [
        "# Offline 410m Reranker Ablation",
        "",
        f"- source run: `{run_dir}`",
        f"- shortlist rule: `{shortlist_rule}`",
        f"- shortlist size: `{reranker_shortlist_size}`",
        f"- reranker fusion: `{reranker_fusion}`",
        f"- reranker RRF k: `{reranker_rrf_k}`",
        f"- alignment quantile: `{alignment_quantile}`",
        f"- oracle: `{_format_pair(str(oracle['revision']), int(oracle['layer_idx']))}` -> `{float(oracle['avg_main_score']):.10f}`",
        f"- shortlist best by avg_main: `{_format_pair(str(shortlist_best['revision']), int(shortlist_best['layer_idx']))}` -> `{float(shortlist_best['avg_main_score']):.10f}`",
    ]

    if reference_selected is not None:
        lines.append(
            f"- reference `alignment_uniformity`: `{_format_pair(str(reference_selected['selected_revision']), int(reference_selected['selected_layer']))}` -> `{float(reference_selected['selected_avg_main_score']):.10f}`"
        )

    lines.extend(
        [
            "",
            "## Feature Set Results",
            "",
            "| Feature Set | Selected Pair | Avg Main | Delta vs Oracle | Delta vs alignment_uniformity | Delta vs Shortlist Best | Shortlist Rank |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in results_df.iterrows():
        delta_vs_alignment = (
            f"{float(row['delta_vs_alignment_uniformity']):+.10f}"
            if pd.notna(row["delta_vs_alignment_uniformity"])
            else "n/a"
        )
        lines.append(
            f"| `{row['feature_set']}` | `{row['selected_pair']}` | {float(row['selected_avg_main_score']):.10f} "
            f"| {float(row['delta_vs_oracle']):+.10f} | {delta_vs_alignment} "
            f"| {float(row['delta_vs_shortlist_best']):+.10f} | {int(row['selected_shortlist_rank'])} |"
        )

    if not feature_spearman_df.empty:
        lines.extend(
            [
                "",
                "## Shortlist Feature Spearman",
                "",
                "| Feature | Expected Direction | Spearman vs Avg Main |",
                "| --- | --- | ---: |",
            ]
        )
        for _, row in feature_spearman_df.sort_values("spearman_with_avg_main", ascending=False).iterrows():
            score_text = "n/a" if pd.isna(row["spearman_with_avg_main"]) else f"{float(row['spearman_with_avg_main']):+.3f}"
            lines.append(
                f"| `{row['feature_name']}` | `{row['expected_direction']}` | {score_text} |"
            )

    (out_dir / "offline_reranker_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "pair_metrics": str(pair_metrics_path),
                "summary": str(summary_path),
                "shortlist_rule": shortlist_rule,
                "shortlist_meta": shortlist_meta,
                "reranker_shortlist_size": reranker_shortlist_size,
                "alignment_quantile": alignment_quantile,
                "reranker_fusion": reranker_fusion,
                "reranker_rrf_k": reranker_rrf_k,
                "feature_set_presets": presets,
                "feature_sets": [{"name": name, "features": features} for name, features in feature_sets],
                "oracle_pair": {
                    "revision": str(oracle["revision"]),
                    "layer_idx": int(oracle["layer_idx"]),
                    "avg_main_score": float(oracle["avg_main_score"]),
                },
                "shortlist_best_pair": {
                    "revision": str(shortlist_best["revision"]),
                    "layer_idx": int(shortlist_best["layer_idx"]),
                    "avg_main_score": float(shortlist_best["avg_main_score"]),
                },
                "output_dir": str(out_dir),
            },
            f,
            indent=2,
        )

    print(str(out_dir))


if __name__ == "__main__":
    main()
