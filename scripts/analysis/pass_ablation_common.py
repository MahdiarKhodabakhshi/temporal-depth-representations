from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PASS_ABLATION_ROOT_NAME = "pass_ablations"
PAIR_METRICS_FILENAME = "pair_metrics.csv"
SUMMARY_FILENAME = "summary.json"

PASS_RUN_MARKERS = {
    "410m": [
        REPO_ROOT / "slurm_logs/latest_p410m_pass_global_pair_selector_root.txt",
        REPO_ROOT / "slurm_logs/latest_p410m_global_best_pair_root.txt",
    ],
    "70m": [
        REPO_ROOT / "slurm_logs/latest_p70m_pass_global_pair_selector_root.txt",
        REPO_ROOT / "slurm_logs/latest_p70m_global_best_pair_root.txt",
    ],
}

PASS_REQUIRED_COLUMNS = {
    "revision",
    "layer_idx",
    "avg_main_score",
    "dispersion",
    "uniformity_score",
    "alignment",
    "pass_phase_score",
    "pass_volatility",
    "pass_score",
}

PASS_RAW_COMPONENTS = {
    "D": "dispersion",
    "U": "uniformity_score",
    "A": "alignment",
    "P": "pass_phase_score",
    "V": "pass_volatility",
}

PASS_NORM_COMPONENTS = {
    "D": "pass_dispersion_norm",
    "U": "pass_uniformity_norm",
    "A": "pass_alignment_norm",
    "P": "pass_phase_norm",
    "V": "pass_volatility_norm",
}

DEFAULT_WEIGHTS = {
    "D": 1.0,
    "U": 0.75,
    "A": 0.75,
    "P": 1.0,
    "V": 0.5,
}

POSITIVE_COMPONENTS = {"D", "U", "P"}
NEGATIVE_COMPONENTS = {"A", "V"}
ALL_COMPONENTS = ("D", "U", "A", "P", "V")


@dataclass(frozen=True)
class PassRunArtifacts:
    run_dir: Path
    pair_metrics_path: Path
    summary_path: Path | None


def _normalize_model_size(model_size: str) -> str:
    return "".join(ch for ch in str(model_size).lower() if ch.isalnum())


def _marker_candidates(model_size: str) -> list[Path]:
    normalized = _normalize_model_size(model_size)
    markers = list(PASS_RUN_MARKERS.get(normalized, []))
    if normalized not in PASS_RUN_MARKERS:
        for values in PASS_RUN_MARKERS.values():
            markers.extend(values)
    else:
        for key, values in PASS_RUN_MARKERS.items():
            if key != normalized:
                markers.extend(values)
    return markers


def _candidate_run_dirs(search_root: Path, model_size: str) -> list[Path]:
    if not search_root.exists():
        return []

    candidate_dirs: dict[str, Path] = {}
    if (search_root / PAIR_METRICS_FILENAME).is_file():
        candidate_dirs[str(search_root.resolve())] = search_root.resolve()

    for pair_metrics_path in search_root.rglob(PAIR_METRICS_FILENAME):
        parent = pair_metrics_path.parent.resolve()
        candidate_dirs[str(parent)] = parent

    normalized_size = _normalize_model_size(model_size)
    candidates = list(candidate_dirs.values())
    if normalized_size:
        filtered = [path for path in candidates if normalized_size in str(path).lower()]
        if filtered:
            candidates = filtered

    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def resolve_pass_run_artifacts(
    run_dir: str | Path = "",
    pair_metrics_path: str | Path = "",
    model_size: str = "410m",
) -> PassRunArtifacts:
    if pair_metrics_path:
        resolved_pair_metrics = Path(pair_metrics_path).expanduser().resolve()
        if not resolved_pair_metrics.is_file():
            raise FileNotFoundError(f"pair_metrics.csv not found: {resolved_pair_metrics}")
        summary_path = resolved_pair_metrics.parent / SUMMARY_FILENAME
        return PassRunArtifacts(
            run_dir=resolved_pair_metrics.parent,
            pair_metrics_path=resolved_pair_metrics,
            summary_path=summary_path if summary_path.is_file() else None,
        )

    if run_dir:
        search_root = Path(run_dir).expanduser().resolve()
        candidates = _candidate_run_dirs(search_root, model_size=model_size)
        if not candidates:
            raise FileNotFoundError(
                f"Could not find {PAIR_METRICS_FILENAME} under run_dir {search_root}"
            )
        selected_run_dir = candidates[0]
        summary_path = selected_run_dir / SUMMARY_FILENAME
        return PassRunArtifacts(
            run_dir=selected_run_dir,
            pair_metrics_path=selected_run_dir / PAIR_METRICS_FILENAME,
            summary_path=summary_path if summary_path.is_file() else None,
        )

    for marker_path in _marker_candidates(model_size):
        if not marker_path.is_file():
            continue
        marker_target = Path(marker_path.read_text(encoding="utf-8").strip()).expanduser()
        if not marker_target.exists():
            continue
        candidates = _candidate_run_dirs(marker_target.resolve(), model_size=model_size)
        if candidates:
            selected_run_dir = candidates[0]
            summary_path = selected_run_dir / SUMMARY_FILENAME
            return PassRunArtifacts(
                run_dir=selected_run_dir,
                pair_metrics_path=selected_run_dir / PAIR_METRICS_FILENAME,
                summary_path=summary_path if summary_path.is_file() else None,
            )

    fallback_root = REPO_ROOT / "experiments" / "results_reruns"
    fallback_candidates = _candidate_run_dirs(fallback_root, model_size=model_size)
    if fallback_candidates:
        selected_run_dir = fallback_candidates[0]
        summary_path = selected_run_dir / SUMMARY_FILENAME
        return PassRunArtifacts(
            run_dir=selected_run_dir,
            pair_metrics_path=selected_run_dir / PAIR_METRICS_FILENAME,
            summary_path=summary_path if summary_path.is_file() else None,
        )

    raise FileNotFoundError(
        "Could not resolve a PASS run directory. Pass --run_dir or --pair_metrics_csv explicitly."
    )


def default_output_dir(
    artifacts: PassRunArtifacts,
    study_name: str,
    output_dir: str | Path = "",
) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return artifacts.run_dir / DEFAULT_PASS_ABLATION_ROOT_NAME / study_name


def default_output_root(
    artifacts: PassRunArtifacts,
    output_root: str | Path = "",
) -> Path:
    if output_root:
        return Path(output_root).expanduser().resolve()
    return artifacts.run_dir / DEFAULT_PASS_ABLATION_ROOT_NAME


def load_selector_summary(path: str | Path | None) -> dict[str, object]:
    if not path:
        return {}
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {}
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_summary_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_pair_metrics(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    required = PASS_REQUIRED_COLUMNS
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"pair_metrics is missing required columns: {missing}")
    if "revision_step" not in df.columns:
        df["revision_step"] = df["revision"].astype(str).str.extract(r"(\d+)").astype(float)
    df["layer_idx"] = df["layer_idx"].astype(int)
    return df


def safe_float(value: float | int | None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def robust_standardize(values: pd.Series, eps: float = 1e-12) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    finite = s[np.isfinite(s.to_numpy(dtype=float))]
    if finite.empty:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    median = float(finite.median())
    mad = float(np.median(np.abs(finite.to_numpy(dtype=float) - median)))
    if not math.isfinite(mad) or mad <= eps:
        out = pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    else:
        out = (s - median) / (mad + eps)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def normalize_component(
    df: pd.DataFrame,
    raw_col: str,
    mode: str,
    revision_col: str = "revision",
    layer_col: str = "layer_idx",
) -> pd.Series:
    if mode == "global":
        return robust_standardize(df[raw_col])

    if mode == "per_layer":
        return df.groupby(layer_col, group_keys=False)[raw_col].apply(robust_standardize)

    if mode == "per_checkpoint":
        return df.groupby(revision_col, group_keys=False)[raw_col].apply(robust_standardize)

    if mode == "two_way":
        s = pd.to_numeric(df[raw_col], errors="coerce")
        grand = float(s.mean(skipna=True))
        by_layer = df.groupby(layer_col)[raw_col].transform("mean")
        by_revision = df.groupby(revision_col)[raw_col].transform("mean")
        residual = s - by_layer - by_revision + grand
        return robust_standardize(residual)

    raise ValueError(f"Unsupported normalization mode: {mode}")


def build_component_table(df: pd.DataFrame, normalization_mode: str = "global") -> pd.DataFrame:
    out = df.copy()
    for short_name, raw_col in PASS_RAW_COMPONENTS.items():
        out[f"norm_{short_name}"] = normalize_component(out, raw_col, normalization_mode)
    return out


def make_variant_weights(kind: str) -> dict[str, float]:
    w = dict(DEFAULT_WEIGHTS)
    if kind == "full_pass":
        return w
    if kind == "drop_D":
        w["D"] = 0.0
        return w
    if kind == "drop_U":
        w["U"] = 0.0
        return w
    if kind == "drop_A":
        w["A"] = 0.0
        return w
    if kind == "drop_P":
        w["P"] = 0.0
        return w
    if kind == "drop_V":
        w["V"] = 0.0
        return w
    if kind == "static_only":
        w["P"] = 0.0
        w["V"] = 0.0
        return w
    if kind == "time_only":
        w["D"] = 0.0
        w["U"] = 0.0
        w["A"] = 0.0
        return w
    if kind == "D_only":
        return {k: (DEFAULT_WEIGHTS[k] if k == "D" else 0.0) for k in ALL_COMPONENTS}
    if kind == "U_only":
        return {k: (DEFAULT_WEIGHTS[k] if k == "U" else 0.0) for k in ALL_COMPONENTS}
    if kind == "A_only":
        return {k: (DEFAULT_WEIGHTS[k] if k == "A" else 0.0) for k in ALL_COMPONENTS}
    if kind == "P_only":
        return {k: (DEFAULT_WEIGHTS[k] if k == "P" else 0.0) for k in ALL_COMPONENTS}
    if kind == "V_only":
        return {k: (DEFAULT_WEIGHTS[k] if k == "V" else 0.0) for k in ALL_COMPONENTS}
    raise ValueError(f"Unsupported variant kind: {kind}")


def variant_kinds() -> list[str]:
    return [
        "full_pass",
        "drop_D",
        "drop_U",
        "drop_A",
        "drop_P",
        "drop_V",
        "static_only",
        "time_only",
        "D_only",
        "U_only",
        "A_only",
        "P_only",
        "V_only",
    ]


def score_from_weights(df: pd.DataFrame, weights: dict[str, float], prefix: str = "norm_") -> pd.Series:
    score = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    for component, weight in weights.items():
        col = f"{prefix}{component}"
        if col not in df.columns:
            raise ValueError(f"Missing normalized component column: {col}")
        sign = -1.0 if component in NEGATIVE_COMPONENTS else 1.0
        score = score + sign * float(weight) * df[col]
    return score


def add_variant_score(df: pd.DataFrame, variant_name: str, weights: dict[str, float], normalization_mode: str) -> pd.DataFrame:
    out = build_component_table(df, normalization_mode=normalization_mode)
    out["variant_name"] = variant_name
    out["normalization_mode"] = normalization_mode
    out["ablation_score"] = score_from_weights(out, weights)
    return out


def oracle_pair(df: pd.DataFrame) -> pd.Series:
    evaluable = df.dropna(subset=["avg_main_score"]).copy()
    if evaluable.empty:
        raise ValueError("No rows with avg_main_score available.")
    return evaluable.sort_values(["avg_main_score", "revision_step", "layer_idx"], ascending=[False, True, True]).iloc[0]


def oracle_checkpoint_table(df: pd.DataFrame) -> pd.DataFrame:
    evaluable = df.dropna(subset=["avg_main_score"]).copy()
    grp = evaluable.groupby("revision", as_index=False).agg(
        oracle_checkpoint_score=("avg_main_score", "max"),
        oracle_checkpoint_mean=("avg_main_score", "mean"),
        oracle_checkpoint_median=("avg_main_score", "median"),
        num_layers=("layer_idx", "nunique"),
    )
    return grp


def rank_series_desc(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(method="average", ascending=False)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    xs = pd.to_numeric(x, errors="coerce")
    ys = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(xs.to_numpy(dtype=float)) & np.isfinite(ys.to_numpy(dtype=float))
    if mask.sum() < 2:
        return float("nan")
    xr = xs[mask].rank(method="average")
    yr = ys[mask].rank(method="average")
    return float(xr.corr(yr, method="pearson"))


def safe_kendall(x: pd.Series, y: pd.Series) -> float:
    xs = pd.to_numeric(x, errors="coerce")
    ys = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(xs.to_numpy(dtype=float)) & np.isfinite(ys.to_numpy(dtype=float))
    if mask.sum() < 2:
        return float("nan")
    x_vals = xs[mask].to_numpy(dtype=float)
    y_vals = ys[mask].to_numpy(dtype=float)
    n = len(x_vals)
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x_vals[i] - x_vals[j]
            dy = y_vals[i] - y_vals[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
                continue
            if dy == 0:
                ties_y += 1
                continue
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def ndcg_at_k(df: pd.DataFrame, score_col: str, rel_col: str = "avg_main_score", k: int = 10) -> float:
    sub = df.dropna(subset=[score_col, rel_col]).copy()
    if sub.empty:
        return float("nan")
    ranked = sub.sort_values(score_col, ascending=False).head(k)
    rel = ranked[rel_col].to_numpy(dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = float(np.sum(rel * discounts))
    ideal = np.sort(sub[rel_col].to_numpy(dtype=float))[::-1][: len(rel)]
    idcg = float(np.sum(ideal * discounts))
    if idcg <= 0:
        return float("nan")
    return dcg / idcg


def hit_at_k(df: pd.DataFrame, score_col: str, target_revision: str, target_layer: int, k: int) -> int:
    ranked = df.dropna(subset=[score_col]).sort_values([score_col, "revision_step", "layer_idx"], ascending=[False, True, True]).head(k)
    match = ((ranked["revision"].astype(str) == str(target_revision)) & (ranked["layer_idx"].astype(int) == int(target_layer))).any()
    return int(bool(match))


def checkpoint_hit_at_k(checkpoint_scores: pd.DataFrame, score_col: str, target_revision: str, k: int) -> int:
    ranked = checkpoint_scores.dropna(subset=[score_col]).sort_values(score_col, ascending=False).head(k)
    return int((ranked["revision"].astype(str) == str(target_revision)).any())


def evaluate_pair_selection(df: pd.DataFrame, score_col: str) -> dict[str, float | int | str | None]:
    evaluable = df.dropna(subset=["avg_main_score", score_col]).copy()
    if evaluable.empty:
        return {"selected_revision": None}
    oracle = oracle_pair(evaluable)
    selected = evaluable.sort_values([score_col, "revision_step", "layer_idx"], ascending=[False, True, True]).iloc[0]
    out: dict[str, float | int | str | None] = {
        "selected_revision": str(selected["revision"]),
        "selected_layer": int(selected["layer_idx"]),
        "selected_score": float(selected[score_col]),
        "selected_avg_main_score": float(selected["avg_main_score"]),
        "oracle_revision": str(oracle["revision"]),
        "oracle_layer": int(oracle["layer_idx"]),
        "oracle_avg_main_score": float(oracle["avg_main_score"]),
        "delta_vs_oracle": float(selected["avg_main_score"] - oracle["avg_main_score"]),
        "ndcg_at_5": ndcg_at_k(evaluable, score_col, k=5),
        "ndcg_at_10": ndcg_at_k(evaluable, score_col, k=10),
        "pair_hit_at_1": hit_at_k(evaluable, score_col, str(oracle["revision"]), int(oracle["layer_idx"]), 1),
        "pair_hit_at_5": hit_at_k(evaluable, score_col, str(oracle["revision"]), int(oracle["layer_idx"]), 5),
        "pair_hit_at_10": hit_at_k(evaluable, score_col, str(oracle["revision"]), int(oracle["layer_idx"]), 10),
        "spearman_pairs": safe_spearman(evaluable[score_col], evaluable["avg_main_score"]),
        "kendall_pairs": safe_kendall(evaluable[score_col], evaluable["avg_main_score"]),
    }
    return out


def checkpoint_axis_evaluation(df: pd.DataFrame, score_col: str, aggregator: str) -> dict[str, float | int | str | None]:
    evaluable = df.dropna(subset=["avg_main_score", score_col]).copy()
    if evaluable.empty:
        return {"aggregator": aggregator}
    if aggregator == "max":
        score_table = evaluable.groupby("revision", as_index=False).agg(score=(score_col, "max"))
    elif aggregator == "mean":
        score_table = evaluable.groupby("revision", as_index=False).agg(score=(score_col, "mean"))
    elif aggregator == "median":
        score_table = evaluable.groupby("revision", as_index=False).agg(score=(score_col, "median"))
    else:
        raise ValueError(aggregator)
    oracle_table = oracle_checkpoint_table(evaluable)
    merged = score_table.merge(oracle_table, on="revision", how="inner")
    if merged.empty:
        return {"aggregator": aggregator}
    target_revision = oracle_table.sort_values(["oracle_checkpoint_score", "revision"], ascending=[False, True]).iloc[0]["revision"]
    merged = merged.sort_values(["score", "revision"], ascending=[False, True]).reset_index(drop=True)
    ckpt_rank = None
    match = merged.index[merged["revision"].astype(str) == str(target_revision)]
    if len(match) > 0:
        ckpt_rank = int(match[0] + 1)
    return {
        "aggregator": aggregator,
        "target_revision": str(target_revision),
        "spearman_checkpoint": safe_spearman(merged["score"], merged["oracle_checkpoint_score"]),
        "kendall_checkpoint": safe_kendall(merged["score"], merged["oracle_checkpoint_score"]),
        "checkpoint_hit_at_1": checkpoint_hit_at_k(merged, "score", str(target_revision), 1),
        "checkpoint_hit_at_3": checkpoint_hit_at_k(merged, "score", str(target_revision), 3),
        "checkpoint_hit_at_5": checkpoint_hit_at_k(merged, "score", str(target_revision), 5),
        "oracle_checkpoint_rank": ckpt_rank,
    }


def layer_axis_evaluation(df: pd.DataFrame, score_col: str) -> dict[str, float | int | None]:
    evaluable = df.dropna(subset=["avg_main_score", score_col]).copy()
    if evaluable.empty:
        return {}
    rows: list[dict[str, float | int | str | None]] = []
    for revision, group in evaluable.groupby("revision"):
        g = group.sort_values([score_col, "layer_idx"], ascending=[False, True]).copy()
        if g.empty:
            continue
        selected = g.iloc[0]
        oracle = group.sort_values(["avg_main_score", "layer_idx"], ascending=[False, True]).iloc[0]
        rows.append({
            "revision": str(revision),
            "selected_layer": int(selected["layer_idx"]),
            "oracle_layer": int(oracle["layer_idx"]),
            "layer_hit": int(int(selected["layer_idx"]) == int(oracle["layer_idx"])),
            "selected_avg_main_score": float(selected["avg_main_score"]),
            "oracle_avg_main_score": float(oracle["avg_main_score"]),
            "layer_gap": float(selected["avg_main_score"] - oracle["avg_main_score"]),
            "spearman_within_revision": safe_spearman(group[score_col], group["avg_main_score"]),
            "kendall_within_revision": safe_kendall(group[score_col], group["avg_main_score"]),
            "selected_is_min_layer": int(int(selected["layer_idx"]) == int(group["layer_idx"].min())),
            "selected_is_max_layer": int(int(selected["layer_idx"]) == int(group["layer_idx"].max())),
        })
    if not rows:
        return {}
    per_revision = pd.DataFrame(rows)
    return {
        "num_checkpoints": int(len(per_revision)),
        "layer_hit_rate": float(per_revision["layer_hit"].mean()),
        "mean_layer_gap": float(per_revision["layer_gap"].mean()),
        "median_layer_gap": float(per_revision["layer_gap"].median()),
        "mean_within_checkpoint_spearman": float(per_revision["spearman_within_revision"].mean()),
        "mean_within_checkpoint_kendall": float(per_revision["kendall_within_revision"].mean()),
        "boundary_min_rate": float(per_revision["selected_is_min_layer"].mean()),
        "boundary_max_rate": float(per_revision["selected_is_max_layer"].mean()),
    }


def per_revision_layer_table(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    evaluable = df.dropna(subset=["avg_main_score", score_col]).copy()
    rows = []
    for revision, group in evaluable.groupby("revision"):
        selected = group.sort_values([score_col, "layer_idx"], ascending=[False, True]).iloc[0]
        oracle = group.sort_values(["avg_main_score", "layer_idx"], ascending=[False, True]).iloc[0]
        rows.append({
            "revision": str(revision),
            "selected_layer": int(selected["layer_idx"]),
            "oracle_layer": int(oracle["layer_idx"]),
            "selected_avg_main_score": float(selected["avg_main_score"]),
            "oracle_avg_main_score": float(oracle["avg_main_score"]),
            "layer_gap": float(selected["avg_main_score"] - oracle["avg_main_score"]),
            "selected_score": float(selected[score_col]),
        })
    return pd.DataFrame(rows)


def variance_decomposition(df: pd.DataFrame, value_col: str) -> dict[str, float | str]:
    sub = df[["revision", "layer_idx", value_col]].dropna().copy()
    if sub.empty:
        return {"value_col": value_col}
    values = sub[value_col].to_numpy(dtype=float)
    total_var = float(np.var(values))
    by_checkpoint = sub.groupby("revision")[value_col].mean()
    by_layer = sub.groupby("layer_idx")[value_col].mean()
    ckpt_var = float(np.var(by_checkpoint.to_numpy(dtype=float)))
    layer_var = float(np.var(by_layer.to_numpy(dtype=float)))
    grand = float(np.mean(values))
    ckpt_mean = sub.groupby("revision")[value_col].transform("mean")
    layer_mean = sub.groupby("layer_idx")[value_col].transform("mean")
    residual = sub[value_col] - ckpt_mean - layer_mean + grand
    residual_var = float(np.var(residual.to_numpy(dtype=float)))
    denom = total_var if total_var > 1e-12 else float("nan")
    return {
        "value_col": value_col,
        "total_var": total_var,
        "checkpoint_var": ckpt_var,
        "layer_var": layer_var,
        "residual_var": residual_var,
        "checkpoint_var_fraction": ckpt_var / denom if math.isfinite(denom) else float("nan"),
        "layer_var_fraction": layer_var / denom if math.isfinite(denom) else float("nan"),
        "residual_var_fraction": residual_var / denom if math.isfinite(denom) else float("nan"),
    }


def component_bias_table(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        rows.append({
            "value_col": col,
            "spearman_vs_layer": safe_spearman(df[col], df["layer_idx"]),
            "kendall_vs_layer": safe_kendall(df[col], df["layer_idx"]),
            "spearman_vs_revision_step": safe_spearman(df[col], df["revision_step"]),
            "kendall_vs_revision_step": safe_kendall(df[col], df["revision_step"]),
        })
    return pd.DataFrame(rows)
