"""
Compare post-hoc isotropy-correction variants for global checkpoint-layer pair
selection without modifying the existing selector scripts.

This script keeps the previous methods intact and evaluates a new family of
post-pooling transforms before scoring pairs:

1. raw embeddings
2. mean-center + remove top-r principal components
3. whitening, optionally with dimensionality reduction
4. shortlist-only normalizing-flow isotropization via an affine-coupling flow

The downstream selection rules remain the same geometry-based unsupervised rules
used by the original selector. Outputs are emitted in a long format with an
`isotropy_variant` column so variants can be compared side by side.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments" / "results"
DEFAULT_RERUNS_ROOT = REPO_ROOT / "experiments" / "results_reruns"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.analysis.select_global_pair_unsupervised import (
    _augment_texts,
    _build_global_pool,
    _build_shortlist_from_rule,
    _cap_task_texts,
    _compute_alignment,
    _compute_uniformity_loss,
    _model_slug,
    _normalize_rows,
    _resolve_two_stage_shortlist_rule,
    _select_rule,
    add_metric_ranks,
    add_temporal_stability,
    build_pair_entropy_tiebreak_table,
    load_pair_avg_main_table,
)
from scripts.analysis.select_taskwise_pairs_factorized import (
    TASK_TO_DATASET,
    SelectionError,
    _parse_revision_order_key,
    _revision_seed_value,
    compute_dispersion,
    load_sampled_task_texts,
)


SUPPORTED_SELECTION_RULES = [
    "dispersion_only",
    "dispersion_entropy_tiebreak",
    "alignment_uniformity",
    "alignment_uniformity_entropy_tiebreak",
]

DEFAULT_SELECTION_RULES = [
    "dispersion_only",
    "dispersion_entropy_tiebreak",
    "alignment_uniformity",
    "alignment_uniformity_entropy_tiebreak",
]

DEFAULT_ISOTROPY_VARIANTS = [
    "raw",
    "pc_remove_r1",
    "pc_remove_r2",
    "pc_remove_r4",
    "pc_remove_r8",
    "whiten_full",
    "whiten_dim256",
    "whiten_dim384",
    "flow",
]

EXTRA_GEOMETRY_COLUMNS = {
    "top_pc_dominance": np.nan,
    "effective_rank": np.nan,
    "participation_ratio": np.nan,
    "knn_aug_stability": np.nan,
    "checkpoint_neighbor_stability": np.nan,
    "checkpoint_neighbor_count": 0,
    "layer_neighbor_stability": np.nan,
    "layer_neighbor_count": 0,
}


@dataclass(frozen=True)
class IsotropyVariantSpec:
    name: str
    kind: str
    parameter: int | None = None


@dataclass
class _LinearTransformState:
    mean: np.ndarray
    components: np.ndarray | None
    scales: np.ndarray | None = None
    effective_dim: int = 0


@dataclass
class _FlowState:
    preprocessor: _LinearTransformState
    model: nn.Module
    device: torch.device
    input_dim: int
    best_nll: float
    best_epoch: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone isotropy-correction selector. Keeps previous selector scripts unchanged "
            "and compares raw, PC-removal, whitening, and shortlist-only affine-coupling flow variants."
        )
    )
    parser.add_argument("--model_family", type=str, default="Pythia")
    parser.add_argument("--model_size", type=str, default="410m")
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
    parser.add_argument("--pooling_method", type=str, default="mean")
    parser.add_argument("--pool_samples_per_task", type=int, default=32)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--dispersion_num_pairs", type=int, default=50000)
    parser.add_argument("--uniformity_num_pairs", type=int, default=50000)
    parser.add_argument("--uniformity_temperature", type=float, default=2.0)
    parser.add_argument("--alignment_quantile", type=float, default=0.30)
    parser.add_argument("--top_m_pairs", type=int, default=10)
    parser.add_argument("--stability_lambda", type=float, default=0.0)
    parser.add_argument(
        "--entropy_tiebreak_mode",
        type=str,
        default="minmax_sum",
        choices=["dataset_entropy", "dime", "infonce", "rank_sum", "zsum", "minmax_sum"],
    )
    parser.add_argument(
        "--selection_rules",
        type=str,
        default=",".join(DEFAULT_SELECTION_RULES),
        help="Comma-separated list from: " + ", ".join(SUPPORTED_SELECTION_RULES),
    )
    parser.add_argument(
        "--isotropy_variants",
        type=str,
        default=",".join(DEFAULT_ISOTROPY_VARIANTS),
        help=(
            "Comma-separated isotropy variants. Supported forms: raw, pc_remove_r{1,2,4,8,...}, "
            "whiten_full, whiten_dim{N}, flow, flow_dim{N}."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_sample_length", type=int, default=512)
    parser.add_argument("--baseline_revision", type=str, default="step143000")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument(
        "--task_names",
        type=str,
        default="",
        help="Optional comma-separated subset of task names to use in the global pool.",
    )
    parser.add_argument(
        "--cache_task_texts_path",
        type=str,
        default="",
        help="Optional JSON cache for sampled per-task texts.",
    )
    parser.add_argument(
        "--disable_post_transform_l2_normalization",
        action="store_true",
        help="Disable L2 renormalization after PC-removal / whitening / flow transforms.",
    )
    parser.add_argument(
        "--whitening_eps",
        type=float,
        default=1e-6,
        help="Numerical floor used when inverting singular values for whitening.",
    )
    parser.add_argument(
        "--flow_shortlist_rule",
        type=str,
        default="auto",
        choices=["auto", "alignment_uniformity", "dispersion_only"],
        help="Shortlist generator used to decide which pairs receive the heavier flow transform.",
    )
    parser.add_argument(
        "--flow_shortlist_size",
        type=int,
        default=20,
        help="How many raw shortlisted pairs receive the flow transform.",
    )
    parser.add_argument(
        "--flow_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used to train the flow model.",
    )
    parser.add_argument(
        "--flow_num_layers",
        type=int,
        default=4,
        help="Number of affine-coupling layers in the normalizing flow proxy used for shortlist-only flow isotropization.",
    )
    parser.add_argument(
        "--flow_hidden_dim",
        type=int,
        default=512,
        help="Hidden width of the coupling networks.",
    )
    parser.add_argument(
        "--flow_num_epochs",
        type=int,
        default=200,
        help="Training epochs for the flow model on each shortlisted pair.",
    )
    parser.add_argument(
        "--flow_batch_size",
        type=int,
        default=256,
        help="Mini-batch size for flow fitting. Use 0 or a negative value for full-batch.",
    )
    parser.add_argument(
        "--flow_learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for flow fitting.",
    )
    parser.add_argument(
        "--flow_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for flow fitting.",
    )
    parser.add_argument(
        "--flow_scale_clip",
        type=float,
        default=5.0,
        help="Tanh-clipped maximum absolute log-scale used in affine couplings.",
    )
    parser.add_argument(
        "--flow_grad_clip",
        type=float,
        default=5.0,
        help="Gradient clipping norm for flow fitting.",
    )
    parser.add_argument(
        "--flow_log_interval",
        type=int,
        default=0,
        help="If >0, print a per-pair flow loss line every N epochs.",
    )
    return parser.parse_args()


def _parse_isotropy_variants(spec: str) -> list[IsotropyVariantSpec]:
    variants: list[IsotropyVariantSpec] = []
    seen: set[str] = set()
    for token in [x.strip() for x in spec.split(",") if x.strip()]:
        if token == "raw":
            parsed = IsotropyVariantSpec(name=token, kind="raw")
        elif match := re.fullmatch(r"pc_remove_r(\d+)", token):
            parsed = IsotropyVariantSpec(name=token, kind="pc_remove", parameter=int(match.group(1)))
        elif token == "whiten_full":
            parsed = IsotropyVariantSpec(name=token, kind="whiten", parameter=None)
        elif match := re.fullmatch(r"whiten_dim(\d+)", token):
            parsed = IsotropyVariantSpec(name=token, kind="whiten", parameter=int(match.group(1)))
        elif token == "flow":
            parsed = IsotropyVariantSpec(name=token, kind="flow", parameter=None)
        elif match := re.fullmatch(r"flow_dim(\d+)", token):
            parsed = IsotropyVariantSpec(name=token, kind="flow", parameter=int(match.group(1)))
        else:
            raise SelectionError(f"Unsupported isotropy variant: {token}")
        if parsed.name in seen:
            raise SelectionError(f"Duplicate isotropy variant requested: {parsed.name}")
        seen.add(parsed.name)
        variants.append(parsed)

    if not variants:
        raise SelectionError("At least one isotropy variant must be requested.")

    if "raw" not in seen:
        variants.insert(0, IsotropyVariantSpec(name="raw", kind="raw"))
    return variants


def _maybe_l2_normalize(embeddings: np.ndarray, enabled: bool) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    return _normalize_rows(matrix) if enabled else matrix


def _fit_pc_removal_transform(embeddings: np.ndarray, num_components: int) -> _LinearTransformState:
    matrix = np.asarray(embeddings, dtype=np.float64)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    if centered.ndim != 2 or centered.shape[0] < 2:
        return _LinearTransformState(mean=mean.astype(np.float32), components=None, effective_dim=int(centered.shape[-1]))

    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise SelectionError(f"PC-removal SVD failed: {exc}") from exc

    keep = min(int(num_components), int(vt.shape[0]))
    components = vt[:keep].T.astype(np.float32) if keep > 0 else None
    return _LinearTransformState(
        mean=mean.astype(np.float32),
        components=components,
        effective_dim=int(centered.shape[1]),
    )


def _apply_pc_removal_transform(
    embeddings: np.ndarray,
    state: _LinearTransformState,
    l2_normalize: bool,
) -> np.ndarray:
    centered = np.asarray(embeddings, dtype=np.float32) - state.mean
    if state.components is not None and state.components.size > 0:
        centered = centered - (centered @ state.components) @ state.components.T
    return _maybe_l2_normalize(centered, enabled=l2_normalize)


def _fit_whitening_transform(
    embeddings: np.ndarray,
    target_dim: int | None,
    eps: float,
) -> _LinearTransformState:
    matrix = np.asarray(embeddings, dtype=np.float64)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    if centered.ndim != 2 or centered.shape[0] < 2:
        return _LinearTransformState(mean=mean.astype(np.float32), components=None, scales=None, effective_dim=0)

    try:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise SelectionError(f"Whitening SVD failed: {exc}") from exc

    valid = singular_values > float(eps)
    singular_values = singular_values[valid]
    vt = vt[valid]
    if singular_values.size == 0:
        return _LinearTransformState(mean=mean.astype(np.float32), components=None, scales=None, effective_dim=0)

    if target_dim is None:
        keep = int(singular_values.size)
    else:
        keep = min(int(target_dim), int(singular_values.size))
    components = vt[:keep].T.astype(np.float32)
    scales = (math.sqrt(max(1, centered.shape[0] - 1)) / np.clip(singular_values[:keep], eps, None)).astype(np.float32)
    return _LinearTransformState(
        mean=mean.astype(np.float32),
        components=components,
        scales=scales,
        effective_dim=int(keep),
    )


def _apply_whitening_transform(
    embeddings: np.ndarray,
    state: _LinearTransformState,
    l2_normalize: bool,
) -> np.ndarray:
    centered = np.asarray(embeddings, dtype=np.float32) - state.mean
    if state.components is None or state.scales is None or state.effective_dim == 0:
        return _maybe_l2_normalize(centered, enabled=l2_normalize)
    whitened = (centered @ state.components) * state.scales
    return _maybe_l2_normalize(whitened, enabled=l2_normalize)


def _fit_flow_preprocessor(embeddings: np.ndarray, target_dim: int | None) -> _LinearTransformState:
    matrix = np.asarray(embeddings, dtype=np.float64)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    if target_dim is None or target_dim <= 0 or centered.shape[1] <= target_dim:
        return _LinearTransformState(
            mean=mean.astype(np.float32),
            components=None,
            effective_dim=int(centered.shape[1]),
        )

    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise SelectionError(f"Flow PCA preprocessor SVD failed: {exc}") from exc

    keep = min(int(target_dim), int(vt.shape[0]), int(centered.shape[1]))
    components = vt[:keep].T.astype(np.float32)
    return _LinearTransformState(
        mean=mean.astype(np.float32),
        components=components,
        effective_dim=int(keep),
    )


def _apply_flow_preprocessor(embeddings: np.ndarray, state: _LinearTransformState) -> np.ndarray:
    centered = np.asarray(embeddings, dtype=np.float32) - state.mean
    if state.components is None:
        return centered
    return centered @ state.components


class _CouplingMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.net(x)
        log_scale, shift = output.chunk(2, dim=-1)
        return log_scale, shift


class _AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor, scale_clip: float) -> None:
        super().__init__()
        self.net = _CouplingMLP(dim, hidden_dim)
        self.register_buffer("mask", mask.view(1, dim))
        self.scale_clip = float(scale_clip)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask
        masked_x = x * mask
        log_scale, shift = self.net(masked_x)
        log_scale = torch.tanh(log_scale) * self.scale_clip
        inv_mask = 1.0 - mask
        log_scale = log_scale * inv_mask
        shift = shift * inv_mask
        y = masked_x + inv_mask * (x * torch.exp(log_scale) + shift)
        log_det = log_scale.sum(dim=-1)
        return y, log_det


class _Permutation(nn.Module):
    def __init__(self, permutation: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("permutation", permutation.clone())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[:, self.permutation], torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


class _NormalizingFlow(nn.Module):
    def __init__(self, dim: int, num_layers: int, hidden_dim: int, scale_clip: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        base_mask = torch.zeros(dim, dtype=torch.float32)
        base_mask[: dim // 2] = 1.0
        if base_mask.sum() == 0 or base_mask.sum() == dim:
            base_mask = torch.arange(dim, dtype=torch.float32) % 2
        for layer_idx in range(max(1, num_layers)):
            mask = base_mask if layer_idx % 2 == 0 else (1.0 - base_mask)
            layers.append(_AffineCoupling(dim=dim, hidden_dim=hidden_dim, mask=mask, scale_clip=scale_clip))
            permutation = torch.roll(torch.arange(dim), shifts=max(1, dim // 3))
            layers.append(_Permutation(permutation))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        y = x
        for layer in self.layers:
            y, delta = layer(y)
            log_det = log_det + delta
        return y, log_det


def _resolve_flow_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SelectionError("--flow_device=cuda was requested but CUDA is not available.")
    return torch.device(requested)


def _fit_flow_transform(
    embeddings: np.ndarray,
    pair_seed: int,
    variant: IsotropyVariantSpec,
    args: argparse.Namespace,
    log_prefix: str,
) -> _FlowState:
    preprocessor = _fit_flow_preprocessor(embeddings, target_dim=variant.parameter)
    prepared = _apply_flow_preprocessor(embeddings, preprocessor)
    if prepared.ndim != 2 or prepared.shape[0] < 2 or prepared.shape[1] < 2:
        raise SelectionError(
            f"Flow transform requires at least 2 samples and 2 dimensions, got {prepared.shape}"
        )

    device = _resolve_flow_device(args.flow_device)
    tensor = torch.from_numpy(prepared.astype(np.float32)).to(device)
    torch.manual_seed(int(pair_seed + 90_000_000))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(pair_seed + 90_000_000))

    model = _NormalizingFlow(
        dim=int(tensor.shape[1]),
        num_layers=int(args.flow_num_layers),
        hidden_dim=int(args.flow_hidden_dim),
        scale_clip=float(args.flow_scale_clip),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.flow_learning_rate),
        weight_decay=float(args.flow_weight_decay),
    )

    batch_size = int(args.flow_batch_size)
    if batch_size <= 0:
        batch_size = int(tensor.shape[0])

    best_state: dict[str, torch.Tensor] | None = None
    best_nll = float("inf")
    best_epoch = -1
    two_pi_log = math.log(2.0 * math.pi)

    for epoch_idx in range(int(args.flow_num_epochs)):
        model.train()
        permutation = torch.randperm(tensor.shape[0], device=device)
        epoch_losses: list[float] = []
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[permutation[start : start + batch_size]]
            z, log_det = model(batch)
            nll = 0.5 * torch.sum(z * z, dim=1) + 0.5 * float(z.shape[1]) * two_pi_log - log_det
            loss = nll.mean()
            if not torch.isfinite(loss):
                raise SelectionError(f"Flow loss became non-finite for {log_prefix} at epoch {epoch_idx + 1}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.flow_grad_clip))
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        epoch_nll = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        if epoch_nll < best_nll:
            best_nll = epoch_nll
            best_epoch = epoch_idx + 1
            best_state = copy.deepcopy(model.state_dict())
        if args.flow_log_interval > 0 and ((epoch_idx + 1) % int(args.flow_log_interval) == 0):
            print(
                f"[flow] {log_prefix} epoch={epoch_idx + 1}/{args.flow_num_epochs} "
                f"nll={epoch_nll:.6f} best={best_nll:.6f}"
            )

    if best_state is None:
        raise SelectionError(f"Flow fitting did not record a valid checkpoint for {log_prefix}.")

    model.load_state_dict(best_state)
    model.eval()
    return _FlowState(
        preprocessor=preprocessor,
        model=model,
        device=device,
        input_dim=int(prepared.shape[1]),
        best_nll=float(best_nll),
        best_epoch=int(best_epoch),
    )


def _apply_flow_transform(
    embeddings: np.ndarray,
    flow_state: _FlowState,
    l2_normalize: bool,
) -> np.ndarray:
    prepared = _apply_flow_preprocessor(embeddings, flow_state.preprocessor)
    tensor = torch.from_numpy(prepared.astype(np.float32)).to(flow_state.device)
    with torch.no_grad():
        transformed, _ = flow_state.model(tensor)
    return _maybe_l2_normalize(transformed.detach().cpu().numpy(), enabled=l2_normalize)


def _row_template(
    variant: IsotropyVariantSpec,
    revision: str,
    layer_idx: int,
    pair_seed: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "isotropy_variant": variant.name,
        "transform_kind": variant.kind,
        "transform_parameter": variant.parameter,
        "revision": str(revision),
        "layer_idx": int(layer_idx),
        "num_pool_texts": np.nan,
        "embedding_dim": np.nan,
        "dispersion": np.nan,
        "dispersion_pair_seed": int(pair_seed),
        "alignment": np.nan,
        "uniformity_loss": np.nan,
        "uniformity_score": np.nan,
        "uniformity_pair_seed": np.nan,
        "transform_status": "pending",
        "transform_error": "",
        "flow_train_nll": np.nan,
        "flow_best_epoch": np.nan,
        "flow_effective_input_dim": np.nan,
        "flow_shortlisted_pair": False,
    }
    row.update(EXTRA_GEOMETRY_COLUMNS)
    return row


def _populate_geometry_metrics(
    row: dict[str, Any],
    embeddings: np.ndarray | None,
    augmented_embeddings: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if embeddings is None:
        return row
    matrix = np.asarray(embeddings, dtype=np.float32)
    row["num_pool_texts"] = int(matrix.shape[0])
    row["embedding_dim"] = int(matrix.shape[1])
    row["dispersion"] = float(
        compute_dispersion(
            matrix,
            num_pairs=args.dispersion_num_pairs,
            seed=int(row["dispersion_pair_seed"]),
        )
    )
    if augmented_embeddings is not None:
        augmented = np.asarray(augmented_embeddings, dtype=np.float32)
        row["alignment"] = float(_compute_alignment(matrix, augmented))
        uniformity_seed = int(row["dispersion_pair_seed"]) + 1_000_000
        row["uniformity_pair_seed"] = int(uniformity_seed)
        uniformity_loss = _compute_uniformity_loss(
            matrix,
            num_pairs=args.uniformity_num_pairs,
            temperature=args.uniformity_temperature,
            seed=uniformity_seed,
        )
        row["uniformity_loss"] = float(uniformity_loss)
        row["uniformity_score"] = float(-uniformity_loss)
    row["transform_status"] = "ok"
    return row


def _apply_nonflow_variant(
    variant: IsotropyVariantSpec,
    embeddings: np.ndarray,
    augmented_embeddings: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray | None]:
    l2_normalize = not bool(args.disable_post_transform_l2_normalization)
    if variant.kind == "raw":
        transformed = np.asarray(embeddings, dtype=np.float32)
        transformed_aug = np.asarray(augmented_embeddings, dtype=np.float32) if augmented_embeddings is not None else None
        return transformed, transformed_aug
    if variant.kind == "pc_remove":
        state = _fit_pc_removal_transform(embeddings, num_components=int(variant.parameter or 0))
        transformed = _apply_pc_removal_transform(embeddings, state, l2_normalize=l2_normalize)
        transformed_aug = (
            _apply_pc_removal_transform(augmented_embeddings, state, l2_normalize=l2_normalize)
            if augmented_embeddings is not None
            else None
        )
        return transformed, transformed_aug
    if variant.kind == "whiten":
        state = _fit_whitening_transform(
            embeddings,
            target_dim=variant.parameter,
            eps=float(args.whitening_eps),
        )
        transformed = _apply_whitening_transform(embeddings, state, l2_normalize=l2_normalize)
        transformed_aug = (
            _apply_whitening_transform(augmented_embeddings, state, l2_normalize=l2_normalize)
            if augmented_embeddings is not None
            else None
        )
        return transformed, transformed_aug
    raise ValueError(f"Unsupported non-flow variant: {variant}")


def _compute_nonflow_geometry_rows(
    global_texts: list[str],
    revisions: list[str],
    candidate_layers: list[int],
    variants: list[IsotropyVariantSpec],
    args: argparse.Namespace,
    needs_alignment: bool,
) -> pd.DataFrame:
    nonflow_variants = [variant for variant in variants if variant.kind != "flow"]
    rows: list[dict[str, Any]] = []
    augmented_texts = _augment_texts(global_texts, seed=args.sample_seed + 17) if needs_alignment else None

    from experiments.utils.model_definitions.text_automodel_wrapper import (
        TextLayerwiseAutoModelWrapper,
        TextModelSpecifications,
    )

    max_layer = max(candidate_layers)
    for revision in revisions:
        revision_seed_value = _revision_seed_value(revision)
        model_specs = TextModelSpecifications(args.model_family, args.model_size, revision)
        model = TextLayerwiseAutoModelWrapper(
            model_specs,
            device_map=args.device_map,
            evaluation_layer_idx=max_layer,
        )
        model.print_loading_message()
        _, layerwise_embeddings = model.encode(
            global_texts,
            batch_size=args.batch_size,
            num_workers=0,
            pooling_method=args.pooling_method,
            max_sample_length=args.max_sample_length,
            verbose=False,
            return_layerwise_encodings=True,
        )
        if needs_alignment:
            _, layerwise_aug_embeddings = model.encode(
                augmented_texts,
                batch_size=args.batch_size,
                num_workers=0,
                pooling_method=args.pooling_method,
                max_sample_length=args.max_sample_length,
                verbose=False,
                return_layerwise_encodings=True,
            )
        else:
            layerwise_aug_embeddings = None

        for layer_idx in candidate_layers:
            if layer_idx >= layerwise_embeddings.shape[0]:
                continue
            base_embeddings = np.asarray(layerwise_embeddings[layer_idx], dtype=np.float32)
            augmented_embeddings = (
                np.asarray(layerwise_aug_embeddings[layer_idx], dtype=np.float32)
                if layerwise_aug_embeddings is not None
                else None
            )
            pair_seed = int(args.sample_seed + revision_seed_value + 10_000 * layer_idx)

            for variant in nonflow_variants:
                row = _row_template(variant, revision, layer_idx, pair_seed)
                try:
                    transformed, transformed_aug = _apply_nonflow_variant(
                        variant=variant,
                        embeddings=base_embeddings,
                        augmented_embeddings=augmented_embeddings,
                        args=args,
                    )
                    row = _populate_geometry_metrics(
                        row=row,
                        embeddings=transformed,
                        augmented_embeddings=transformed_aug,
                        args=args,
                    )
                except Exception as exc:  # pragma: no cover - defensive per-pair guard
                    row["transform_status"] = "failed"
                    row["transform_error"] = str(exc)
                rows.append(row)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise SelectionError("No non-flow geometry rows were produced.")
    return pd.DataFrame(rows)


def _build_pair_metrics_for_variant(
    geometry_df: pd.DataFrame,
    pair_eval_df: pd.DataFrame,
    usable_revisions: list[str],
    args: argparse.Namespace,
    entropy_pair_scores: pd.DataFrame,
) -> pd.DataFrame:
    out = geometry_df[geometry_df["revision"].isin(usable_revisions)].copy()
    all_pairs = pair_eval_df[pair_eval_df["revision"].isin(usable_revisions)][["revision", "layer_idx"]].drop_duplicates()
    out = all_pairs.merge(out, on=["revision", "layer_idx"], how="left")
    out["isotropy_variant"] = out["isotropy_variant"].fillna(str(geometry_df["isotropy_variant"].iloc[0]))
    out["transform_kind"] = out["transform_kind"].fillna(str(geometry_df["transform_kind"].iloc[0]))
    if "transform_parameter" not in out:
        out["transform_parameter"] = np.nan
    for col_name, default_value in EXTRA_GEOMETRY_COLUMNS.items():
        if col_name not in out:
            out[col_name] = default_value
    if "transform_status" not in out:
        out["transform_status"] = "missing"
    if "transform_error" not in out:
        out["transform_error"] = ""
    if "flow_train_nll" not in out:
        out["flow_train_nll"] = np.nan
    if "flow_best_epoch" not in out:
        out["flow_best_epoch"] = np.nan
    if "flow_effective_input_dim" not in out:
        out["flow_effective_input_dim"] = np.nan
    if "flow_shortlisted_pair" not in out:
        out["flow_shortlisted_pair"] = False

    out = out.merge(pair_eval_df, on=["revision", "layer_idx"], how="left")
    out = add_temporal_stability(out, usable_revisions, args.stability_lambda)

    if not entropy_pair_scores.empty:
        out = out.merge(entropy_pair_scores, on=["revision", "layer_idx"], how="left")
    else:
        out["entropy_tiebreak_score"] = np.nan
        out["entropy_tiebreak_score_median"] = np.nan
        out["entropy_tiebreak_support_tasks"] = np.nan
        out["dataset_entropy_mean"] = np.nan
        out["infonce_for_min_mean"] = np.nan
        out["dime_mean"] = np.nan

    out = add_metric_ranks(out)
    out = out.sort_values(["revision_step", "layer_idx"], ascending=[True, True]).reset_index(drop=True)
    return out


def _compute_global_baselines(pair_df: pd.DataFrame, baseline_revision: str) -> dict[str, Any]:
    eligible = pair_df.dropna(subset=["avg_main_score"]).copy()
    if eligible.empty:
        raise SelectionError("No evaluable pairs are available after merging geometry with average-main scores.")

    oracle = eligible.sort_values(
        ["avg_main_score", "revision_step", "layer_idx"],
        ascending=[False, True, True],
    ).iloc[0]
    baseline_rows = eligible[eligible["revision"] == baseline_revision].copy()
    if baseline_rows.empty:
        raise SelectionError(f"No evaluable rows were found for baseline_revision={baseline_revision}")
    baseline_best = baseline_rows.sort_values(
        ["avg_main_score", "layer_idx"],
        ascending=[False, True],
    ).iloc[0]
    baseline_last = baseline_rows[baseline_rows["layer_idx"] == baseline_rows["layer_idx"].max()].iloc[0]
    return {
        "oracle_best_pair": {
            "revision": str(oracle["revision"]),
            "layer": int(oracle["layer_idx"]),
            "avg_main_score": float(oracle["avg_main_score"]),
            "avg_main_rank_desc": float(oracle["avg_main_rank_desc"]) if not pd.isna(oracle["avg_main_rank_desc"]) else None,
        },
        "baseline_revision_best_layer": {
            "revision": str(baseline_best["revision"]),
            "layer": int(baseline_best["layer_idx"]),
            "avg_main_score": float(baseline_best["avg_main_score"]),
        },
        "baseline_revision_last_layer": {
            "revision": str(baseline_last["revision"]),
            "layer": int(baseline_last["layer_idx"]),
            "avg_main_score": float(baseline_last["avg_main_score"]),
        },
    }


def _select_results_for_variant(
    pair_df: pd.DataFrame,
    variant_name: str,
    selection_rules: list[str],
    args: argparse.Namespace,
    baselines: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, Any]] = []
    shortlist_rows: list[pd.DataFrame] = []

    for rule in selection_rules:
        base_result: dict[str, Any] = {
            "isotropy_variant": variant_name,
            "rule": rule,
            "selection_error": None,
        }
        try:
            selected, meta, shortlist_df = _select_rule(
                pair_df,
                rule=rule,
                model_size=args.model_size,
                top_m_pairs=args.top_m_pairs,
                reranker_shortlist_size=max(1, args.flow_shortlist_size),
                alignment_quantile=args.alignment_quantile,
                reranker_features=[],
                reranker_fusion="rrf",
                reranker_rrf_k=60,
            )
            result = {
                "selected_revision": str(selected["revision"]),
                "selected_layer": int(selected["layer_idx"]),
                "selected_avg_main_score": float(selected["avg_main_score"]),
                "selected_num_tasks_eval": int(selected["num_tasks_eval"]),
                "selected_dispersion": float(selected["dispersion"]) if not pd.isna(selected["dispersion"]) else None,
                "selected_dispersion_stable": float(selected["dispersion_stable"]) if not pd.isna(selected["dispersion_stable"]) else None,
                "selected_dispersion_delta_prev": float(selected["dispersion_delta_prev"]) if not pd.isna(selected["dispersion_delta_prev"]) else None,
                "selected_alignment": float(selected["alignment"]) if not pd.isna(selected["alignment"]) else None,
                "selected_uniformity_score": float(selected["uniformity_score"]) if not pd.isna(selected["uniformity_score"]) else None,
                "selected_entropy_tiebreak_score": float(selected["entropy_tiebreak_score"]) if not pd.isna(selected["entropy_tiebreak_score"]) else None,
                "selected_avg_main_rank_desc": float(selected["avg_main_rank_desc"]) if not pd.isna(selected["avg_main_rank_desc"]) else None,
                "selected_dispersion_rank_desc": float(selected["dispersion_rank_desc"]) if not pd.isna(selected["dispersion_rank_desc"]) else None,
                "selected_dispersion_stable_rank_desc": float(selected["dispersion_stable_rank_desc"]) if not pd.isna(selected["dispersion_stable_rank_desc"]) else None,
                "selected_alignment_rank_asc": float(selected["alignment_rank_asc"]) if not pd.isna(selected["alignment_rank_asc"]) else None,
                "selected_uniformity_rank_desc": float(selected["uniformity_rank_desc"]) if not pd.isna(selected["uniformity_rank_desc"]) else None,
                "selected_entropy_tiebreak_rank_asc": float(selected["entropy_tiebreak_rank_asc"]) if not pd.isna(selected["entropy_tiebreak_rank_asc"]) else None,
                "selected_transform_kind": str(selected.get("transform_kind", "")),
                "selected_transform_parameter": (
                    int(selected["transform_parameter"])
                    if "transform_parameter" in selected and not pd.isna(selected["transform_parameter"])
                    else None
                ),
                "selected_transform_status": str(selected.get("transform_status", "")),
                "selected_transform_error": str(selected.get("transform_error", "")),
                "selected_embedding_dim": int(selected["embedding_dim"]) if not pd.isna(selected["embedding_dim"]) else None,
                "selected_flow_train_nll": float(selected["flow_train_nll"]) if not pd.isna(selected["flow_train_nll"]) else None,
                "selected_flow_best_epoch": int(selected["flow_best_epoch"]) if not pd.isna(selected["flow_best_epoch"]) else None,
                "selected_flow_effective_input_dim": int(selected["flow_effective_input_dim"]) if not pd.isna(selected["flow_effective_input_dim"]) else None,
                "selected_flow_shortlisted_pair": bool(selected["flow_shortlisted_pair"]) if "flow_shortlisted_pair" in selected else False,
                "delta_vs_oracle": float(selected["avg_main_score"] - baselines["oracle_best_pair"]["avg_main_score"]),
                "delta_vs_baseline_best": float(
                    selected["avg_main_score"] - baselines["baseline_revision_best_layer"]["avg_main_score"]
                ),
                "delta_vs_baseline_last": float(
                    selected["avg_main_score"] - baselines["baseline_revision_last_layer"]["avg_main_score"]
                ),
            }
            result.update(meta)
            base_result.update(result)
            if not shortlist_df.empty:
                shortlist_export = shortlist_df.copy()
                shortlist_export["isotropy_variant"] = variant_name
                shortlist_export["selected_by_rule"] = (
                    (shortlist_export["revision"].astype(str) == str(selected["revision"]))
                    & (shortlist_export["layer_idx"].astype(int) == int(selected["layer_idx"]))
                )
                shortlist_rows.append(shortlist_export)
        except Exception as exc:
            base_result["selection_error"] = str(exc)
        results.append(base_result)

    shortlist_payload = pd.concat(shortlist_rows, ignore_index=True) if shortlist_rows else pd.DataFrame()
    return pd.DataFrame(results), shortlist_payload


def _write_summary_markdown(
    output_dir: str,
    args: argparse.Namespace,
    usable_revisions: list[str],
    global_pool_df: pd.DataFrame,
    selected_results_df: pd.DataFrame,
    baselines: dict[str, Any],
    variant_specs: list[IsotropyVariantSpec],
    flow_shortlist_rule: str,
    flow_shortlist_df: pd.DataFrame,
) -> None:
    variant_names = ", ".join(spec.name for spec in variant_specs)
    lines = [
        "# Global Unsupervised Best-Pair Selector with Post-hoc Isotropy",
        "",
        "## Configuration",
        f"- model: `{args.model_family} {args.model_size}`",
        f"- usable revisions: `{', '.join(usable_revisions)}`",
        f"- candidate layers: `{args.layer_start}..{args.layer_end}`",
        f"- isotropy variants: `{variant_names}`",
        f"- per-task texts in global pool: `{args.pool_samples_per_task}`",
        f"- total global pool size: `{len(global_pool_df)}`",
        f"- selection rules: `{args.selection_rules}`",
        f"- alignment quantile: `{args.alignment_quantile}`",
        f"- post-transform L2 normalization: `{not args.disable_post_transform_l2_normalization}`",
        f"- flow shortlist rule: `{flow_shortlist_rule}`",
        f"- flow shortlist size: `{args.flow_shortlist_size}`",
        f"- flow epochs: `{args.flow_num_epochs}`",
        f"- flow hidden dim: `{args.flow_hidden_dim}`",
        f"- flow layers: `{args.flow_num_layers}`",
        "",
        "## Baselines",
        f"- oracle best static pair: `{baselines['oracle_best_pair']['revision']}/layer_{baselines['oracle_best_pair']['layer']}` -> `{baselines['oracle_best_pair']['avg_main_score']:.10f}`",
        f"- `{args.baseline_revision}` best layer: `layer_{baselines['baseline_revision_best_layer']['layer']}` -> `{baselines['baseline_revision_best_layer']['avg_main_score']:.10f}`",
        f"- `{args.baseline_revision}` last layer: `layer_{baselines['baseline_revision_last_layer']['layer']}` -> `{baselines['baseline_revision_last_layer']['avg_main_score']:.10f}`",
        "",
        "## Flow Coverage",
        f"- shortlisted raw pairs for flow fitting: `{len(flow_shortlist_df)}`",
        "",
        "## Results",
    ]

    ordered = selected_results_df.copy()
    ordered["sort_key"] = ordered["selected_avg_main_score"].fillna(-np.inf)
    ordered = ordered.sort_values(["sort_key", "isotropy_variant", "rule"], ascending=[False, True, True])
    for _, row in ordered.iterrows():
        if row.get("selection_error"):
            lines.append(
                f"- `{row['isotropy_variant']}` / `{row['rule']}`: failed with `{row['selection_error']}`"
            )
            continue
        pair_label = f"{row['selected_revision']}/layer_{int(row['selected_layer'])}"
        lines.append(
            f"- `{row['isotropy_variant']}` / `{row['rule']}`: `{pair_label}` -> `{row['selected_avg_main_score']:.10f}` "
            f"(vs `{args.baseline_revision}` best: `{row['delta_vs_baseline_best']:+.10f}`, "
            f"vs `{args.baseline_revision}` last: `{row['delta_vs_baseline_last']:+.10f}`, "
            f"vs oracle: `{row['delta_vs_oracle']:+.10f}`)"
        )

    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _compute_flow_geometry_rows(
    global_texts: list[str],
    revisions: list[str],
    candidate_layers: list[int],
    flow_variant_specs: list[IsotropyVariantSpec],
    flow_candidate_pairs: dict[str, set[int]],
    args: argparse.Namespace,
    needs_alignment: bool,
) -> pd.DataFrame:
    if not flow_variant_specs:
        return pd.DataFrame()
    if not flow_candidate_pairs:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    augmented_texts = _augment_texts(global_texts, seed=args.sample_seed + 17) if needs_alignment else None

    from experiments.utils.model_definitions.text_automodel_wrapper import (
        TextLayerwiseAutoModelWrapper,
        TextModelSpecifications,
    )

    max_layer = max(candidate_layers)
    for revision in revisions:
        wanted_layers = sorted(flow_candidate_pairs.get(revision, set()))
        if not wanted_layers:
            continue

        revision_seed_value = _revision_seed_value(revision)
        model_specs = TextModelSpecifications(args.model_family, args.model_size, revision)
        model = TextLayerwiseAutoModelWrapper(
            model_specs,
            device_map=args.device_map,
            evaluation_layer_idx=max_layer,
        )
        model.print_loading_message()
        _, layerwise_embeddings = model.encode(
            global_texts,
            batch_size=args.batch_size,
            num_workers=0,
            pooling_method=args.pooling_method,
            max_sample_length=args.max_sample_length,
            verbose=False,
            return_layerwise_encodings=True,
        )
        if needs_alignment:
            _, layerwise_aug_embeddings = model.encode(
                augmented_texts,
                batch_size=args.batch_size,
                num_workers=0,
                pooling_method=args.pooling_method,
                max_sample_length=args.max_sample_length,
                verbose=False,
                return_layerwise_encodings=True,
            )
        else:
            layerwise_aug_embeddings = None

        for layer_idx in wanted_layers:
            if layer_idx >= layerwise_embeddings.shape[0]:
                continue
            base_embeddings = np.asarray(layerwise_embeddings[layer_idx], dtype=np.float32)
            augmented_embeddings = (
                np.asarray(layerwise_aug_embeddings[layer_idx], dtype=np.float32)
                if layerwise_aug_embeddings is not None
                else None
            )
            pair_seed = int(args.sample_seed + revision_seed_value + 10_000 * layer_idx)
            for variant in flow_variant_specs:
                row = _row_template(variant, revision, layer_idx, pair_seed)
                row["flow_shortlisted_pair"] = True
                log_prefix = f"{variant.name}:{revision}/layer_{layer_idx}"
                try:
                    flow_state = _fit_flow_transform(
                        embeddings=base_embeddings,
                        pair_seed=pair_seed,
                        variant=variant,
                        args=args,
                        log_prefix=log_prefix,
                    )
                    transformed = _apply_flow_transform(
                        base_embeddings,
                        flow_state=flow_state,
                        l2_normalize=not bool(args.disable_post_transform_l2_normalization),
                    )
                    transformed_aug = (
                        _apply_flow_transform(
                            augmented_embeddings,
                            flow_state=flow_state,
                            l2_normalize=not bool(args.disable_post_transform_l2_normalization),
                        )
                        if augmented_embeddings is not None
                        else None
                    )
                    row["flow_train_nll"] = float(flow_state.best_nll)
                    row["flow_best_epoch"] = int(flow_state.best_epoch)
                    row["flow_effective_input_dim"] = int(flow_state.input_dim)
                    row = _populate_geometry_metrics(row, transformed, transformed_aug, args)
                except Exception as exc:  # pragma: no cover - defensive per-pair guard
                    row["transform_status"] = "failed"
                    row["transform_error"] = str(exc)
                rows.append(row)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    revisions = [x.strip() for x in args.expected_revisions.split(",") if x.strip()]
    candidate_layers = list(range(args.layer_start, args.layer_end + 1))
    selection_rules = [x.strip() for x in args.selection_rules.split(",") if x.strip()]
    invalid_rules = sorted(set(selection_rules) - set(SUPPORTED_SELECTION_RULES))
    if invalid_rules:
        raise SelectionError(f"Unsupported selection rules requested: {invalid_rules}")
    if args.flow_shortlist_size < 1:
        raise SelectionError("--flow_shortlist_size must be >= 1")
    if args.top_m_pairs < 1:
        raise SelectionError("--top_m_pairs must be >= 1")

    variant_specs = _parse_isotropy_variants(args.isotropy_variants)
    flow_variant_specs = [variant for variant in variant_specs if variant.kind == "flow"]

    task_names = [x.strip() for x in args.task_names.split(",") if x.strip()]
    task_to_dataset = TASK_TO_DATASET.copy()
    if task_names:
        missing = sorted(set(task_names) - set(task_to_dataset))
        if missing:
            raise SelectionError(f"Unknown task_names requested: {missing}")
        task_to_dataset = {key: task_to_dataset[key] for key in task_names}

    output_dir = os.path.join(
        args.output_root,
        f"{_model_slug(args.model_family, args.model_size)}_global_pair_isotropy_variants_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)
    variant_dir = os.path.join(output_dir, "variant_pair_metrics")
    os.makedirs(variant_dir, exist_ok=True)

    cache_task_texts_path = args.cache_task_texts_path.strip() or os.path.join(output_dir, "sampled_task_texts.json")
    task_texts = load_sampled_task_texts(
        task_to_dataset,
        args.pool_samples_per_task,
        cache_task_texts_path,
        args.sample_seed,
    )
    task_texts = {task: texts for task, texts in task_texts.items() if task in task_to_dataset}
    task_texts = _cap_task_texts(task_texts, args.pool_samples_per_task, args.sample_seed)
    usable_task_texts = {task: texts for task, texts in task_texts.items() if texts}
    if not usable_task_texts:
        raise SelectionError("No task texts were available for the global unlabeled pool.")

    global_texts, global_pool_df = _build_global_pool(usable_task_texts)
    global_pool_df.to_csv(os.path.join(output_dir, "global_pool_texts.csv"), index=False)
    with open(os.path.join(output_dir, "global_pool_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "pool_samples_per_task": int(args.pool_samples_per_task),
                "sample_seed": int(args.sample_seed),
                "num_tasks": int(len(usable_task_texts)),
                "num_texts": int(len(global_texts)),
                "task_counts": {key: len(values) for key, values in usable_task_texts.items()},
                "cache_task_texts_path": cache_task_texts_path,
            },
            handle,
            indent=2,
        )

    needs_alignment = any("alignment_uniformity" in rule for rule in selection_rules)
    if args.flow_shortlist_rule == "alignment_uniformity":
        needs_alignment = True
    if args.flow_shortlist_rule == "auto":
        resolved_auto_shortlist = _resolve_two_stage_shortlist_rule("two_stage", args.model_size)
        if resolved_auto_shortlist == "alignment_uniformity":
            needs_alignment = True
    nonflow_geometry_df = _compute_nonflow_geometry_rows(
        global_texts=global_texts,
        revisions=revisions,
        candidate_layers=candidate_layers,
        variants=variant_specs,
        args=args,
        needs_alignment=needs_alignment,
    )

    pair_eval_df = load_pair_avg_main_table(args.main_runs_root, revisions, candidate_layers)
    raw_geometry_df = nonflow_geometry_df[nonflow_geometry_df["isotropy_variant"] == "raw"].copy()
    usable_revisions = sorted(
        set(raw_geometry_df["revision"]).intersection(pair_eval_df["revision"]),
        key=_parse_revision_order_key,
    )
    if not usable_revisions:
        raise SelectionError("No overlap between geometry results and evaluable average-main-score revisions.")

    entropy_pair_export, task_entropy_export, entropy_support_tasks = build_pair_entropy_tiebreak_table(
        args,
        usable_revisions,
        {task: task_to_dataset[task] for task in usable_task_texts},
        candidate_layers,
    )

    variant_metrics_map: dict[str, pd.DataFrame] = {}
    for variant_name, variant_geometry_df in nonflow_geometry_df.groupby("isotropy_variant", sort=False):
        pair_metrics = _build_pair_metrics_for_variant(
            geometry_df=variant_geometry_df,
            pair_eval_df=pair_eval_df,
            usable_revisions=usable_revisions,
            args=args,
            entropy_pair_scores=entropy_pair_export,
        )
        variant_metrics_map[str(variant_name)] = pair_metrics

    if not entropy_pair_export.empty:
        entropy_pair_export.to_csv(os.path.join(output_dir, "pair_entropy_tiebreak_scores.csv"), index=False)
    if not task_entropy_export.empty:
        task_entropy_export.to_csv(os.path.join(output_dir, "task_layer_entropy_scores.csv"), index=False)

    resolved_flow_shortlist_rule = (
        _resolve_two_stage_shortlist_rule("two_stage", args.model_size)
        if args.flow_shortlist_rule == "auto"
        else args.flow_shortlist_rule
    )

    flow_shortlist_df = pd.DataFrame()
    flow_candidate_pairs: dict[str, set[int]] = {}
    if flow_variant_specs:
        raw_pair_metrics = variant_metrics_map["raw"]
        flow_shortlist_df, flow_shortlist_meta = _build_shortlist_from_rule(
            raw_pair_metrics,
            shortlist_rule=resolved_flow_shortlist_rule,
            shortlist_size=args.flow_shortlist_size,
            alignment_quantile=args.alignment_quantile,
        )
        _ = flow_shortlist_meta
        flow_shortlist_df = flow_shortlist_df.copy()
        flow_shortlist_df.to_csv(os.path.join(output_dir, "flow_shortlist_from_raw.csv"), index=False)
        for row in flow_shortlist_df.itertuples(index=False):
            flow_candidate_pairs.setdefault(str(row.revision), set()).add(int(row.layer_idx))

        flow_geometry_subset_df = _compute_flow_geometry_rows(
            global_texts=global_texts,
            revisions=usable_revisions,
            candidate_layers=candidate_layers,
            flow_variant_specs=flow_variant_specs,
            flow_candidate_pairs=flow_candidate_pairs,
            args=args,
            needs_alignment=needs_alignment,
        )
        all_pairs_index = (
            pair_eval_df[pair_eval_df["revision"].isin(usable_revisions)][["revision", "layer_idx"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        for variant in flow_variant_specs:
            subset = flow_geometry_subset_df[flow_geometry_subset_df["isotropy_variant"] == variant.name].copy()
            if subset.empty:
                subset = all_pairs_index.copy()
                subset["isotropy_variant"] = variant.name
                subset["transform_kind"] = variant.kind
                subset["transform_parameter"] = variant.parameter
                subset["transform_status"] = "not_shortlisted"
                subset["transform_error"] = ""
                subset["flow_shortlisted_pair"] = False
                for col_name, default_value in EXTRA_GEOMETRY_COLUMNS.items():
                    subset[col_name] = default_value
                subset["dispersion"] = np.nan
                subset["dispersion_pair_seed"] = np.nan
                subset["alignment"] = np.nan
                subset["uniformity_loss"] = np.nan
                subset["uniformity_score"] = np.nan
                subset["uniformity_pair_seed"] = np.nan
                subset["num_pool_texts"] = np.nan
                subset["embedding_dim"] = np.nan
                subset["flow_train_nll"] = np.nan
                subset["flow_best_epoch"] = np.nan
                subset["flow_effective_input_dim"] = np.nan
            pair_metrics = _build_pair_metrics_for_variant(
                geometry_df=subset,
                pair_eval_df=pair_eval_df,
                usable_revisions=usable_revisions,
                args=args,
                entropy_pair_scores=entropy_pair_export,
            )
            flow_shortlisted_mask = pair_metrics.set_index(["revision", "layer_idx"]).index.isin(
                flow_shortlist_df.set_index(["revision", "layer_idx"]).index
            )
            pair_metrics["flow_shortlisted_pair"] = flow_shortlisted_mask
            pair_metrics.loc[
                (~pair_metrics["flow_shortlisted_pair"]) & pair_metrics["transform_status"].isna(),
                "transform_status",
            ] = "not_shortlisted"
            variant_metrics_map[variant.name] = pair_metrics

    baselines = _compute_global_baselines(variant_metrics_map["raw"], args.baseline_revision)

    variant_results_frames: list[pd.DataFrame] = []
    variant_shortlist_frames: list[pd.DataFrame] = []
    for variant_name, pair_metrics in variant_metrics_map.items():
        pair_metrics.to_csv(os.path.join(variant_dir, f"{variant_name}.csv"), index=False)
        selected_results_df, shortlist_df = _select_results_for_variant(
            pair_df=pair_metrics,
            variant_name=variant_name,
            selection_rules=selection_rules,
            args=args,
            baselines=baselines,
        )
        variant_results_frames.append(selected_results_df)
        if not shortlist_df.empty:
            variant_shortlist_frames.append(shortlist_df)

    pair_metrics_by_variant_df = pd.concat(
        [variant_metrics_map[name] for name in [spec.name for spec in variant_specs] if name in variant_metrics_map],
        ignore_index=True,
    )
    selected_rule_pairs_df = pd.concat(variant_results_frames, ignore_index=True)
    shortlisted_candidates_df = (
        pd.concat(variant_shortlist_frames, ignore_index=True)
        if variant_shortlist_frames
        else pd.DataFrame()
    )

    pair_metrics_by_variant_df.to_csv(os.path.join(output_dir, "pair_metrics_by_variant.csv"), index=False)
    selected_rule_pairs_df.to_csv(os.path.join(output_dir, "selected_rule_pairs_by_variant.csv"), index=False)
    if not shortlisted_candidates_df.empty:
        shortlisted_candidates_df.to_csv(os.path.join(output_dir, "shortlisted_candidates_by_variant.csv"), index=False)

    summary_payload = {
        "config": {
            "model_family": args.model_family,
            "model_size": args.model_size,
            "requested_revisions": revisions,
            "usable_revisions": usable_revisions,
            "candidate_layers": candidate_layers,
            "pooling_method": args.pooling_method,
            "pool_samples_per_task": args.pool_samples_per_task,
            "sample_seed": args.sample_seed,
            "dispersion_num_pairs": args.dispersion_num_pairs,
            "uniformity_num_pairs": args.uniformity_num_pairs,
            "uniformity_temperature": args.uniformity_temperature,
            "alignment_quantile": args.alignment_quantile,
            "top_m_pairs": args.top_m_pairs,
            "stability_lambda": args.stability_lambda,
            "entropy_tiebreak_mode": args.entropy_tiebreak_mode,
            "selection_rules": selection_rules,
            "isotropy_variants": [spec.name for spec in variant_specs],
            "baseline_revision": args.baseline_revision,
            "disable_post_transform_l2_normalization": bool(args.disable_post_transform_l2_normalization),
            "whitening_eps": args.whitening_eps,
            "flow_shortlist_rule_requested": args.flow_shortlist_rule,
            "flow_shortlist_rule_resolved": resolved_flow_shortlist_rule,
            "flow_shortlist_size": args.flow_shortlist_size,
            "flow_device": args.flow_device,
            "flow_num_layers": args.flow_num_layers,
            "flow_hidden_dim": args.flow_hidden_dim,
            "flow_num_epochs": args.flow_num_epochs,
            "flow_batch_size": args.flow_batch_size,
            "flow_learning_rate": args.flow_learning_rate,
            "flow_weight_decay": args.flow_weight_decay,
            "flow_scale_clip": args.flow_scale_clip,
            "flow_grad_clip": args.flow_grad_clip,
        },
        "global_pool": {
            "num_tasks": int(len(usable_task_texts)),
            "num_texts": int(len(global_texts)),
            "task_counts": {key: len(values) for key, values in usable_task_texts.items()},
        },
        "baselines": baselines,
        "entropy_tiebreak_tasks_used": entropy_support_tasks,
        "num_pair_rows_by_variant": {
            name: int(len(df)) for name, df in variant_metrics_map.items()
        },
        "num_evaluable_rows_by_variant": {
            name: int(df["avg_main_score"].notna().sum()) for name, df in variant_metrics_map.items()
        },
        "flow_shortlist_pairs": (
            flow_shortlist_df[["revision", "layer_idx"]].to_dict(orient="records")
            if not flow_shortlist_df.empty
            else []
        ),
        "variant_rule_results": selected_rule_pairs_df.to_dict(orient="records"),
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    _write_summary_markdown(
        output_dir=output_dir,
        args=args,
        usable_revisions=usable_revisions,
        global_pool_df=global_pool_df,
        selected_results_df=selected_rule_pairs_df,
        baselines=baselines,
        variant_specs=variant_specs,
        flow_shortlist_rule=resolved_flow_shortlist_rule,
        flow_shortlist_df=flow_shortlist_df,
    )

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
