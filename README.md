# Temporal Depth Representations

Pythia-focused research workspace for checkpoint and layer evaluation, entropy-style metric generation, and unsupervised checkpoint-layer selection.

## Layout

- `MTEB-Harness.py`: task and metric runner
- `experiments/utils/`: text-model, metric, and result-loading utilities
- `scripts/analysis/`: analysis entrypoints organized by purpose with `select_*`, `compare_*`, `ablate_*`, `build_*`, and `train_*` prefixes
- `scripts/slurm_scripts/`: workflow automation scripts organized as `submit_*` for single Slurm jobs, `launch_*` for multi-job launchers, and `prepare_*` for local cache setup

## Intended Workflow

1. Run layerwise MTEB evaluations across selected checkpoints.
2. Run entropy-style metrics across the same checkpoints.
3. Build average main-score tables.
4. Run factorized or global unsupervised selector analyses.
5. Compare PASS, isotropy, and ablation variants as needed.

## Naming Convention

- Analysis scripts are named for the experiment action first, then the study target.
- Slurm job files are named for the model, experiment family, and resource profile.
- Multi-submit shell wrappers use `launch_*` so they are distinct from single-job `submit_*` entrypoints.

## Notes

- Generated outputs are expected under `experiments/results/` and `experiments/results_reruns/`.
- Local environments, caches, logs, and generated results should remain untracked.

## Reference Input Layout

- `experiments/results/Pythia/410m/<revision>/metrics/mteb/` holds copied entropy-style metric inputs for PASS and related selectors.
- `experiments/results_reruns/<revision>/` is the curated flattened view of layerwise downstream evaluation outputs for each checkpoint.
- `experiments/results_reruns/task_text_caches/` stores reusable sampled unlabeled text pools so selector runs can stay reproducible across ablations.

This keeps prerequisite inputs easy to find while leaving timestamped reruns and new ablation outputs free to accumulate separately under `experiments/results_reruns/`.

## PASS Selector

- `scripts/analysis/select_global_pair_unsupervised.py` is the canonical global selector implementation and already writes the PASS-specific fields into `pair_metrics.csv`.
- `scripts/analysis/select_global_pair_unsupervised_pass.py` is the dedicated PASS entrypoint. It forwards to the canonical selector while defaulting `--selection_rules pass_metric`.
- `scripts/slurm_scripts/submit_pythia410m_pass_global_pair_selector_h100_2h.sbatch` and `scripts/slurm_scripts/submit_pythia70m_pass_global_pair_selector_h100_2h.sbatch` remain the repo-native Slurm entrypoints for the heavy PASS runs.

## PASS Ablations

These offline analyses consume the `pair_metrics.csv` and `summary.json` produced by a completed PASS selector run. They resolve the latest PASS run automatically from the existing `slurm_logs/latest_*pass_global_pair_selector_root.txt` marker files, or you can pass `--run_dir` or `--pair_metrics_csv` explicitly.

- `scripts/analysis/pass_ablation_common.py`: shared utilities for PASS run discovery, pair-metric validation, normalization, and evaluation tables
- `scripts/analysis/ablate_pass_components.py`: necessity and responsibility ablation across PASS components
- `scripts/analysis/ablate_pass_normalization.py`: normalization comparison across global, per-layer, per-checkpoint, and two-way schemes
- `scripts/analysis/ablate_pass_two_stage_factorization.py`: checkpoint-vs-layer factorization using time-only checkpoint choice and static-only layer choice
- `scripts/analysis/diagnose_pass_layer_boundaries.py`: depth-boundary and layer-window diagnostics
- `scripts/analysis/ablate_pass_suite.py`: convenience runner that executes the full ablation suite in the recommended order

### PASS Inputs

- Required selector artifact: `pair_metrics.csv`
- Optional selector metadata: `summary.json`
- Required columns: `revision`, `layer_idx`, `avg_main_score`, `dispersion`, `uniformity_score`, `alignment`, `pass_phase_score`, `pass_volatility`, `pass_score`

### PASS Outputs

- Default output root for offline studies: `<selector_run_dir>/pass_ablations/`
- Component ablation output: `<selector_run_dir>/pass_ablations/component_ablation/`
- Normalization ablation output: `<selector_run_dir>/pass_ablations/normalization_ablation/`
- Two-stage factorization output: `<selector_run_dir>/pass_ablations/two_stage_ablation/`
- Boundary diagnostics output: `<selector_run_dir>/pass_ablations/boundary_diagnostics/`

### PASS Commands

Run the fixed PASS selector directly:

```bash
./venv/bin/python scripts/analysis/select_global_pair_unsupervised_pass.py \
  --model_family Pythia \
  --model_size 410m \
  --output_root experiments/results_reruns
```

Run the component ablation on the latest 410m PASS run:

```bash
./venv/bin/python scripts/analysis/ablate_pass_components.py \
  --run_dir "$(cat slurm_logs/latest_p410m_pass_global_pair_selector_root.txt)"
```

Run the normalization ablation:

```bash
./venv/bin/python scripts/analysis/ablate_pass_normalization.py \
  --run_dir "$(cat slurm_logs/latest_p410m_pass_global_pair_selector_root.txt)"
```

Run the two-stage factorization ablation:

```bash
./venv/bin/python scripts/analysis/ablate_pass_two_stage_factorization.py \
  --run_dir "$(cat slurm_logs/latest_p410m_pass_global_pair_selector_root.txt)"
```

Run the boundary diagnostics:

```bash
./venv/bin/python scripts/analysis/diagnose_pass_layer_boundaries.py \
  --run_dir "$(cat slurm_logs/latest_p410m_pass_global_pair_selector_root.txt)"
```

Run the full PASS ablation suite:

```bash
./venv/bin/python scripts/analysis/ablate_pass_suite.py \
  --run_dir "$(cat slurm_logs/latest_p410m_pass_global_pair_selector_root.txt)"
```

### Recommended PASS Ablation Order

1. `ablate_pass_components.py`
2. `ablate_pass_normalization.py`
3. `ablate_pass_two_stage_factorization.py`
4. `diagnose_pass_layer_boundaries.py`
