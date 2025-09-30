#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
mkdir -p slurm_logs

SBATCH_SCRIPT="${REPO_ROOT}/scripts/slurm_scripts/submit_pythia410m_entropy_metric_sweep_selected_layers_step143000_h100_7h.sbatch"
if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "Missing SBATCH script: ${SBATCH_SCRIPT}" >&2
  exit 1
fi

# User-requested checkpoints only.
CHECKPOINTS=(
  step143000
  step135000
  step128000
  step115000
  step105000
  step95000
  step85000
  step75000
  step65000
  step55000
)

# Optional overrides:
#   BASE_RESULTS_ROOT=/path/to/results_reruns
#   METRIC_LAYER_START=15
#   METRIC_LAYER_END=24
#   TIME_LIMIT=07:00:00
BASE_RESULTS_ROOT="${BASE_RESULTS_ROOT:-${REPO_ROOT}/experiments/results_reruns}"
METRIC_LAYER_START="${METRIC_LAYER_START:-15}"
METRIC_LAYER_END="${METRIC_LAYER_END:-24}"
TIME_LIMIT="${TIME_LIMIT:-07:00:00}"
FORCE_HF_DATASETS_CACHE="${FORCE_HF_DATASETS_CACHE:-${REPO_ROOT}/.hf_cache/datasets}"
FORCE_HF_MODULES_CACHE="${FORCE_HF_MODULES_CACHE:-${REPO_ROOT}/.hf_cache/modules}"
MTEB_DATASET_CONFIG_OVERRIDES="${MTEB_DATASET_CONFIG_OVERRIDES:-mteb/amazon_counterfactual:en,mteb/amazon_reviews_multi:en}"

echo "Submitting ${#CHECKPOINTS[@]} entropy jobs..."
for rev in "${CHECKPOINTS[@]}"; do
  run_tag="pythia410m_entropy_metric_sweep_${rev}_layers${METRIC_LAYER_START}-${METRIC_LAYER_END}_h100_${TIME_LIMIT//:/}_$(date -u +%Y%m%d_%H%M%S)"
  time_tag="${TIME_LIMIT%%:*}h"
  job_name="p410m-${rev}-ent-l$((METRIC_LAYER_END-METRIC_LAYER_START+1))-${time_tag}"

  echo "Submitting ${job_name} (RUN_TAG=${run_tag})"
  sbatch \
    -t "${TIME_LIMIT}" \
    -J "${job_name}" \
    --export=ALL,REVISION="${rev}",RUN_TAG="${run_tag}",BASE_RESULTS_ROOT="${BASE_RESULTS_ROOT}",METRIC_LAYER_START="${METRIC_LAYER_START}",METRIC_LAYER_END="${METRIC_LAYER_END}",FORCE_HF_DATASETS_CACHE="${FORCE_HF_DATASETS_CACHE}",FORCE_HF_MODULES_CACHE="${FORCE_HF_MODULES_CACHE}",MTEB_DATASET_CONFIG_OVERRIDES="${MTEB_DATASET_CONFIG_OVERRIDES}" \
    "${SBATCH_SCRIPT}"
  sleep 1
done

echo "Done."
