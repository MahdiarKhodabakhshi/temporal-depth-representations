#!/bin/bash
#SBATCH -J p410m-factorized-abl
#SBATCH -A def-btaati
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 03:00:00
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
mkdir -p slurm_logs

module load gcc arrow/22.0.0 2>/dev/null || true
source "${REPO_ROOT}/venv/bin/activate"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/venv/bin/python3}"
BASE_RESULTS_ROOT="${BASE_RESULTS_ROOT:-${REPO_ROOT}/experiments/results_reruns}"
RUN_TAG="${RUN_TAG:-pythia410m_factorized_selector_ablation_suite_h1003h_${SLURM_JOB_ID}}"
RUN_ROOT="${BASE_RESULTS_ROOT}/${RUN_TAG}"
mkdir -p "${RUN_ROOT}"

echo "RUN_ROOT=${RUN_ROOT}"
echo "${RUN_ROOT}" > "${REPO_ROOT}/slurm_logs/latest_p410m_factorized_selector_ablation_suite_root.txt"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export NLTK_DATA="${NLTK_DATA:-${REPO_ROOT}/.nltk_data}"
export IFLOW_TORCH_DTYPE="${IFLOW_TORCH_DTYPE:-auto}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export GOTO_NUM_THREADS="${GOTO_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CACHE_PATH="${CACHE_PATH:-${RUN_ROOT}/shared_sampled_task_texts.json}"

echo "CACHE_PATH=${CACHE_PATH}"

auto_run() {
  local disp_layer="$1"
  local choice_mode="$2"
  local log_path="${RUN_ROOT}/run_disp${disp_layer}_${choice_mode}.log"

  echo "=== dispersion_layer=${disp_layer} checkpoint_choice_mode=${choice_mode} ==="
  "${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/analysis/select_taskwise_pairs_factorized.py" \
    --model_family Pythia \
    --model_size 410m \
    --expected_revisions step143000,step135000,step128000,step115000,step105000,step95000,step85000,step75000,step65000,step55000 \
    --layer_start 15 \
    --layer_end 24 \
    --dispersion_layer "${disp_layer}" \
    --top_k 3 \
    --layer_score_mode minmax_sum \
    --checkpoint_choice_mode "${choice_mode}" \
    --dispersion_stability_lambda 0.0 \
    --entropy_stability_mu 0.0 \
    --dispersion_num_samples 1000 \
    --sample_seed 0 \
    --dispersion_num_pairs 50000 \
    --batch_size 32 \
    --max_sample_length 512 \
    --baseline_revision step143000 \
    --cache_task_texts_path "${CACHE_PATH}" \
    --output_root "${RUN_ROOT}" \
    2>&1 | tee "${log_path}"
}

for DISP_LAYER in -1 17 20; do
  for CHOICE_MODE in dispersion_only combined_rank; do
    auto_run "${DISP_LAYER}" "${CHOICE_MODE}"
  done
done

echo "=== Completed all 6 ablations ==="
echo "RUN_ROOT=${RUN_ROOT}"
