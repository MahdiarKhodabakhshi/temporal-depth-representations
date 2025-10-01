#!/bin/bash
#SBATCH --account=def-btaati
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=p410m-pass-pair
#SBATCH --output=slurm_logs/p410m-pass-pair-%j.out
#SBATCH --error=slurm_logs/p410m-pass-pair-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${REPO_ROOT}" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

cd "${REPO_ROOT}"
module load gcc arrow/22.0.0 2>/dev/null || true
if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
  source "${REPO_ROOT}/venv/bin/activate"
fi

LEGACY_PYTHON="/home/mahdiar/information_flow/venv/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="${PYTHON_BIN}"
elif [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/venv/bin/python"
elif [[ -x "${LEGACY_PYTHON}" ]]; then
  PYTHON="${LEGACY_PYTHON}"
else
  PYTHON="$(command -v python3)"
fi

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
DEFAULT_SHARED_HF_HOME="/scratch/${USER}/hf_cache"
DEFAULT_SHARED_DATASETS_CACHE="/home/mahdiar/links/projects/def-btaati/mahdiar/information_flow/.hf_cache/datasets"
DEFAULT_SHARED_MODULES_CACHE="/home/mahdiar/links/projects/def-btaati/mahdiar/information_flow/.hf_cache/modules"
if [[ ! -d "${DEFAULT_SHARED_HF_HOME}" ]]; then
  DEFAULT_SHARED_HF_HOME="${REPO_ROOT}/.hf_cache"
fi
if [[ ! -d "${DEFAULT_SHARED_DATASETS_CACHE}" ]]; then
  DEFAULT_SHARED_DATASETS_CACHE="${REPO_ROOT}/.hf_cache/datasets"
fi
if [[ ! -d "${DEFAULT_SHARED_MODULES_CACHE}" ]]; then
  DEFAULT_SHARED_MODULES_CACHE="${REPO_ROOT}/.hf_cache/modules"
fi
export HF_HOME="${HF_HOME:-${DEFAULT_SHARED_HF_HOME}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${DEFAULT_SHARED_DATASETS_CACHE}}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${DEFAULT_SHARED_MODULES_CACHE}}"
export NLTK_DATA="${NLTK_DATA:-/home/mahdiar/nltk_data}"
export MTEB_DATASET_CONFIG_OVERRIDES="${MTEB_DATASET_CONFIG_OVERRIDES:-mteb/amazon_counterfactual:en,mteb/amazon_reviews_multi:en}"
export IFLOW_TORCH_DTYPE=auto
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/xdg-cache-${SLURM_JOB_ID}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

RUN_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiments/results_reruns}/pythia410m_pass_global_pair_selector_h1002h_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" | tee "${REPO_ROOT}/slurm_logs/latest_p410m_pass_global_pair_selector_root.txt"

if [[ "${PREPARE_MTEB_OFFLINE_ASSETS:-1}" == "1" ]]; then
  "${PYTHON}" "${REPO_ROOT}/scripts/slurm_scripts/prepare_mteb_offline_dataset_assets.py" \
    --project-root "${REPO_ROOT}" \
    --datasets-cache "${HF_DATASETS_CACHE}"
fi

CACHE_PATH="${CACHE_PATH:-${REPO_ROOT}/experiments/results_reruns/task_text_caches/pythia410m_pass_seed0_samples32/shared_sampled_task_texts.json}"
if [[ ! -f "${CACHE_PATH}" ]]; then
  CACHE_PATH=""
fi

echo "PYTHON=${PYTHON}"
echo "HF_HOME=${HF_HOME}"
echo "HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "HF_MODULES_CACHE=${HF_MODULES_CACHE}"
echo "CACHE_PATH=${CACHE_PATH:-<auto-discover>}"

"${PYTHON}" -u "${REPO_ROOT}/scripts/analysis/select_global_pair_unsupervised_pass.py" \
  --model_family Pythia \
  --model_size 410m \
  --expected_revisions step143000,step135000,step128000,step115000,step105000,step95000,step85000,step75000,step65000,step55000 \
  --layer_start 15 \
  --layer_end 24 \
  --pool_samples_per_task 32 \
  --sample_seed 0 \
  --dispersion_num_pairs 50000 \
  --uniformity_num_pairs 50000 \
  --uniformity_temperature 2.0 \
  --alignment_quantile 0.30 \
  --top_m_pairs 10 \
  --reranker_analysis_num_texts 0 \
  --pass_tau_rank 0.01 \
  --pass_tau_spectral 0.1 \
  --pass_weight_dispersion 1.0 \
  --pass_weight_uniformity 0.75 \
  --pass_weight_alignment 0.75 \
  --pass_weight_phase 1.0 \
  --pass_weight_volatility 0.5 \
  --stability_lambda 0.0 \
  --entropy_tiebreak_mode minmax_sum \
  --selection_rules pass_metric \
  --batch_size 32 \
  --max_sample_length 512 \
  --baseline_revision step143000 \
  --cache_task_texts_path "$CACHE_PATH" \
  --output_root "$RUN_ROOT"
