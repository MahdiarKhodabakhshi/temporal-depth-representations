#!/bin/bash
#SBATCH --account=def-btaati
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=p410m-pass-abl
#SBATCH --output=slurm_logs/p410m-pass-abl-%j.out
#SBATCH --error=slurm_logs/p410m-pass-abl-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${REPO_ROOT}" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

cd "${REPO_ROOT}"
mkdir -p slurm_logs

module load gcc arrow/22.0.0 2>/dev/null || true
if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
  # Match the rest of the repo's Slurm environment when the project venv exists.
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export GOTO_NUM_THREADS="${GOTO_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_SIZE="${MODEL_SIZE:-410m}"
RUN_DIR="${RUN_DIR:-}"
PAIR_METRICS_CSV="${PAIR_METRICS_CSV:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

if [[ -z "${RUN_DIR}" && -z "${PAIR_METRICS_CSV}" ]]; then
  case "${MODEL_SIZE}" in
    410m)
      MARKER_PATH="${REPO_ROOT}/slurm_logs/latest_p410m_pass_global_pair_selector_root.txt"
      ;;
    70m)
      MARKER_PATH="${REPO_ROOT}/slurm_logs/latest_p70m_pass_global_pair_selector_root.txt"
      ;;
    *)
      MARKER_PATH=""
      ;;
  esac

  if [[ -n "${MARKER_PATH}" && -f "${MARKER_PATH}" ]]; then
    RUN_DIR="$(<"${MARKER_PATH}")"
  fi
fi

if [[ -n "${RUN_DIR}" && -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${RUN_DIR}/pass_ablations"
fi

echo "MODEL_SIZE=${MODEL_SIZE}"
echo "RUN_DIR=${RUN_DIR:-<auto-resolve>}"
echo "PAIR_METRICS_CSV=${PAIR_METRICS_CSV:-<auto-resolve>}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT:-<selector-default>}"
echo "PYTHON=${PYTHON}"

ARGS=(
  "${REPO_ROOT}/scripts/analysis/ablate_pass_suite.py"
  "--model_size" "${MODEL_SIZE}"
)

if [[ -n "${RUN_DIR}" ]]; then
  ARGS+=("--run_dir" "${RUN_DIR}")
fi
if [[ -n "${PAIR_METRICS_CSV}" ]]; then
  ARGS+=("--pair_metrics_csv" "${PAIR_METRICS_CSV}")
fi
if [[ -n "${OUTPUT_ROOT}" ]]; then
  ARGS+=("--output_root" "${OUTPUT_ROOT}")
fi

"${PYTHON}" -u "${ARGS[@]}"

if [[ -n "${OUTPUT_ROOT}" ]]; then
  echo "${OUTPUT_ROOT}" | tee "${REPO_ROOT}/slurm_logs/latest_p410m_pass_ablation_suite_root.txt"
fi
