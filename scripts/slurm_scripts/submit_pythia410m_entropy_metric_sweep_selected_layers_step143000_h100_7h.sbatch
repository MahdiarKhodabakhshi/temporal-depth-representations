#!/bin/bash
#SBATCH -J p410m-step143k-ent-l10
#SBATCH -A def-btaati
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 07:00:00
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
mkdir -p slurm_logs

module load gcc arrow/22.0.0 2>/dev/null || true
source "${REPO_ROOT}/venv/bin/activate"

MODEL_FAMILY="${MODEL_FAMILY:-Pythia}"
MODEL_SIZE="${MODEL_SIZE:-410m}"
REVISION="${REVISION:-step143000}"
METRIC_LAYER_START="${METRIC_LAYER_START:-15}"
METRIC_LAYER_END="${METRIC_LAYER_END:-24}"
EVALUATION_LAYER="${EVALUATION_LAYER:-${METRIC_LAYER_END}}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/venv/bin/python3}"

# Offline/cache settings
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
SHARED_HF_HOME="${SHARED_HF_HOME:-${REPO_ROOT}/.hf_cache}"
SHARED_HF_HUB_CACHE="${SHARED_HF_HUB_CACHE:-${SHARED_HF_HOME}/hub}"
PROJECT_HF_DATASETS_CACHE="${REPO_ROOT}/.hf_cache/datasets"
PROJECT_HF_MODULES_CACHE="${REPO_ROOT}/.hf_cache/modules"
export HF_DATASETS_CACHE="${FORCE_HF_DATASETS_CACHE:-${PROJECT_HF_DATASETS_CACHE}}"
export HF_MODULES_CACHE="${FORCE_HF_MODULES_CACHE:-${PROJECT_HF_MODULES_CACHE}}"
export NLTK_DATA="${NLTK_DATA:-${REPO_ROOT}/.nltk_data}"
unset TRANSFORMERS_CACHE
export MTEB_DATASET_CONFIG_OVERRIDES="${MTEB_DATASET_CONFIG_OVERRIDES:-mteb/amazon_counterfactual:en,mteb/amazon_reviews_multi:en}"

# Stage checkpoint snapshot to node-local storage to avoid shared filesystem stalls.
STAGE_MODEL_TO_LOCAL="${STAGE_MODEL_TO_LOCAL:-1}"
if [[ "${STAGE_MODEL_TO_LOCAL}" == "1" && -n "${SLURM_TMPDIR:-}" && "${MODEL_FAMILY}" == "Pythia" ]]; then
  REPO_ID="EleutherAI/pythia-${MODEL_SIZE}"
  REPO_CACHE_NAME="models--${REPO_ID//\//--}"
  SHARED_REPO_CACHE="${SHARED_HF_HUB_CACHE}/${REPO_CACHE_NAME}"
  LOCAL_HF_HOME="${SLURM_TMPDIR}/hf_cache_${SLURM_JOB_ID}"
  LOCAL_HF_HUB_CACHE="${LOCAL_HF_HOME}/hub"
  LOCAL_REPO_CACHE="${LOCAL_HF_HUB_CACHE}/${REPO_CACHE_NAME}"

  if [[ ! -d "${SHARED_REPO_CACHE}" ]]; then
    echo "Missing shared model cache: ${SHARED_REPO_CACHE}" >&2
    exit 2
  fi

  mkdir -p "${LOCAL_REPO_CACHE}/refs" "${LOCAL_REPO_CACHE}/snapshots"
  COMMIT_HASH=""
  if [[ -f "${SHARED_REPO_CACHE}/refs/${REVISION}" ]]; then
    COMMIT_HASH="$(<"${SHARED_REPO_CACHE}/refs/${REVISION}")"
  elif [[ -d "${SHARED_REPO_CACHE}/snapshots/${REVISION}" ]]; then
    COMMIT_HASH="${REVISION}"
  fi

  if [[ -z "${COMMIT_HASH}" || ! -d "${SHARED_REPO_CACHE}/snapshots/${COMMIT_HASH}" ]]; then
    echo "Could not resolve snapshot for ${REPO_ID} revision ${REVISION}" >&2
    exit 3
  fi

  SRC_SNAPSHOT="${SHARED_REPO_CACHE}/snapshots/${COMMIT_HASH}"
  DST_SNAPSHOT="${LOCAL_REPO_CACHE}/snapshots/${COMMIT_HASH}"
  mkdir -p "${DST_SNAPSHOT}"
  echo "Staging ${REPO_ID}@${REVISION} (${COMMIT_HASH}) to local cache: ${DST_SNAPSHOT}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -aL --delete "${SRC_SNAPSHOT}/" "${DST_SNAPSHOT}/"
  else
    cp -aL "${SRC_SNAPSHOT}/." "${DST_SNAPSHOT}/"
  fi

  echo "${COMMIT_HASH}" > "${LOCAL_REPO_CACHE}/refs/${REVISION}"
  if [[ "${REVISION}" != "main" && -f "${SHARED_REPO_CACHE}/refs/main" ]]; then
    cp -f "${SHARED_REPO_CACHE}/refs/main" "${LOCAL_REPO_CACHE}/refs/main"
  fi

  for req in config.json tokenizer.json model.safetensors; do
    if [[ ! -f "${DST_SNAPSHOT}/${req}" ]]; then
      echo "Staged snapshot missing required file: ${DST_SNAPSHOT}/${req}" >&2
      exit 4
    fi
  done

  export HF_HOME="${SHARED_HF_HOME}"
  export HF_HUB_CACHE="${LOCAL_HF_HUB_CACHE}"
  echo "HF_HUB_CACHE=${HF_HUB_CACHE} (node-local staged cache)"
else
  export HF_HOME="${SHARED_HF_HOME}"
  export HF_HUB_CACHE="${SHARED_HF_HUB_CACHE}"
  echo "HF_HUB_CACHE=${HF_HUB_CACHE} (shared cache)"
fi
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "HF_MODULES_CACHE=${HF_MODULES_CACHE}"
echo "MTEB_DATASET_CONFIG_OVERRIDES=${MTEB_DATASET_CONFIG_OVERRIDES}"

# Startup/perf stabilizers
export IFLOW_TORCH_DTYPE="${IFLOW_TORCH_DTYPE:-auto}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Prepare local files required by offline dataset scripts that use relative paths.
if [[ "${PREPARE_MTEB_OFFLINE_ASSETS:-1}" == "1" ]]; then
  echo "Preparing offline MTEB local assets..."
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/slurm_scripts/prepare_mteb_offline_dataset_assets.py" \
    --project-root "${REPO_ROOT}" \
    --datasets-cache "${HF_DATASETS_CACHE}"
fi

BASE_RESULTS_ROOT="${BASE_RESULTS_ROOT:-${REPO_ROOT}/experiments/results_reruns}"
RUN_TAG="${RUN_TAG:-pythia410m_entropy_metric_sweep_${REVISION}_layers${METRIC_LAYER_START}-${METRIC_LAYER_END}_h1007h_${SLURM_JOB_ID}}"
RUN_ROOT="${BASE_RESULTS_ROOT}/${RUN_TAG}"

mkdir -p "${RUN_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "${RUN_ROOT}" > "${REPO_ROOT}/slurm_logs/latest_p410m_entropy_metric_sweep_root.txt"

echo "=== Running entropy metrics for ${MODEL_FAMILY}-${MODEL_SIZE} ${REVISION} layers ${METRIC_LAYER_START}-${METRIC_LAYER_END} ==="
"${PYTHON_BIN}" -u "${REPO_ROOT}/MTEB-Harness.py" \
  --model_family "${MODEL_FAMILY}" \
  --model_size "${MODEL_SIZE}" \
  --revision "${REVISION}" \
  --evaluation_layer "${EVALUATION_LAYER}" \
  --metric_layer_start "${METRIC_LAYER_START}" \
  --metric_layer_end "${METRIC_LAYER_END}" \
  --base_results_path "${RUN_ROOT}" \
  --purpose run_entropy_metrics \
  --raise_error

echo "=== Completed entropy run ==="
