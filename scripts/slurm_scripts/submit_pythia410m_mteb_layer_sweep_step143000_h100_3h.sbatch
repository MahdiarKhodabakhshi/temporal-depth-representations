#!/bin/bash
#SBATCH -J p410m-step143k-alllayers
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

MODEL_FAMILY="${MODEL_FAMILY:-Pythia}"
MODEL_SIZE="${MODEL_SIZE:-410m}"
REVISION="${REVISION:-step143000}"
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-24}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/venv/bin/python3}"

# Offline/cache settings
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
SHARED_HF_HOME="${SHARED_HF_HOME:-${REPO_ROOT}/.hf_cache}"
SHARED_HF_HUB_CACHE="${SHARED_HF_HUB_CACHE:-${SHARED_HF_HOME}/hub}"
PROJECT_HF_DATASETS_CACHE="${REPO_ROOT}/.hf_cache/datasets"
PROJECT_HF_MODULES_CACHE="${REPO_ROOT}/.hf_cache/modules"
# Use project caches by default to avoid inheriting broken/incomplete submit-shell paths.
export HF_DATASETS_CACHE="${FORCE_HF_DATASETS_CACHE:-${PROJECT_HF_DATASETS_CACHE}}"
export HF_MODULES_CACHE="${FORCE_HF_MODULES_CACHE:-${PROJECT_HF_MODULES_CACHE}}"
export NLTK_DATA="${NLTK_DATA:-${REPO_ROOT}/.nltk_data}"
unset TRANSFORMERS_CACHE

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

  # Keep HF_HOME shared so datasets/modules cache remains available offline.
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
RUN_TAG="${RUN_TAG:-pythia410m_mteb_layer_sweep_${REVISION}_layers${LAYER_START}-${LAYER_END}_h1003h_${SLURM_JOB_ID}}"
RUN_ROOT="${BASE_RESULTS_ROOT}/${RUN_TAG}"

mkdir -p "${RUN_ROOT}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "${RUN_ROOT}" > "${REPO_ROOT}/slurm_logs/latest_p410m_mteb_layer_sweep_root.txt"

for layer in $(seq "${LAYER_START}" "${LAYER_END}"); do
  echo "=== Running ${MODEL_FAMILY}-${MODEL_SIZE} ${REVISION} layer_${layer} ==="
  "${PYTHON_BIN}" -u "${REPO_ROOT}/MTEB-Harness.py" \
    --model_family "${MODEL_FAMILY}" \
    --model_size "${MODEL_SIZE}" \
    --revision "${REVISION}" \
    --evaluation_layer "${layer}" \
    --base_results_path "${RUN_ROOT}" \
    --purpose run_tasks \
    --raise_error
done

echo "=== Computing average main score across layers ==="
RESULT_ROOT="${RUN_ROOT}" MODEL_FAMILY="${MODEL_FAMILY}" MODEL_SIZE="${MODEL_SIZE}" REVISION="${REVISION}" \
"${PYTHON_BIN}" - <<'PY'
import csv
import glob
import json
import os
import re
import statistics

base = os.environ["RESULT_ROOT"]
model_family = os.environ["MODEL_FAMILY"]
model_size = os.environ["MODEL_SIZE"]
revision = os.environ["REVISION"]

mteb_dir = os.path.join(base, model_family, model_size, revision, "mteb")
rows = []
for layer_dir in glob.glob(os.path.join(mteb_dir, "layer_*")):
    m = re.fullmatch(r"layer_(\d+)", os.path.basename(layer_dir))
    if not m:
        continue
    layer_idx = int(m.group(1))
    scores = []
    for fp in glob.glob(os.path.join(layer_dir, "*.json")):
        if os.path.basename(fp) == "model_meta.json":
            continue
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        try:
            scores.append(float(data["scores"]["test"][0]["main_score"]))
        except Exception:
            pass
    if scores:
        rows.append((layer_idx, len(scores), statistics.mean(scores)))

rows.sort(key=lambda x: x[0])
if not rows:
    raise RuntimeError(f"No layer results found under {mteb_dir}")

out_dir = os.path.join(base, model_family, model_size, revision, "average_main_score")
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "avg_main_score_by_layer.csv")
md_path = os.path.join(out_dir, "avg_main_score_by_layer.md")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["layer", "num_tasks", "avg_main_score"])
    for layer_idx, n, avg in rows:
        w.writerow([f"layer_{layer_idx}", n, f"{avg:.10f}"])

best = max(rows, key=lambda x: x[2])
with open(md_path, "w", encoding="utf-8") as f:
    f.write("| layer | num_tasks | avg_main_score |\n")
    f.write("|:--|--:|--:|\n")
    for layer_idx, n, avg in rows:
        f.write(f"| layer_{layer_idx} | {n} | {avg:.6f} |\n")
    f.write(
        f"\nBest layer: `layer_{best[0]}` with avg main score `{best[2]:.6f}` "
        f"across `{best[1]}` tasks.\n"
    )

print(f"Wrote: {csv_path}")
print(f"Wrote: {md_path}")
PY

echo "=== Completed all layers ==="
