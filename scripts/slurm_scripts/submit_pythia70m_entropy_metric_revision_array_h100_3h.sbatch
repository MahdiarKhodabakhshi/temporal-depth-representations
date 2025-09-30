#!/bin/bash
#SBATCH -J p70m-entropy-steps
#SBATCH -A def-btaati
#SBATCH -p gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 03:00:00
#SBATCH --array=0-1
#SBATCH -o slurm_logs/%x-%A_%a.out
#SBATCH -e slurm_logs/%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
mkdir -p slurm_logs

module load gcc arrow/22.0.0 2>/dev/null || true
source "${REPO_ROOT}/venv/bin/activate"

# Offline + cache settings
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export NLTK_DATA="${NLTK_DATA:-${REPO_ROOT}/.nltk_data}"
export MTEB_DATASET_CONFIG_OVERRIDES="mteb/amazon_counterfactual:en"

# Avoid deprecated cache warning
unset TRANSFORMERS_CACHE

# Ensure both MTEB and wikitext are available in the active datasets cache root
SHARED_DATASETS_CACHE="${SHARED_DATASETS_CACHE:-}"
if [ -n "$SHARED_DATASETS_CACHE" ] && [ ! -d "$HF_DATASETS_CACHE/wikitext" ] && [ -d "$SHARED_DATASETS_CACHE/wikitext" ]; then
  rsync -a "$SHARED_DATASETS_CACHE/wikitext" "$HF_DATASETS_CACHE/"
fi

# Override at submit time, e.g.:
# sbatch --export=ALL,REV_A=step95000,REV_B=step85000 <this_script>
REV_A="${REV_A:-step115000}"
REV_B="${REV_B:-step105000}"
REVISIONS=("${REV_A}" "${REV_B}")
REVISION="${REVISIONS[$SLURM_ARRAY_TASK_ID]}"

echo "=== Running entropy metrics for Pythia-70m ${REVISION} ==="

"${REPO_ROOT}/venv/bin/python3" -u "${REPO_ROOT}/MTEB-Harness.py" \
  --model_family Pythia \
  --model_size 70m \
  --revision "${REVISION}" \
  --base_results_path experiments/results \
  --purpose run_entropy_metrics \
  --raise_error

echo "=== Completed ${REVISION} ==="
