#!/bin/bash
#SBATCH -J p70m-mteb-alllayers
#SBATCH -A def-btaati
#SBATCH -p gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 05:00:00
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
mkdir -p slurm_logs

module load gcc arrow/22.0.0 2>/dev/null || true
source "${REPO_ROOT}/venv/bin/activate"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export NLTK_DATA="${NLTK_DATA:-${REPO_ROOT}/.nltk_data}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/venv/bin/python3}"

for layer in $(seq 0 6); do
    echo "=== Running layer ${layer} ==="

    results_dir="experiments/results/Pythia/70m/main/mteb/layer_${layer}"
    mkdir -p "$results_dir"
    find "$results_dir" -maxdepth 1 -type f -name '*.json' -delete

        "$PYTHON_BIN" -u "${REPO_ROOT}/MTEB-Harness.py" \
        --model_family "Pythia" \
        --model_size "70m" \
        --revision "main" \
        --evaluation_layer "$layer" \
        --base_results_path "experiments/results" \
        --purpose run_tasks \
        --overwrite_results \
        --raise_error
done

echo "All layers completed."
