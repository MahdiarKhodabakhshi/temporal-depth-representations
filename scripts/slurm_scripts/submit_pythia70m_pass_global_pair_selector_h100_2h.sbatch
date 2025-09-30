#!/bin/bash
#SBATCH --account=def-btaati
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=p70m-pass-pair
#SBATCH --output=slurm_logs/p70m-pass-pair-%j.out
#SBATCH --error=slurm_logs/p70m-pass-pair-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
module load gcc arrow/22.0.0 2>/dev/null || true
source "${REPO_ROOT}/venv/bin/activate"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export NLTK_DATA="${NLTK_DATA:-${REPO_ROOT}/.nltk_data}"
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

RUN_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiments/results_reruns}/pythia70m_pass_global_pair_selector_h1002h_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" | tee "${REPO_ROOT}/slurm_logs/latest_p70m_pass_global_pair_selector_root.txt"

CACHE_PATH="${CACHE_PATH:-}"

"${REPO_ROOT}/venv/bin/python" -u "${REPO_ROOT}/scripts/analysis/select_global_pair_unsupervised_pass.py" \
  --model_family Pythia \
  --model_size 70m \
  --entropy_root "${REPO_ROOT}/experiments/results/Pythia/70m" \
  --main_runs_root "${REPO_ROOT}/experiments/results/Pythia/70m" \
  --expected_revisions main,step45000,step55000,step65000,step75000,step85000,step95000,step105000,step115000,step125000 \
  --layer_start 0 \
  --layer_end 6 \
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
  --baseline_revision main \
  --cache_task_texts_path "$CACHE_PATH" \
  --output_root "$RUN_ROOT"
