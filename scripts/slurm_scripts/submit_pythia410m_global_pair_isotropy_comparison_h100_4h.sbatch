#!/bin/bash
#SBATCH --account=def-btaati
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --job-name=p410m-isotropy
#SBATCH --output=slurm_logs/p410m-isotropy-%j.out
#SBATCH --error=slurm_logs/p410m-isotropy-%j.err

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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export TOKENIZERS_PARALLELISM=false

RUN_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/experiments/results_reruns}/pythia410m_global_pair_isotropy_comparison_h1004h_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" | tee "${REPO_ROOT}/slurm_logs/latest_p410m_global_pair_isotropy_comparison_root.txt"

CACHE_PATH="${CACHE_PATH:-}"

"${REPO_ROOT}/venv/bin/python" -u "${REPO_ROOT}/scripts/analysis/compare_global_pair_isotropy_variants.py" \
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
  --selection_rules alignment_uniformity,alignment_uniformity_entropy_tiebreak \
  --isotropy_variants raw,pc_remove_r1,pc_remove_r2,pc_remove_r4,pc_remove_r8,whiten_full,whiten_dim256,whiten_dim384,flow \
  --flow_shortlist_rule alignment_uniformity \
  --flow_shortlist_size 20 \
  --flow_device auto \
  --flow_num_layers 4 \
  --flow_hidden_dim 512 \
  --flow_num_epochs 200 \
  --flow_batch_size 256 \
  --flow_learning_rate 1e-3 \
  --flow_weight_decay 0.0 \
  --flow_scale_clip 5.0 \
  --flow_grad_clip 5.0 \
  --stability_lambda 0.0 \
  --entropy_tiebreak_mode minmax_sum \
  --batch_size 32 \
  --max_sample_length 512 \
  --baseline_revision step143000 \
  --cache_task_texts_path "$CACHE_PATH" \
  --output_root "$RUN_ROOT"
