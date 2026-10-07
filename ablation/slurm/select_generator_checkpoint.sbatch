#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=slurm_logs/ablation/%x-%j.log
#SBATCH --error=slurm_logs/ablation/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=abl_select_gen
#SBATCH --time=02:00:00

set -euo pipefail

MODEL="${1:?Usage: select_generator_checkpoint.sbatch MODEL TRAIT CHECKPOINT_DIR [selector args...]}"
TRAIT="${2:?Usage: select_generator_checkpoint.sbatch MODEL TRAIT CHECKPOINT_DIR [selector args...]}"
CHECKPOINT_DIR="${3:?Usage: select_generator_checkpoint.sbatch MODEL TRAIT CHECKPOINT_DIR [selector args...]}"
shift 3

if [[ "$MODEL" != "conditional_vae" && "$MODEL" != "conditional_wgan_gp" && "$MODEL" != "conditional_crbm" ]]; then
  echo "select_generator_checkpoint.sbatch only supports conditional_vae, conditional_wgan_gp, or conditional_crbm, got: $MODEL" >&2
  exit 2
fi

SNPGEN_REPO_ROOT="${SNPGEN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
: "${SNPGEN_REPO_ROOT:?Set SNPGEN_REPO_ROOT when submitting, or submit from the repository root}"
source "$SNPGEN_REPO_ROOT/ablation/slurm/slurm_common.sh"
snpgen_repo_root
snpgen_activate_python
cd "$SNPGEN_REPO_ROOT"

snpgen_python -m ablation.select_generator_checkpoint \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  "$@"
