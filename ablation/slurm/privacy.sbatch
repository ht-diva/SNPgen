#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=slurm_logs/ablation/%x-%j.log
#SBATCH --error=slurm_logs/ablation/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=96GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=abl_privacy
#SBATCH --time=01:30:00

set -euo pipefail

MODEL="${1:?Usage: privacy.sbatch MODEL TRAIT CHECKPOINT_DIR [run_privacy args...]}"
TRAIT="${2:?Usage: privacy.sbatch MODEL TRAIT CHECKPOINT_DIR [run_privacy args...]}"
CHECKPOINT_DIR="${3:?Usage: privacy.sbatch MODEL TRAIT CHECKPOINT_DIR [run_privacy args...]}"
shift 3

SNPGEN_REPO_ROOT="${SNPGEN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
: "${SNPGEN_REPO_ROOT:?Set SNPGEN_REPO_ROOT when submitting, or submit from the repository root}"
source "$SNPGEN_REPO_ROOT/ablation/slurm/slurm_common.sh"
snpgen_repo_root
snpgen_activate_python
cd "$SNPGEN_REPO_ROOT"
export PYTHONUNBUFFERED=1

PRIVACY_ARGS=()
if [[ "$MODEL" == "conditional_vae" ]]; then
  PRIVACY_ARGS+=(--evaluate-reconstructed)
fi

snpgen_python -u -m ablation.evaluation.run_privacy \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --model-name "${TRAIT}_${MODEL}" \
  "${PRIVACY_ARGS[@]}" \
  "$@"
