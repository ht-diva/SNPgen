#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=slurm_logs/ablation/%x-%j.log
#SBATCH --error=slurm_logs/ablation/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=abl_evaluate
#SBATCH --time=09:00:00

set -euo pipefail

MODEL="${1:?Usage: evaluate.sbatch MODEL TRAIT CHECKPOINT_DIR [run_evaluate args...]}"
TRAIT="${2:?Usage: evaluate.sbatch MODEL TRAIT CHECKPOINT_DIR [run_evaluate args...]}"
CHECKPOINT_DIR="${3:?Usage: evaluate.sbatch MODEL TRAIT CHECKPOINT_DIR [run_evaluate args...]}"
shift 3

SNPGEN_REPO_ROOT="${SNPGEN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
: "${SNPGEN_REPO_ROOT:?Set SNPGEN_REPO_ROOT when submitting, or submit from the repository root}"
source "$SNPGEN_REPO_ROOT/ablation/slurm/slurm_common.sh"
snpgen_repo_root
snpgen_activate_python
cd "$SNPGEN_REPO_ROOT"

if [[ "$MODEL" == "conditional_vae" ]]; then
  SYN_TYPES=(complete augmented reconstructed)
else
  SYN_TYPES=(complete augmented)
fi

snpgen_python -m ablation.evaluation.run_evaluate \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --syn-dataset-types "${SYN_TYPES[@]}" \
  "$@"
