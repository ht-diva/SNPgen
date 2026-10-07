#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=slurm_logs/ablation/%x-%j.log
#SBATCH --error=slurm_logs/ablation/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ddpm_label_ablation
#SBATCH --time=06:00:00

set -euo pipefail

CHECKPOINT_DIR="${1:?Usage: ddpm_label_ablation.sbatch DDPM_CHECKPOINT_DIR_OR_CONFIG [run_label_ablation args...]}"
shift

SNPGEN_REPO_ROOT="${SNPGEN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
: "${SNPGEN_REPO_ROOT:?Set SNPGEN_REPO_ROOT when submitting, or submit from the repository root}"
source "$SNPGEN_REPO_ROOT/ablation/slurm/slurm_common.sh"
snpgen_repo_root
snpgen_activate_python
cd "$SNPGEN_REPO_ROOT"

snpgen_python -m ablation.ddpm.run_label_ablation \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --skip-real-training \
  "$@"
