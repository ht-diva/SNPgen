#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=slurm_logs/ddpm_evaluation/%x-%j.log
#SBATCH --error=slurm_logs/ddpm_evaluation/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ddpm_generate
#SBATCH --time=12:00:00

set -euo pipefail
CHECKPOINT_DIR="${1:?Usage: generate_ddpm.sbatch CHECKPOINT_DIR [generate_ddpm args...]}"
shift

SNPGEN_REPO_ROOT="${SNPGEN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
: "${SNPGEN_REPO_ROOT:?Set SNPGEN_REPO_ROOT when submitting, or submit from the repository root}"
source "$SNPGEN_REPO_ROOT/ablation/slurm/slurm_common.sh"
snpgen_repo_root
snpgen_activate_python
cd "$SNPGEN_REPO_ROOT"
snpgen_python generate_ddpm.py "$@"
