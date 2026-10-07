#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=slurm_logs/ablation/%x-%j.log
#SBATCH --error=slurm_logs/ablation/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=abl_train
#SBATCH --time=36:00:00

set -euo pipefail

MODEL="${1:?Usage: train.sbatch MODEL TRAIT CHECKPOINT_DIR|AUTO [run_train overrides...]}"
TRAIT="${2:?Usage: train.sbatch MODEL TRAIT CHECKPOINT_DIR|AUTO [run_train overrides...]}"
CHECKPOINT_DIR="${3:?Usage: train.sbatch MODEL TRAIT CHECKPOINT_DIR|AUTO [run_train overrides...]}"
shift 3

SNPGEN_REPO_ROOT="${SNPGEN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
: "${SNPGEN_REPO_ROOT:?Set SNPGEN_REPO_ROOT when submitting, or submit from the repository root}"
source "$SNPGEN_REPO_ROOT/ablation/slurm/slurm_common.sh"
snpgen_repo_root
snpgen_activate_python
cd "$SNPGEN_REPO_ROOT"

TRAITS_CONFIG="${SNPGEN_TRAITS_CONFIG:-ablation/configs/traits.yaml}"
export SNPGEN_TRAIT="$TRAIT"
export SNPGEN_TRAITS_CONFIG="$TRAITS_CONFIG"

OUTPUT_ARGS=()
if [[ "$CHECKPOINT_DIR" != "AUTO" ]]; then
  mkdir -p "$CHECKPOINT_DIR"
  OUTPUT_ARGS=(--output-dir "$CHECKPOINT_DIR")
fi

if [[ "$MODEL" == "conditional_vae" ]]; then
  SAVED_VAE_CONFIG="$(snpgen_python - <<'PY'
from omegaconf import OmegaConf
import os
cfg = OmegaConf.load(os.environ["SNPGEN_TRAITS_CONFIG"])
trait = cfg.traits[os.environ["SNPGEN_TRAIT"]]
print(trait.get("saved_vae_config", "") or "")
PY
)"
  if [[ -n "$SAVED_VAE_CONFIG" ]]; then
    VAE_CONFIGS=(--config "$SAVED_VAE_CONFIG" ablation/configs/conditional_vae_overlay.yaml)
  else
    VAE_CONFIGS=(--config ablation/configs/conditional_vae.yaml)
  fi
  snpgen_python -m ablation.run_train \
    "${VAE_CONFIGS[@]}" \
    --traits-config "$TRAITS_CONFIG" \
    --trait "$TRAIT" \
    --ablation-name "$MODEL" \
    "${OUTPUT_ARGS[@]}" \
    "$@"
elif [[ "$MODEL" == "conditional_wgan_gp" ]]; then
  snpgen_python -m ablation.run_train \
    --config ablation/configs/conditional_wgan_gp.yaml \
    --traits-config "$TRAITS_CONFIG" \
    --trait "$TRAIT" \
    --ablation-name "$MODEL" \
    "${OUTPUT_ARGS[@]}" \
    "$@"
elif [[ "$MODEL" == "conditional_crbm" ]]; then
  snpgen_python -m ablation.run_train \
    --config ablation/configs/conditional_crbm.yaml \
    --traits-config "$TRAITS_CONFIG" \
    --trait "$TRAIT" \
    --ablation-name "$MODEL" \
    "${OUTPUT_ARGS[@]}" \
    "$@"
else
  echo "Unknown ablation model: $MODEL" >&2
  exit 2
fi
