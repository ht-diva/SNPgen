#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ablation/slurm/submit_ablation_pipeline.sh MODEL [options] [-- run_train overrides...]

MODEL:
  conditional_vae
  conditional_wgan_gp
  conditional_crbm

Options:
  --traits TRAIT [TRAIT ...]      Traits to run. Default: all.
  --traits-config FILE           Trait registry (default: ablation/configs/traits.yaml).
  --output-root PATH              Required root for outputs (or set SNPGEN_OUTPUT_ROOT).
  --run-id ID                      Optional run directory name (default: unique timestamp).
  --dry-run                       Print sbatch commands without submitting.

Examples:
  ablation/slurm/submit_ablation_pipeline.sh conditional_vae --output-root /path/to/results
  ablation/slurm/submit_ablation_pipeline.sh conditional_wgan_gp --output-root /path/to/results --traits example
  ablation/slurm/submit_ablation_pipeline.sh conditional_vae -- --no-wandb training.max_epochs=5
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

MODEL="$1"
shift

if [[ "$MODEL" != "conditional_vae" && "$MODEL" != "conditional_wgan_gp" && "$MODEL" != "conditional_crbm" ]]; then
  echo "Unknown model: $MODEL" >&2
  usage
  exit 2
fi

OUTPUT_ROOT="${SNPGEN_OUTPUT_ROOT:-}"
TRAITS_CONFIG="${SNPGEN_TRAITS_CONFIG:-ablation/configs/traits.yaml}"
DRY_RUN=0
TRAITS=()
TRAIN_ARGS=()
RUN_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --traits)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        TRAITS+=("$1")
        shift
      done
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?--output-root requires a path}"
      shift 2
      ;;
    --traits-config)
      TRAITS_CONFIG="${2:?--traits-config requires a file}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:?--run-id requires an id}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      TRAIN_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

SLURM_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SNPGEN_REPO_ROOT="${SNPGEN_REPO_ROOT:-$(cd -- "$SLURM_SCRIPT_DIR/../.." && pwd)}"
export SNPGEN_TRAITS_CONFIG="$TRAITS_CONFIG"
source "$SNPGEN_REPO_ROOT/ablation/slurm/slurm_common.sh"
snpgen_repo_root
snpgen_sbatch_site_args
cd "$SNPGEN_REPO_ROOT"

if [[ -z "$OUTPUT_ROOT" ]]; then
  echo "Provide --output-root PATH or set SNPGEN_OUTPUT_ROOT; no site-specific scratch path is assumed." >&2
  exit 2
fi
if [[ ! -r "$TRAITS_CONFIG" ]]; then
  echo "Cannot read trait registry: $TRAITS_CONFIG" >&2
  exit 2
fi
snpgen_activate_python
export SNPGEN_OUTPUT_ROOT="$OUTPUT_ROOT"

CONFIG_TRAITS="$(snpgen_python - "$TRAITS_CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf
config = OmegaConf.load(sys.argv[1])
for name in config.get("traits", {}):
    print(name)
PY
)"
if [[ -z "$CONFIG_TRAITS" ]]; then
  echo "Trait registry has no entries: $TRAITS_CONFIG" >&2
  exit 2
fi
if [[ ${#TRAITS[@]} -eq 0 ]]; then
  mapfile -t TRAITS <<< "$CONFIG_TRAITS"
fi
snpgen_python - "$TRAITS_CONFIG" "${TRAITS[@]}" <<'PY'
import sys
from omegaconf import OmegaConf
path, *names = sys.argv[1:]
available = OmegaConf.load(path).get("traits", {})
unknown = [name for name in names if name not in available]
if unknown:
    raise SystemExit(f"Unknown trait(s) {', '.join(unknown)} in {path}; available: {', '.join(available)}")
PY

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%S)_$$_${RANDOM}"
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "--run-id must contain only letters, numbers, period, underscore, and hyphen: $RUN_ID" >&2
  exit 2
fi

submit_or_print() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

for TRAIT in "${TRAITS[@]}"; do
  LOG_DIR="slurm_logs/ablation/$TRAIT"
  if [[ "$DRY_RUN" -eq 0 ]]; then mkdir -p "$LOG_DIR"; fi
  LOG_OUT="$LOG_DIR/%x-%j.log"
  LOG_ERR="$LOG_DIR/%x-%j.err"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    CHECKPOINT_DIR="$OUTPUT_ROOT/$TRAIT/$MODEL/$RUN_ID/checkpoint"
    TRAIN_CHECKPOINT_ARG="$CHECKPOINT_DIR"

    submit_or_print sbatch "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
      --parsable \
      --output="$LOG_OUT" \
      --error="$LOG_ERR" \
      ablation/slurm/train.sbatch \
      "$MODEL" "$TRAIT" "$TRAIN_CHECKPOINT_ARG" "${TRAIN_ARGS[@]}"

    if [[ "$MODEL" == "conditional_vae" ]]; then
      submit_or_print sbatch "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
        --parsable \
        --dependency=afterok:'<TRAIN_JOB>' \
        --output="$LOG_OUT" \
        --error="$LOG_ERR" \
        ablation/slurm/select_generator_checkpoint.sbatch \
        "$MODEL" "$TRAIT" "$CHECKPOINT_DIR"

      submit_or_print sbatch "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
        --parsable \
        --dependency=afterok:'<SELECT_JOB>' \
        --output="$LOG_OUT" \
        --error="$LOG_ERR" \
        ablation/slurm/generate.sbatch \
        "$MODEL" "$TRAIT" "$CHECKPOINT_DIR"

      submit_or_print sbatch "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
        --parsable \
        --dependency=afterok:'<SELECT_JOB>' \
        --output="$LOG_OUT" \
        --error="$LOG_ERR" \
        ablation/slurm/reconstruct.sbatch \
        "$MODEL" "$TRAIT" "$CHECKPOINT_DIR"

      submit_or_print sbatch "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
        --parsable \
        --dependency=afterok:'<GENERATE_JOB>:<RECONSTRUCT_JOB>' \
        --output="$LOG_OUT" \
        --error="$LOG_ERR" \
        ablation/slurm/evaluate.sbatch \
        "$MODEL" "$TRAIT" "$CHECKPOINT_DIR"

      submit_or_print sbatch "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
        --parsable \
        --dependency=afterok:'<GENERATE_JOB>:<RECONSTRUCT_JOB>' \
        --output="$LOG_OUT" \
        --error="$LOG_ERR" \
        ablation/slurm/privacy.sbatch \
        "$MODEL" "$TRAIT" "$CHECKPOINT_DIR"
    else
      submit_or_print sbatch "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
        --parsable \
        --dependency=afterok:'<TRAIN_JOB>' \
        --output="$LOG_OUT" \
        --error="$LOG_ERR" \
        ablation/slurm/select_generator_checkpoint.sbatch \
        "$MODEL" "$TRAIT" "$CHECKPOINT_DIR"

      submit_or_print sbatch "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
        --parsable \
        --dependency=afterok:'<SELECT_JOB>' \
        --output="$LOG_OUT" \
        --error="$LOG_ERR" \
        ablation/slurm/generate.sbatch \
        "$MODEL" "$TRAIT" "$CHECKPOINT_DIR"

      submit_or_print sbatch "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
        --parsable \
        --dependency=afterok:'<GENERATE_JOB>' \
        --output="$LOG_OUT" \
        --error="$LOG_ERR" \
        ablation/slurm/evaluate.sbatch \
        "$MODEL" "$TRAIT" "$CHECKPOINT_DIR"

      submit_or_print sbatch "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
        --parsable \
        --dependency=afterok:'<GENERATE_JOB>' \
        --output="$LOG_OUT" \
        --error="$LOG_ERR" \
        ablation/slurm/privacy.sbatch \
        "$MODEL" "$TRAIT" "$CHECKPOINT_DIR"
    fi

    continue
  fi

  RUN_DIR="$OUTPUT_ROOT/$TRAIT/$MODEL/$RUN_ID"
  if [[ -e "$RUN_DIR" ]]; then
    echo "Run directory already exists; choose a new --run-id: $RUN_DIR" >&2
    exit 2
  fi
  mkdir -p "$(dirname -- "$RUN_DIR")"
  mkdir "$RUN_DIR"
  CHECKPOINT_DIR="$RUN_DIR/checkpoint"
  TRAIN_CHECKPOINT_ARG="$CHECKPOINT_DIR"

  TRAIN_JOB="$(snpgen_submit_job "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
    --parsable \
    --output="$LOG_OUT" \
    --error="$LOG_ERR" \
    ablation/slurm/train.sbatch \
    "$MODEL" "$TRAIT" "$TRAIN_CHECKPOINT_ARG" "${TRAIN_ARGS[@]}")"

  if [[ "$MODEL" == "conditional_vae" ]]; then
    SELECT_JOB="$(snpgen_submit_job "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
      --parsable \
      --dependency=afterok:"$TRAIN_JOB" \
      --output="$LOG_OUT" \
      --error="$LOG_ERR" \
      ablation/slurm/select_generator_checkpoint.sbatch \
      "$MODEL" "$TRAIT" "$CHECKPOINT_DIR")"

    GENERATE_JOB="$(snpgen_submit_job "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
      --parsable \
      --dependency=afterok:"$SELECT_JOB" \
      --output="$LOG_OUT" \
      --error="$LOG_ERR" \
      ablation/slurm/generate.sbatch \
      "$MODEL" "$TRAIT" "$CHECKPOINT_DIR")"

    RECONSTRUCT_JOB="$(snpgen_submit_job "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
      --parsable \
      --dependency=afterok:"$SELECT_JOB" \
      --output="$LOG_OUT" \
      --error="$LOG_ERR" \
      ablation/slurm/reconstruct.sbatch \
      "$MODEL" "$TRAIT" "$CHECKPOINT_DIR")"

    EVALUATE_JOB="$(snpgen_submit_job "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
      --parsable \
      --dependency=afterok:"$GENERATE_JOB:$RECONSTRUCT_JOB" \
      --output="$LOG_OUT" \
      --error="$LOG_ERR" \
      ablation/slurm/evaluate.sbatch \
      "$MODEL" "$TRAIT" "$CHECKPOINT_DIR")"

    PRIVACY_JOB="$(snpgen_submit_job "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
      --parsable \
      --dependency=afterok:"$GENERATE_JOB:$RECONSTRUCT_JOB" \
      --output="$LOG_OUT" \
      --error="$LOG_ERR" \
      ablation/slurm/privacy.sbatch \
      "$MODEL" "$TRAIT" "$CHECKPOINT_DIR")"

    echo "$TRAIT $MODEL checkpoint=$CHECKPOINT_DIR train=$TRAIN_JOB select=$SELECT_JOB generate=$GENERATE_JOB reconstruct=$RECONSTRUCT_JOB evaluate=$EVALUATE_JOB privacy=$PRIVACY_JOB"
  else
    SELECT_JOB="$(snpgen_submit_job "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
      --parsable \
      --dependency=afterok:"$TRAIN_JOB" \
      --output="$LOG_OUT" \
      --error="$LOG_ERR" \
      ablation/slurm/select_generator_checkpoint.sbatch \
      "$MODEL" "$TRAIT" "$CHECKPOINT_DIR")"

GENERATE_JOB="$(snpgen_submit_job "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
  --parsable \
  --dependency=afterok:"$SELECT_JOB" \
  --output="$LOG_OUT" \
  --error="$LOG_ERR" \
  ablation/slurm/generate.sbatch \
  "$MODEL" "$TRAIT" "$CHECKPOINT_DIR")"

EVALUATE_JOB="$(snpgen_submit_job "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
  --parsable \
  --dependency=afterok:"$GENERATE_JOB" \
  --output="$LOG_OUT" \
  --error="$LOG_ERR" \
  ablation/slurm/evaluate.sbatch \
  "$MODEL" "$TRAIT" "$CHECKPOINT_DIR")"

PRIVACY_JOB="$(snpgen_submit_job "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" \
  --parsable \
  --dependency=afterok:"$GENERATE_JOB" \
  --output="$LOG_OUT" \
  --error="$LOG_ERR" \
  ablation/slurm/privacy.sbatch \
  "$MODEL" "$TRAIT" "$CHECKPOINT_DIR")"

    echo "$TRAIT $MODEL checkpoint=$CHECKPOINT_DIR train=$TRAIN_JOB select=$SELECT_JOB generate=$GENERATE_JOB evaluate=$EVALUATE_JOB privacy=$PRIVACY_JOB"
  fi
done
