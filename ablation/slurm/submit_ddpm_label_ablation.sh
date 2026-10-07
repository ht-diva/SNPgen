#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ablation/slurm/submit_ddpm_label_ablation.sh [options] [-- run_label_ablation args...]

Options:
  --traits TRAIT [TRAIT ...]      Traits to run. Default: all registry traits.
  --traits-config FILE            Registry containing each trait's saved_ddpm_config.
  --dry-run                       Print sbatch commands without submitting.

Examples:
  ablation/slurm/submit_ddpm_label_ablation.sh --traits cohort_a --traits-config /path/to/traits.yaml --dry-run
  ablation/slurm/submit_ddpm_label_ablation.sh --traits cohort_a --traits-config /path/to/traits.yaml
  ablation/slurm/submit_ddpm_label_ablation.sh -- --variants permuted prevalence --label-seed 123
EOF
}

DRY_RUN=0
TRAITS=()
EXTRA_ARGS=()
TRAITS_CONFIG="${SNPGEN_TRAITS_CONFIG:-ablation/configs/traits.yaml}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --traits)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        TRAITS+=("$1")
        shift
      done
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --traits-config)
      TRAITS_CONFIG="${2:?--traits-config requires a file}"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
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
source "$SNPGEN_REPO_ROOT/ablation/slurm/slurm_common.sh"
snpgen_repo_root
snpgen_activate_python
snpgen_sbatch_site_args
cd "$SNPGEN_REPO_ROOT"

if [[ ! -r "$TRAITS_CONFIG" ]]; then
  echo "Cannot read trait registry: $TRAITS_CONFIG" >&2
  exit 2
fi
if [[ ${#TRAITS[@]} -eq 0 ]]; then
  CONFIG_TRAITS="$(snpgen_python - "$TRAITS_CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf
for name in OmegaConf.load(sys.argv[1]).get("traits", {}):
    print(name)
PY
)"
  [[ -n "$CONFIG_TRAITS" ]] || { echo "Trait registry has no entries: $TRAITS_CONFIG" >&2; exit 2; }
  mapfile -t TRAITS <<< "$CONFIG_TRAITS"
fi

for TRAIT in "${TRAITS[@]}"; do
  if ! DDPM_CONFIG="$(snpgen_python -m ablation.slurm.resolve_trait_value --traits-config "$TRAITS_CONFIG" --trait "$TRAIT" --field saved_ddpm_config)"; then
    echo "Trait '$TRAIT' has no saved_ddpm_config in $TRAITS_CONFIG; pass --traits-config with a local registry containing that field." >&2
    exit 2
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%q ' sbatch "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" --parsable ablation/slurm/ddpm_label_ablation.sbatch "$DDPM_CONFIG" "${EXTRA_ARGS[@]}"
    printf '\n'
  else
    mkdir -p slurm_logs/ablation
    JOB_ID="$(sbatch "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" --parsable ablation/slurm/ddpm_label_ablation.sbatch "$DDPM_CONFIG" "${EXTRA_ARGS[@]}")"
    echo "$TRAIT ddpm_label_ablation config=$DDPM_CONFIG job=$JOB_ID"
  fi
done
