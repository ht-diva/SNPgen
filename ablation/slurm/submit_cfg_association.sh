#!/bin/bash
set -euo pipefail

MANIFEST=""
DRY_RUN=0
TRAITS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="${2:?--manifest requires a file}"; shift 2 ;;
    --traits)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do TRAITS+=("$1"); shift; done
      ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      echo "Usage: $0 --manifest FILE [--traits TRAIT ...] [--dry-run]"
      echo "Manifest columns: trait, reference_config, output_dir, cfg1, cfg2, cfg3, cfg5 (tab-separated)."
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MANIFEST" ]]; then
  echo "--manifest FILE is required; no site-specific run manifest is assumed." >&2
  exit 2
fi
if [[ "$MANIFEST" != /* ]]; then MANIFEST="$PWD/$MANIFEST"; fi
if [[ ! -r "$MANIFEST" ]]; then
  echo "Cannot read manifest: $MANIFEST" >&2
  exit 2
fi

SLURM_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SNPGEN_REPO_ROOT="${SNPGEN_REPO_ROOT:-$(cd -- "$SLURM_SCRIPT_DIR/../.." && pwd)}"
source "$SNPGEN_REPO_ROOT/ablation/slurm/slurm_common.sh"
snpgen_repo_root
snpgen_sbatch_site_args
cd "$SNPGEN_REPO_ROOT"

wanted() {
  [[ ${#TRAITS[@]} -eq 0 ]] && return 0
  local candidate="$1"
  local item
  for item in "${TRAITS[@]}"; do [[ "$candidate" == "$item" ]] && return 0; done
  return 1
}

while IFS=$'\t' read -r TRAIT REFERENCE_CONFIG OUTPUT_DIR CFG1 CFG2 CFG3 CFG5; do
  TRIMMED_TRAIT="${TRAIT#"${TRAIT%%[![:space:]]*}"}"
  [[ -z "$TRIMMED_TRAIT" || "$TRIMMED_TRAIT" == \#* || "$TRIMMED_TRAIT" == "trait" ]] && continue
  TRAIT="$TRIMMED_TRAIT"
  wanted "$TRAIT" || continue
  LOG_DIR="slurm_logs/ablation/$TRAIT"
  [[ "$DRY_RUN" -eq 1 ]] || mkdir -p "$LOG_DIR"
  CMD=(sbatch "${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT" --parsable
    --output="$LOG_DIR/%x-%j.log" --error="$LOG_DIR/%x-%j.err"
    --job-name="cfg_assoc_${TRAIT}" ablation/slurm/association.sbatch
    --trait "$TRAIT" --reference-config "$REFERENCE_CONFIG" --output-dir "$OUTPUT_DIR"
    --cohort "cfg1=$CFG1" --cohort "cfg2=$CFG2" --cohort "cfg3=$CFG3" --cohort "cfg5=$CFG5")
  if [[ "$DRY_RUN" -eq 1 ]]; then printf '%q ' "${CMD[@]}"; printf '\n';
  else JOB_ID="$("${CMD[@]}")"; echo "$TRAIT association job=$JOB_ID output=$OUTPUT_DIR"; fi
done < "$MANIFEST"
