#!/usr/bin/env bash
# Submit generation followed by parallel utility and privacy evaluations.
set -euo pipefail

CALLER_DIR="$PWD"
MANIFEST=""
DRY_RUN=0
RESUME=0
OVERWRITE=0
SKIP_REAL=0
SEED=42
DATASET_PATH=""
CONFIG_OVERRIDE=""
declare -a CFG_SCALES=()
declare -a RUN_TRAITS=()
declare -a RUN_CHECKPOINTS=()
declare -a RUN_CONFIGS=()

usage() {
  cat <<'EOF'
Usage:
  submit_ddpm_evaluation.sh --manifest FILE [options]
  submit_ddpm_evaluation.sh --run TRAIT CHECKPOINT [--run TRAIT CHECKPOINT ...] [options]

Sources are explicit: there is no site-specific default manifest. Manifest
rows are tab-separated `trait<TAB>checkpoint_path[<TAB>config_path]`; blank
lines and lines beginning with # are ignored. Relative paths are resolved
from the directory where this command is invoked. A config defaults to
config.yaml beside its checkpoint. `--config FILE` applies to all runs.

Options:
  --cfg-scale SCALE       Repeat to submit a guidance-scale sweep
  --dataset-path FILE     Override the path recorded in each run config
  --seed N                Inference/evaluation seed (default: 42)
  --resume, --skip-existing  Keep existing synthetic HDF5 outputs
  --overwrite             Explicitly replace existing synthetic HDF5 outputs
  --skip-real-training    Reuse/skip the real-data baseline (default trains it)
  --dry-run               Print planned commands; create no directories or jobs
  -h, --help              Show this help without loading a Python environment

Set SNPGEN_REPO_ROOT, SNPGEN_PYTHON or SNPGEN_CONDA_SH/SNPGEN_CONDA_ENV, and
optionally SNPGEN_SBATCH_ARGS for site-specific Slurm options.
EOF
}

fail() { echo "error: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) [[ $# -ge 2 ]] || fail "--manifest requires a file"; MANIFEST="$2"; shift 2 ;;
    --run)
      [[ $# -ge 3 ]] || fail "--run requires TRAIT and CHECKPOINT_PATH"
      RUN_TRAITS+=("$2"); RUN_CHECKPOINTS+=("$3"); RUN_CONFIGS+=("")
      shift 3
      ;;
    --config) [[ $# -ge 2 ]] || fail "--config requires a file"; CONFIG_OVERRIDE="$2"; shift 2 ;;
    --dataset-path) [[ $# -ge 2 ]] || fail "--dataset-path requires a file"; DATASET_PATH="$2"; shift 2 ;;
    --cfg-scale) [[ $# -ge 2 ]] || fail "--cfg-scale requires a number"; CFG_SCALES+=("$2"); shift 2 ;;
    --seed) [[ $# -ge 2 ]] || fail "--seed requires an integer"; SEED="$2"; shift 2 ;;
    --resume|--skip-existing) RESUME=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    --skip-real-training) SKIP_REAL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
done

[[ -n "$MANIFEST" || ${#RUN_TRAITS[@]} -gt 0 ]] || { usage >&2; fail "provide --manifest or at least one --run"; }
[[ -z "$MANIFEST" || ${#RUN_TRAITS[@]} -eq 0 ]] || fail "choose --manifest or --run, not both"
[[ $RESUME -eq 0 || $OVERWRITE -eq 0 ]] || fail "--resume/--skip-existing and --overwrite are mutually exclusive"
[[ "$SEED" =~ ^-?[0-9]+$ ]] || fail "--seed must be an integer"

resolve_from_caller() {
  local path="$1"
  if [[ "$path" = /* ]]; then realpath -m -- "$path"; else realpath -m -- "$CALLER_DIR/$path"; fi
}

declare -a TRAITS=() CHECKPOINTS=() CONFIGS=()
if [[ -n "$MANIFEST" ]]; then
  MANIFEST="$(resolve_from_caller "$MANIFEST")"
  [[ -r "$MANIFEST" ]] || fail "cannot read manifest: $MANIFEST"
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "${line//[[:space:]]/}" || "$line" =~ ^[[:space:]]*# ]] && continue
    IFS=$'\t' read -r -a columns <<< "$line"
    if [[ ${#columns[@]} -ge 2 && "${columns[0],,}" == "trait" && "${columns[1],,}" == "checkpoint_path" ]]; then continue; fi
    [[ ${#columns[@]} -ge 2 && ${#columns[@]} -le 3 ]] || fail "invalid manifest row (expected 2 or 3 tab-separated columns): $line"
    [[ -n "${columns[0]}" && -n "${columns[1]}" ]] || fail "manifest trait and checkpoint path must be non-empty: $line"
    TRAITS+=("${columns[0]}")
    CHECKPOINTS+=("$(resolve_from_caller "${columns[1]}")")
    if [[ -n "${columns[2]:-}" ]]; then CONFIGS+=("$(resolve_from_caller "${columns[2]}")"); else CONFIGS+=(""); fi
  done < "$MANIFEST"
else
  for ((i=0; i<${#RUN_TRAITS[@]}; i++)); do
    TRAITS+=("${RUN_TRAITS[$i]}")
    CHECKPOINTS+=("$(resolve_from_caller "${RUN_CHECKPOINTS[$i]}")")
    CONFIGS+=("")
  done
fi
[[ ${#TRAITS[@]} -gt 0 ]] || fail "source contains no runs"

if [[ -n "$CONFIG_OVERRIDE" ]]; then
  CONFIG_OVERRIDE="$(resolve_from_caller "$CONFIG_OVERRIDE")"
  for ((i=0; i<${#CONFIGS[@]}; i++)); do CONFIGS[$i]="$CONFIG_OVERRIDE"; done
fi
if [[ -n "$DATASET_PATH" ]]; then DATASET_PATH="$(resolve_from_caller "$DATASET_PATH")"; [[ -f "$DATASET_PATH" ]] || fail "dataset not found: $DATASET_PATH"; fi

for scale in "${CFG_SCALES[@]}"; do
  [[ "$scale" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "invalid CFG scale '$scale'"
done
for ((i=0; i<${#TRAITS[@]}; i++)); do
  [[ "${TRAITS[$i]}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "unsafe trait name: ${TRAITS[$i]}"
  [[ -f "${CHECKPOINTS[$i]}" ]] || fail "checkpoint not found: ${CHECKPOINTS[$i]}"
  if [[ -z "${CONFIGS[$i]}" ]]; then CONFIGS[$i]="$(dirname -- "${CHECKPOINTS[$i]}")/config.yaml"; fi
  [[ -r "${CONFIGS[$i]}" ]] || fail "config not found/readable: ${CONFIGS[$i]}"
done

SLURM_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SNPGEN_REPO_ROOT="${SNPGEN_REPO_ROOT:-$(cd -- "$SLURM_SCRIPT_DIR/../.." && pwd)}"
source "$SNPGEN_REPO_ROOT/ablation/slurm/slurm_common.sh"
snpgen_repo_root
snpgen_sbatch_site_args

# Parse and validate every YAML file before submitting the first task. Dry-run
# stays useful from login environments without Python/Conda configured.
if [[ $DRY_RUN -eq 0 ]]; then
  snpgen_activate_python
  py_args=()
  scales=("")
  if [[ ${#CFG_SCALES[@]} -gt 0 ]]; then scales=("${CFG_SCALES[@]}"); fi
  for ((i=0; i<${#TRAITS[@]}; i++)); do
    checkpoint_dir="$(dirname -- "${CHECKPOINTS[$i]}")"
    for scale in "${scales[@]}"; do
      evaluation_dir="$checkpoint_dir"
      [[ -n "$scale" ]] && evaluation_dir="$checkpoint_dir/cfg_scale_${scale//./p}"
      py_args+=("${CHECKPOINTS[$i]}" "${CONFIGS[$i]}" "$evaluation_dir" "$scale")
    done
  done
  [[ -n "$DATASET_PATH" ]] && py_args+=("--dataset-override" "$DATASET_PATH")
  [[ $OVERWRITE -eq 1 ]] && py_args+=("--allow-overwrite")
  snpgen_python - "${py_args[@]}" <<'PY'
import os, sys
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)
args = sys.argv[1:]
override = None
allow_overwrite = "--allow-overwrite" in args
if allow_overwrite: args.remove("--allow-overwrite")
if "--dataset-override" in args:
    i = args.index("--dataset-override"); override = args[i + 1]; args = args[:i]
if len(args) % 4: raise SystemExit("internal error: expected checkpoint/config/output/scale groups")
errors = []
for checkpoint, config_path, output_dir, scale in zip(args[::4], args[1::4], args[2::4], args[3::4]):
    try:
        config = OmegaConf.load(config_path)
        if override: config.dataset_path = override
        if scale:
            scale_key = "model.params.sampler_config.params.guider_config.params.scale"
            if OmegaConf.select(config, scale_key) is None: raise ValueError(f"missing CFG scale at {scale_key}")
            OmegaConf.update(config, scale_key, float(scale))
        dataset = override or config.get("dataset_path")
        if config.get("model") is None: raise ValueError("missing model config")
        if not config.get("data", {}).get("raw_dataset") or not config.get("data", {}).get("dataset"):
            raise ValueError("missing data.raw_dataset or data.dataset config")
        if not dataset: raise ValueError("no dataset_path; provide --dataset-path")
        if not os.path.isfile(dataset): raise ValueError(f"dataset file not found: {dataset}")
        out_config = os.path.join(output_dir, "config.yaml")
        if os.path.isfile(out_config) and not allow_overwrite:
            existing = OmegaConf.to_container(OmegaConf.load(out_config), resolve=True)
            wanted = OmegaConf.to_container(config, resolve=True)
            if existing != wanted: raise ValueError(f"output config differs (use --overwrite): {out_config}")
    except Exception as exc:
        errors.append(f"{config_path} ({checkpoint} -> {output_dir}): {exc}")
if errors:
    raise SystemExit("invalid DDPM run configuration(s):\n" + "\n".join(errors))
PY
fi

# Check every output path before the first job is queued, so a later conflict
# cannot leave only part of the requested run matrix submitted.
scales=("")
if [[ ${#CFG_SCALES[@]} -gt 0 ]]; then scales=("${CFG_SCALES[@]}"); fi
declare -A SEEN_OUTPUT_DIRS=()
for ((i=0; i<${#TRAITS[@]}; i++)); do
  checkpoint_dir="$(dirname -- "${CHECKPOINTS[$i]}")"
  for scale in "${scales[@]}"; do
    evaluation_dir="$checkpoint_dir"
    if [[ -n "$scale" ]]; then evaluation_dir="$checkpoint_dir/cfg_scale_${scale//./p}"; fi
    if [[ -n "${SEEN_OUTPUT_DIRS[$evaluation_dir]:-}" ]]; then fail "multiple requested runs share an evaluation directory: $evaluation_dir"; fi
    SEEN_OUTPUT_DIRS[$evaluation_dir]=1
    if [[ $RESUME -eq 0 && $OVERWRITE -eq 0 ]]; then
      for output in syn_complete_dataset.hdf5 syn_augmented_dataset.hdf5 syn_augmented_dataset_binary_balanced.hdf5; do
        [[ ! -e "$evaluation_dir/$output" ]] || fail "existing output requires --resume or --overwrite: $evaluation_dir/$output"
      done
    fi
  done
done

cd "$SNPGEN_REPO_ROOT"
print_or_submit() {
  if [[ $DRY_RUN -eq 1 ]]; then printf '%q ' "$@"; printf '\n'; return 0; fi
  snpgen_submit_job "$@"
}

for ((i=0; i<${#TRAITS[@]}; i++)); do
  trait="${TRAITS[$i]}"; checkpoint="${CHECKPOINTS[$i]}"; config="${CONFIGS[$i]}"
  checkpoint_dir="$(dirname -- "$checkpoint")"
  log_dir="slurm_logs/ddpm_evaluation/$trait"
  [[ $DRY_RUN -eq 0 ]] && mkdir -p "$log_dir"

  scales=("")
  if [[ ${#CFG_SCALES[@]} -gt 0 ]]; then scales=("${CFG_SCALES[@]}"); fi
  for scale in "${scales[@]}"; do
    evaluation_dir="$checkpoint_dir"
    generation_args=(--checkpoint "$checkpoint" --config "$config" --seed "$SEED")
    [[ -n "$DATASET_PATH" ]] && generation_args+=(--dataset-path "$DATASET_PATH")
    if [[ -n "$scale" ]]; then
      tag="${scale//./p}"
      evaluation_dir="$checkpoint_dir/cfg_scale_$tag"
      generation_args+=(--cfg-scale "$scale" --output-dir "$evaluation_dir")
      scale_log="$log_dir/cfg_scale_$tag"
    else
      scale_log="$log_dir"
    fi
    [[ $DRY_RUN -eq 1 ]] || mkdir -p "$scale_log"
    if [[ $RESUME -eq 1 ]]; then generation_args+=(--skip-existing); fi
    if [[ $OVERWRITE -eq 1 ]]; then generation_args+=(--overwrite); fi

    generate_args=("${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT"
      --parsable --output="$scale_log/%x-%j.log" --error="$scale_log/%x-%j.err"
      ablation/slurm/generate_ddpm.sbatch "$checkpoint_dir" "${generation_args[@]}")
    if [[ $DRY_RUN -eq 1 ]]; then
      printf '%q ' sbatch "${generate_args[@]}"; printf '\n'
      generate_job="<GENERATE_JOB>"
    else
      generate_job="$(snpgen_submit_job "${generate_args[@]}")"
    fi

    evaluate_args=("${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT"
      --parsable --dependency="afterok:$generate_job"
      --output="$scale_log/%x-%j.log" --error="$scale_log/%x-%j.err"
      ablation/slurm/evaluate.sbatch ddpm "$trait" "$evaluation_dir" --seed "$SEED"
      --force-retrain xgboost xgboost_balanced catboost prs)
    [[ $SKIP_REAL -eq 1 ]] && evaluate_args+=(--skip-real-training)
    if [[ $DRY_RUN -eq 1 ]]; then
      printf '%q ' sbatch "${evaluate_args[@]}"; printf '\n'
      evaluate_job="<EVALUATE_JOB>"
    else
      evaluate_job="$(snpgen_submit_job "${evaluate_args[@]}")"
    fi

    privacy_args=("${SNPGEN_SBATCH_ARGS_ARRAY[@]}" --chdir="$SNPGEN_REPO_ROOT"
      --parsable --dependency="afterok:$generate_job"
      --output="$scale_log/%x-%j.log" --error="$scale_log/%x-%j.err"
      ablation/slurm/privacy.sbatch ddpm "$trait" "$evaluation_dir")
    if [[ $DRY_RUN -eq 1 ]]; then
      printf '%q ' sbatch "${privacy_args[@]}"; printf '\n'
    else
      privacy_job="$(snpgen_submit_job "${privacy_args[@]}")"
      echo "$trait checkpoint=$checkpoint evaluation_dir=$evaluation_dir cfg_scale=${scale:-config_default} generate=$generate_job evaluate=$evaluate_job privacy=$privacy_job"
    fi
  done
done
