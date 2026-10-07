#!/usr/bin/env bash
# Shared site configuration for the public SLURM launchers.

snpgen_slurm_dir() {
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd
}

snpgen_repo_root() {
  local root="${SNPGEN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
  if [[ -z "$root" || ! -d "$root/ablation" ]]; then
    echo "Set SNPGEN_REPO_ROOT to the repository root (containing ablation/) before submitting." >&2
    return 2
  fi
  SNPGEN_REPO_ROOT="$(cd -- "$root" && pwd)"
  export SNPGEN_REPO_ROOT
}

snpgen_activate_python() {
  if [[ -n "${SNPGEN_PYTHON:-}" ]]; then
    command -v "$SNPGEN_PYTHON" >/dev/null 2>&1 || {
      echo "SNPGEN_PYTHON is not executable or not on PATH: $SNPGEN_PYTHON" >&2
      return 2
    }
    export SNPGEN_PYTHON
  else
    if [[ -z "${SNPGEN_CONDA_SH:-}" ]]; then
      echo "Set SNPGEN_CONDA_SH to conda.sh, or set SNPGEN_PYTHON to a Python executable." >&2
      return 2
    fi
    if [[ -z "${SNPGEN_CONDA_ENV:-}" ]]; then
      echo "Set SNPGEN_CONDA_ENV to your project Conda environment." >&2
      return 2
    fi
    [[ -r "$SNPGEN_CONDA_SH" ]] || {
      echo "Cannot read SNPGEN_CONDA_SH: $SNPGEN_CONDA_SH" >&2
      return 2
    }
    # shellcheck disable=SC1090
    source "$SNPGEN_CONDA_SH"
    conda activate "$SNPGEN_CONDA_ENV"
    SNPGEN_PYTHON="$(command -v python)"
    export SNPGEN_PYTHON
  fi
  export PYTHONPATH="$SNPGEN_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
}

snpgen_sbatch_site_args() {
  SNPGEN_SBATCH_ARGS_ARRAY=()
  if [[ -n "${SNPGEN_SBATCH_ARGS:-}" ]]; then
    # Space-separated sbatch options; quote values containing spaces in a site config wrapper.
    read -r -a SNPGEN_SBATCH_ARGS_ARRAY <<< "$SNPGEN_SBATCH_ARGS"
  fi
}

snpgen_python() {
  "${SNPGEN_PYTHON:-python}" "$@"
}

snpgen_submit_job() {
  local result
  result="$(sbatch "$@")" || return
  # --parsable can include a cluster suffix; dependencies need only the job ID.
  result="${result%%;*}"
  if [[ ! "$result" =~ ^[0-9]+$ ]]; then
    echo "Unexpected sbatch --parsable response: $result" >&2
    return 2
  fi
  printf '%s\n' "$result"
}
