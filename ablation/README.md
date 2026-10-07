# SNPgen ablation models and evaluation

This directory contains runnable cVAE, CRBM and WGAN-GP training, generation and evaluation code built on the repository's `snpgen/` package. The workflows train from a genotype dataset and the public YAML configs in `ablation/configs/`; they do not require an author's checkpoint. HDF5 input must be readable by `snpgen.data.loader.SplitDataset` and contain the genotype/phenotype arrays and metadata expected by that loader. The configured `seq_len` must match the SNP panel representation used by the dataset.

## Environment

Run commands from the repository root with its source on `PYTHONPATH` (the examples use `PYTHONPATH=.`). The Python environment needs PyTorch, Lightning, OmegaConf, NumPy, h5py, pandas, scikit-learn, SciPy, tqdm, XGBoost, CatBoost and glum. `snpgen.evaluation` imports the model backends when loaded, so those packages are needed even when selecting only `--models prs`. W&B is optional and disabled by default; pass `--use-wandb` to enable it. Use the project's supported package environment for the local `snpgen/` dependencies.

For local training, CPU is the default in these examples. Change `--accelerator` and `--devices` for your hardware. On Slurm, wrappers accept `SNPGEN_CONDA_SH` (the path to `conda.sh`) and `SNPGEN_CONDA_ENV`, or `SNPGEN_PYTHON=/path/to/python` to bypass Conda. Set site options as `SNPGEN_SBATCH_ARGS='--partition=... --account=...'`. The training wrapper requires `--output-root` or `SNPGEN_OUTPUT_ROOT`; pass `--traits-config ablation/configs/traits.yaml` after editing its example data path. See wrapper help for model, trait and dry-run options. Those scripts are scheduler-specific and can be replaced by the same Python commands below on another HPC.

Example training wrapper command (the model is positional; model-specific arguments after `--` pass through to `run_train.py`):

```bash
SNPGEN_PYTHON=/path/to/python SNPGEN_SBATCH_ARGS='--partition=... --account=...' \
  ablation/slurm/submit_ablation_pipeline.sh conditional_crbm \
  --traits example --traits-config ablation/configs/traits.yaml \
  --output-root /path/to/runs --dry-run -- \
  --no-wandb --accelerator cpu --devices 1
```

## Train from scratch

The YAML files are examples for 2048-SNP panels. Set `dataset_path` and `seq_len` for your dataset. The registry in `configs/traits.yaml` is a portable example; edit its `example.dataset_path` and `seq_len` before using `--traits-config`. For one-off training, set the data path directly with an OmegaConf override:

```bash
PYTHONPATH=. python -m ablation.run_train \
  --config ablation/configs/conditional_crbm.yaml \
  --output-dir /path/to/runs/crbm \
  --no-wandb --accelerator cpu --devices 1 \
  dataset_path=/path/to/cohort.hdf5 seq_len=2048
```

```bash
PYTHONPATH=. python -m ablation.run_train \
  --config ablation/configs/conditional_wgan_gp.yaml \
  --output-dir /path/to/runs/wgan \
  --no-wandb --accelerator cpu --devices 1 \
  dataset_path=/path/to/cohort.hdf5 seq_len=2048
```

```bash
PYTHONPATH=. python -m ablation.run_train \
  --config ablation/configs/conditional_vae.yaml \
  --output-dir /path/to/runs/cvae \
  --no-wandb --accelerator cpu --devices 1 \
  dataset_path=/path/to/cohort.hdf5 seq_len=2048
```

For the 1024-SNP breast panel, merge the public architecture overlay after the base cVAE config; it sets four latent channels and KL weight 0.1, yielding latent length 64:

```bash
PYTHONPATH=. python -m ablation.run_train \
  --config ablation/configs/conditional_vae.yaml ablation/configs/conditional_vae_breast.yaml \
  --output-dir /path/to/runs/cvae-breast \
  --no-wandb --accelerator cpu --devices 1 \
  dataset_path=/path/to/breast_cohort.hdf5 seq_len=1024
```

For a short pipeline smoke run, add `--max-epochs 2 --limit-train-batches 2 --limit-val-batches 1`. Integer limits count batches; fractional limits such as `0.5` select a proportion, and `1.0` uses all batches. Remove those limits for a scientific run and record the seed and config with the output. Each run directory contains its resolved `config.yaml` and checkpoints.

To use the trait registry instead of dotlist overrides, add `--traits-config ablation/configs/traits.yaml --trait example`. Training still starts from the selected model YAML. `saved_vae_config` is optional and is only needed for `conditional_vae_overlay.yaml`, which adapts an existing VAE configuration; the standalone `conditional_vae.yaml` trains the cVAE from scratch. Existing-checkpoint DDPM label controls separately require `saved_ddpm_config` in a local trait registry and are not part of the from-scratch neural baseline workflow.

## Select, generate and evaluate

The checkpoint selector scores generated PRS models on the real validation split and records the chosen filename in `selected_checkpoint.txt`. It does not use the independent test split for selection.

```bash
PYTHONPATH=. python -m ablation.select_generator_checkpoint \
  --checkpoint-dir /path/to/runs/crbm --models prs
```

Generate complete-size and balanced augmented synthetic cohorts. The default complete cohort uses training prevalence for labels.

```bash
PYTHONPATH=. python -m ablation.run_generate \
  --checkpoint-dir /path/to/runs/crbm --modes complete augmented
```

Run fold-paired cross-validation against real held-out folds. `--models prs` limits fitting to PRS, while the environment still needs all evaluation backend packages listed above.

```bash
PYTHONPATH=. python -m ablation.evaluation.run_evaluate \
  --checkpoint-dir /path/to/runs/crbm \
  --syn-dataset-types complete augmented --models prs --cpu
```

Run synthetic-cohort privacy diagnostics using the dataset and split configuration saved in the run:

```bash
PYTHONPATH=. python -m ablation.evaluation.run_privacy \
  --checkpoint-dir /path/to/runs/crbm --device cpu
```

For cVAE only, `ablation.run_reconstruct` can also write train/validation and test reconstructions; pass `--splits train_val test`. Evaluate them by adding `reconstructed` to `--syn-dataset-types`. Reconstruction retention and de novo synthetic-cohort privacy are different analyses and should be reported separately.

Association calibration and genotype-structure diagnostics accept a reference config and one or more generated cohorts directly:

```bash
PYTHONPATH=. python -m ablation.evaluation.run_association \
  --trait example --reference-config /path/to/runs/crbm/config.yaml \
  --cohort crbm=/path/to/runs/crbm/syn_complete_dataset.hdf5 \
  --output-dir /path/to/results/association
PYTHONPATH=. python -m ablation.evaluation.run_structure \
  --trait example --reference-config /path/to/runs/crbm/config.yaml \
  --cohort crbm=/path/to/runs/crbm/syn_complete_dataset.hdf5 \
  --output-dir /path/to/results/structure
```

The structure evaluator reports allele-frequency, pairwise-LD, LD-decay, PCA, HWE and rare-locus diagnostics. Real and synthetic inputs must use the same variant order; the reference HDF5 supplies per-variant SNP identifiers, chromosomes and positions. Dense pairwise-LD matrices require memory quadratic in the number of SNPs.

## Model scope

- `conditional_vae` adds phenotype conditioning to the VAE encoder and decoder. The standalone YAML is a complete from-scratch configuration; the overlay is an optional way to inherit architecture and loss settings from a user's existing VAE config.
- `conditional_crbm` implements a two-stage centered binary RBM cascade with deterministic cumulative dosage encoding (`0 -> 00`, `1 -> 10`, `2 -> 11`). Random pseudo-phasing remains an optional sensitivity setting.
- `conditional_wgan_gp` implements the multiscale convolutional WGAN-GP with class-dependent offsets. Its padded fake tails are masked to match zero-padded real inputs. `condition_on_label=false` provides an unconditional model control.

These are phenotype-conditioned project adaptations of Yelmen-inspired CRBM and WGAN-GP designs. Do not attribute these task-specific conditioning choices or their results to the published implementations. Shuffling labels on an already trained conditional DDPM is a label-coupling control; it is not equivalent to training a true unconditional DDPM.

## Local run manifests and internal migration tools

The exact paths and job IDs in the following working manifests are machine- and run-specific and are excluded from a release:

- `configs/cfg_association_runs.tsv`
- `configs/cross_model_structure_runs.tsv`
- `configs/ddpm_runs.tsv`
- `configs/reselection_runs.tsv`

`archive_test_selected_outputs.py` migrates old test-selected outputs from existing runs and is internal maintenance tooling. It is not required to train, select or evaluate a fresh model. The privacy re-audit checker and its exact author-run manifest are internal audit material; the reusable privacy evaluator above is the general entrypoint.

CFG association and cross-model structure analyses use explicit manifests with the schemas in `configs/cfg_association_runs.example.tsv` and `configs/cross_model_structure_runs.example.tsv`. Replace each example path with the corresponding reference config, synthetic cohorts and output directory before running the matching Slurm wrapper. Pass the manifest explicitly:

```bash
ablation/slurm/submit_cfg_association.sh \
  --manifest ablation/configs/cfg_association_runs.tsv --dry-run
ablation/slurm/submit_cross_model_structure.sh \
  --manifest ablation/configs/cross_model_structure_runs.tsv --dry-run
```

The exact `ddpm_runs.tsv` and `reselection_runs.tsv` manifests describe existing-checkpoint maintenance workflows and are excluded from a clean from-scratch release.
