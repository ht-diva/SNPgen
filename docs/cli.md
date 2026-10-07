# Command-line workflows

Run from the repository root after installing `requirements.txt`. Each script
accepts `--help`, which lists arguments and defaults without importing scientific
dependencies. See the [README](../README.md) for example commands.

| Script | Purpose |
|---|---|
| `bed_to_hdf5.py` | Prepare genotype datasets from PLINK BED, phenotypes and GWAS statistics |
| `train_vae.py` | Train the genotype VAE |
| `train_ddpm.py` | Train phenotype-conditioned diffusion using a trained VAE |
| `generate_ddpm.py` | Generate matched and augmented genotype cohorts |
| `reconstruct_vae.py` | Reconstruct genotype records with the VAE |
| `evaluate_downstream.py` | Evaluate predictive utility on independent real test records |
| `evaluate_privacy.py` | Evaluate disclosure risk and genotype fidelity |

## BED-to-HDF5 inputs

Input-format examples are included under **Run Settings** in `bed_to_hdf5.py`.
The script uses `genotype_handler` for preprocessing and `bed_conversion` for
phenotype, GWAS and metadata handling.

- `--bed-file` (or `--bed-prefix`) identifies a PLINK dataset with matching `.bim`
  and `.fam` files alongside it. Variants should already be LD-clumped and sorted
  by chromosome and physical position.
- `--phenotype-file` (`--phenotype`) supplies a CSV with `f.eid` (numeric
  participant ID) and `phenotype` (binary 0/1).
- `--gwas-file` (`--gwas`) supplies tab-delimited statistics. Default columns are
  `markername`, `effect_allele`, `noneffect_allele`, `beta` and `p_dgc`; column
  names are configurable.
- `--ancestry-file` optionally supplies a TSV with `f.eid` and an ethnicity field.
  `--ethnicity-coding-file` supplies hierarchical codes with `coding`, `meaning`
  and `parent_id`. `--ethnicity-field` selects the field and `--desired-ethnicity`
  specifies retained codes, including their immediate subgroups. Omit ancestry
  inputs to keep all phenotype records; missing ancestry is stored as -1.

Preprocessing maps genotypes, filters non-biallelic and singleton loci, imputes
missing calls with NumPy binomial draws, joins phenotypes and aligns GWAS effects
to the counted dosage allele. `--seed` controls preprocessing randomness.
`--top-k` selects the smallest GWAS p-values while retaining genomic order;
zero saves only the full retained panel.

`--output-dir`, `--output-name` and `--output-suffix` configure output naming.
With `--output cohort.hdf5`, all retained SNPs are saved there and top-K SNPs are
saved in `cohort_top2048.hdf5` when K is 2048. `--top-output` and `--snp-ids-output`
override the selected-panel and SNP-list paths. HDF5 files include dosages,
labels, participant IDs, ancestry and variant metadata. Use `--overwrite` to
replace existing outputs.

## Training

Training proceeds through run settings, configuration loading, dataset
construction, model construction and training. Architecture, discriminator,
trait and run naming are configurable. `--dataset-path` selects an explicit
HDF5 input; otherwise data are located under `data/ukb_<trait>` relative to
`--data-root`. `--output-dir` selects the run directory; the default is under
`checkpoints/<trait>`.

Repeat `--config FILE` to merge YAML files in order, replacing the automatically
selected list. Repeat `--set KEY=VALUE` to override nested configuration values.
The VAE otherwise combines base, encoder-size and discriminator configurations.
DDPM inherits its first-stage architecture and data configuration from the VAE
run. Its denoiser RNG seed is independent of the saved VAE training-split seed.
VAE training uses its top-level `seed` for data splitting and records that same
value in both the dataset configuration and `dataset_split_seed`.

Device count, CPU/GPU selection, precision, nodes, strategy, batch sizes,
epoch/step limits and accumulation are configurable. Worker count defaults to
the Slurm allocation, or zero outside Slurm. Learning-rate scaling uses GPU
count, node count and batch size; the device factor is one on CPU. W&B logging
is optional (`--wandb`).

For Lightning batch limits, integer `--limit-train-batches 1` means one batch;
`1.0` means all batches. Default epoch budgets are 400 for VAE and 500 for DDPM.
VAE checkpoints default to validation reconstruction accuracy; DDPM checkpoints
use validation denoising loss. Monitoring criteria are configurable. Each run
saves its effective resolved `config.yaml` beside its checkpoints.

To resume, provide `--resume-from CHECKPOINT`, the saved configuration via
`--config` and the run directory via `--output-dir`, then increase `--max-epochs`.
Learning-rate scaling is not reapplied on resume. Use `--save-last` during
training to save `last.ckpt`. DDPM also needs its associated VAE checkpoint/config.

## Generation and reconstruction

Both scripts require `--checkpoint`. Optional `--config` and `--dataset-path`
override configuration and data locations. Device, batch size, worker count and
inference RNG seed are configurable. Real-data partitions use the saved
training-split seed.

`generate_ddpm.py --modes matched augmented` creates a cohort with the reference
phenotype distribution and a balanced cohort, respectively. Filenames are
`syn_complete_dataset.hdf5` and `syn_augmented_dataset.hdf5`. The `syn_recon` mode
reconstructs validation-source latent representations for diagnostics.
`--cfg-scale` selects one guidance strength; `--cfg-scales 1 2 3 5` repeats
sampling into per-scale directories. `--sampling-steps` overrides the sampler
budget and `--no-ema` selects online weights. Loading restores the full DDPM
checkpoint, including the first stage and phenotype conditioner, so relocated
generation runs need no external VAE initialization file.

`reconstruct_vae.py` defaults to training/validation and test reconstructions.
`--splits` selects source partitions. Posterior sampling is the default;
`--posterior-mean` uses the encoder mean. `--store-latents` and `--store-originals`
include diagnostic arrays. Targets and participant IDs retain source ordering.
Existing generation or reconstruction files require `--skip-existing` or
`--overwrite`.

## Downstream evaluation

Each classifier fits on all but one fitting fold and scores the paired fold of
the independent real test set. Real, matched, augmented and reconstructed data
use the project classifier/PRS trainers and incremental result utilities.
Models, fold count, retraining controls, RNG seed, device and thread count are
configurable.

Use `--complete-path`, `--augmented-path`, `--reconstructed-path` and
`--synthetic-types`, or locate outputs via `--checkpoint-dir`. `--output-dir`
places real and synthetic results in separate subdirectories. Results are
saved as pickles; `--plot` creates the comparison PNG and CSV. Keep classifier
settings and input cohorts consistent when reusing results.

Fitting reconstructions must come from training or validation records.
Test/full reconstruction sources are rejected, and stored participant IDs are
checked for overlap with the real holdout.

## Privacy evaluation

Use `--checkpoint-dir` for automatic loading or `--config` with explicit paths
to relocate inputs. `--traits-config FILE` supports multi-trait analysis through
a YAML mapping. Entries accept `ddpm_checkpoint` and, for manual loading,
`manual`, `dataset_path`, `vae_checkpoint`, `seed`, `val_ratio` and `test_ratio`.
Relative paths use `--checkpoint-base`, or the trait YAML's directory.

Gradient-training records are membership candidates, the independent real test
set is the holdout, and training/validation records provide the fidelity
reference. Saved split metadata takes precedence over the DDPM RNG seed.
Distances, class-specific evaluation, Yelmen-compatible tests, chains, sample
caps and nearest-neighbour batch size are configurable.

Reconstruction privacy runs separately when data are available;
`--no-reconstructed` disables it. `--output-dir` and `--recon-output-dir` redirect
results. Plots and the cross-trait summary are saved to `--plot-dir` or a default
privacy plot directory; `--no-plots` skips visualization.

## Slurm

Set `SNPGEN_PYTHON` or the documented conda environment variables, and configure
`SNPGEN_SBATCH_ARGS` for your site. The DDPM launcher accepts explicit checkpoints
and calls `generate_ddpm.py`:

```bash
export SNPGEN_PYTHON=/path/to/environment/bin/python
export SNPGEN_SBATCH_ARGS='--partition=your_gpu_partition --account=your_account'
bash ablation/slurm/submit_ddpm_evaluation.sh \
  --run trait /path/to/ddpm_run/best_checkpoint.ckpt --cfg-scale 5 --dry-run
```

Omit `--dry-run` to submit. See [the comparison suite](../ablation/README.md) for
additional generators, evaluations and scheduler options.
