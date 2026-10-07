#!/usr/bin/env python3
"""Run downstream prediction with paired-fold cross-validation.

Configuration values come from command-line arguments.
"""

import argparse
import os
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description='Evaluate downstream utility with the paired classifier folds and an independent real test set.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint-dir', help='Directory containing config.yaml and generated HDF5 files.')
    parser.add_argument('--config', help='Explicit config.yaml; required if no checkpoint directory is supplied.')
    parser.add_argument('--dataset-path', help='Override the saved real-data path.')
    parser.add_argument('--synthetic-types', nargs='+', choices=['complete','augmented','reconstructed'], default=['complete','augmented','reconstructed'])
    parser.add_argument('--complete-path')
    parser.add_argument('--augmented-path')
    parser.add_argument('--reconstructed-path')
    parser.add_argument('--reconstruction-split', choices=['train','val','train_val'], default='train_val', help='Source split of fitting reconstructions; test/full are prohibited.')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--models', nargs='+', choices=['xgboost','xgboost_balanced','catboost','prs','random_forest'], default=['xgboost','xgboost_balanced','catboost','prs'])
    parser.add_argument('--force-retrain', nargs='*', default=[])
    parser.add_argument('--skip-real-training', action='store_true')
    parser.add_argument('--seed', type=int, default=42, help='Classifier/fold seed; data split comes from saved training config.')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--threads', type=int, help='CPU thread count for BLAS/OpenMP and PyTorch.')
    parser.add_argument('--output-dir', help='Optional output root with real/<type> subdirectories; default standard paths.')
    parser.add_argument('--real-output-dir')
    parser.add_argument('--plot', action='store_true', help='Plot the comparison and save its image/CSV.')
    parser.add_argument('--plot-output')
    parser.add_argument('--plot-models', nargs='*', default=['xgboost','xgboost_balanced','prs univariate scaled (threshold 0.5)'])
    return parser

def reject_test_reconstructions(path, split, raw_dataset):
    """Guard against fitting on reconstructions of held-out real test records."""
    import h5py
    import numpy as np
    with h5py.File(path, 'r') as handle:
        source = handle.attrs.get('split', split)
        if isinstance(source, bytes):
            source = source.decode()
        if source in ['test', 'full']:
            raise ValueError('Test/full reconstructions overlap the independent test set and cannot be fitting data')
        _, _, test_metadata = raw_dataset.get_split('test', metadata=True)
        if test_metadata and 'eids' in test_metadata and 'eids' in handle:
            overlap = np.intersect1d(np.asarray(test_metadata['eids']), handle['eids'][:])
            if overlap.size:
                raise ValueError('Reconstruction participant IDs overlap the independent real test set')


def run(args):
    import os
    if not args.config and not args.checkpoint_dir:
        raise ValueError('Provide --config or --checkpoint-dir')
    if args.n_folds < 2:
        raise ValueError('--n-folds must be at least 2')
    if args.threads:
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads)
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    import random
    import numpy as np
    import pandas as pd
    import os
    from omegaconf import OmegaConf

    # SNPgen imports
    from snpgen.data.loader import SplitDataset
    from snpgen.utils import instantiate_from_config

    # Evaluation module imports
    from snpgen.evaluation import (
        # Pipeline
        train_models,
        # Results handling
        build_multiindex_df,
        save_cv_results,
        load_cv_results,
        check_results_exist,
        compute_gwas_prs_results,  # For GWAS PRS computation
        # Incremental CV training utilities
        verify_cv_indices,
        get_missing_trainers,
        merge_fold_results,
        # Analysis
        filter_model_list,
        # Plotting
        plot_metrics_with_ci,
        # Cross-validation utilities (supports binary classification)
        get_stratified_kfold,
    )

    # ============================================================
    # RUN SETTINGS — supplied by CLI arguments
    # ============================================================

    # Path to your trained DDPM checkpoint directory.
    # This directory must contain config.yaml and the generated .hdf5 files.
    checkpoint_dir = args.checkpoint_dir or os.path.dirname(os.path.abspath(args.config))

    # Choose the synthetic dataset types to evaluate
    syn_dataset_types = args.synthetic_types

    # For 'reconstructed' mode only: which split to use for VAE reconstructions
    # Note: The reconstructed data will still be tested on the REAL test set
    reconstruction_split = args.reconstruction_split

    # Cross-validation settings
    n_folds = args.n_folds

    # Models to train
    default_models = args.models

    # Optional: force retrain specific models even if results already exist
    # Set to [] to skip. Use 'prs_gwas' to also force retrain GWAS PRS
    force_retrain = args.force_retrain

    # Set to True to skip training on real data and only train on synthetic datasets
    skip_real_training = args.skip_real_training

    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)

    NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    NUM_NODES = int(os.environ.get("SLURM_NNODES", 1))
    ALLOCATED_GPUS_PER_NODE = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))
    SLURM_JOBID = os.environ.get("SLURM_JOB_ID", "local")

    num_gpus = torch.cuda.device_count()
    print(f"{num_gpus} GPU(s) available")
    print(f"Using {NUM_WORKERS} workers for the DataLoader")

    # Load Datasets

    config_path = args.config or os.path.join(checkpoint_dir, 'config.yaml')

    # =============================================================================
    # Load Config and Real Dataset (once)
    # =============================================================================

    assert os.path.exists(config_path), f"config.yaml not found at {config_path}. This command requires a run with saved config.yaml."

    print(f"Loading config from checkpoint directory: {config_path}")
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    config = OmegaConf.load(config_path)
    if args.dataset_path:
        config.dataset_path = args.dataset_path

    h5_path = config['dataset_path']
    print(f"Using dataset path from reloaded config: {h5_path}")
    proj_name = os.path.basename(os.path.dirname(config['dataset_path'])).replace('ukb_', '')
    print(f"Inferred project name: {proj_name}")

    print(f"\nLoading Real Dataset from: {h5_path}")
    split_seed = int(config.get('dataset_split_seed', config.data.raw_dataset.params.get('seed', config.seed)))
    if split_seed != seed:
        print(f"Overriding raw_dataset seed from {seed} to {split_seed}")
    raw_dataset = instantiate_from_config(config.data.raw_dataset, file_path=h5_path, seed=split_seed, onehot=False, data_dtype=None, metadata=True)

    # For reconstructed mode, auto-detect VAE checkpoint directory from DDPM config
    vae_checkpoint_dir = None
    if 'reconstructed' in syn_dataset_types and not args.reconstructed_path:
        vae_ckpt_path = config.model.params.first_stage_config.params.get('ckpt_path', None)
        assert vae_ckpt_path is not None, (
            "'reconstructed' mode requires a VAE checkpoint path in the DDPM config "
            "(model.params.first_stage_config.params.ckpt_path)"
        )
        vae_checkpoint_dir = os.path.dirname(vae_ckpt_path)
        print(f"VAE checkpoint directory (from DDPM config): {vae_checkpoint_dir}")

    # =============================================================================
    # Determine ml_models directory (always binary classification)
    # =============================================================================

    ml_models_dirname = 'ml_models'
    ml_models_path = args.real_output_dir or (os.path.join(args.output_dir, 'real') if args.output_dir else os.path.join(os.path.dirname(h5_path), ml_models_dirname, 'cv'))
    os.makedirs(ml_models_path, exist_ok=True)

    print(f"\nBinary classification task")
    print(f"\nReal models output path: {ml_models_path}")

    # =============================================================================
    # Load Synthetic Datasets for each type
    # =============================================================================

    syn_datasets = {}

    for syn_dataset_type in syn_dataset_types:
        print(f"\n{'='*60}")
        print(f"Loading {syn_dataset_type.upper()} Dataset")
        print(f"{'='*60}")

        assert syn_dataset_type in ['complete', 'augmented', 'reconstructed'], f"Invalid synthetic dataset type: {syn_dataset_type}"

        # Define synthetic dataset path based on type
        if syn_dataset_type == 'complete':
            h5_path_syn = args.complete_path or os.path.join(checkpoint_dir, 'syn_complete_dataset.hdf5')
            ml_models_subdir = 'cv'
        elif syn_dataset_type == 'augmented':
            h5_path_syn = args.augmented_path or os.path.join(checkpoint_dir, 'syn_augmented_dataset.hdf5')
            ml_models_subdir = 'cv_augmented'
        elif syn_dataset_type == 'reconstructed':
            split_suffix = f"_{reconstruction_split}" if reconstruction_split else ""
            h5_path_syn = args.reconstructed_path or os.path.join(vae_checkpoint_dir, f'vae_reconstruction_dataset{split_suffix}.hdf5')
            ml_models_subdir = f'cv_reconstructed_{reconstruction_split}'

        # Check if dataset file exists
        if not os.path.exists(h5_path_syn):
            raise FileNotFoundError(f"Requested {syn_dataset_type} dataset not found: {h5_path_syn}")

        print(f"Path: {h5_path_syn}")

        # Reconstructions containing independent-test records cannot be fitting data.
        if syn_dataset_type == 'reconstructed':
            reject_test_reconstructions(h5_path_syn, reconstruction_split, raw_dataset)
        # Load synthetic dataset
        raw_dataset_syn = instantiate_from_config(
            config.data.raw_dataset,
            file_path=h5_path_syn,
            seed=split_seed,
            onehot=False,
            data_dtype=None,
            metadata=True,
            x_key='syn_samples',
            y_key='targets',
        )

        # Determine output path
        if args.output_dir:
            ml_models_path_syn = os.path.join(args.output_dir, syn_dataset_type)
        elif syn_dataset_type == 'reconstructed':
            ml_models_path_syn = os.path.join(vae_checkpoint_dir, ml_models_dirname, ml_models_subdir)
        else:
            ml_models_path_syn = os.path.join(checkpoint_dir, 'ml_models', ml_models_subdir)
        os.makedirs(ml_models_path_syn, exist_ok=True)

        print(f"  Output path: {ml_models_path_syn}")

        syn_datasets[syn_dataset_type] = {
            'h5_path': h5_path_syn,
            'dataset': raw_dataset_syn,
            'ml_models_path': ml_models_path_syn,
        }

    print(f"\n{'='*60}")
    print(f"Successfully loaded {len(syn_datasets)} synthetic dataset(s): {list(syn_datasets.keys())}")
    print(f"{'='*60}")

    # CV Pipeline

    #
    #
    #         FULL DATASET (real)
    #
    #
    #
    #             TRAINING SET (real) (used to train the generative model)
    #
    #
    #             TEST SET (real)
    #
    #
    #
    #
    #
    #
    #         FULL DATASET (syn)
    #
    #
    #
    #             TRAINING SET (syn)
    #
    #
    #             TEST SET (unused)
    #
    #
    #
    #
    # The splits in each iteration will be the following (example with 5 folds):
    #
    #
    #
    #
    #
    #
    # |   |
    # |---------|
    # | **Iteration 1** |
    # | **Iteration 2** |
    # | **Iteration 3** |
    # | **Iteration 4** |
    # | **Iteration 5** |
    #
    #
    #
    #
    # TRAINING SET (real)
    #
    # | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
    # |--------|--------|--------|--------|--------|
    # | **Unused** | Train  | Train  | Train  | Train  |
    # | Train   | **Unused** | Train  | Train  | Train  |
    # | Train   | Train   | **Unused** | Train  | Train  |
    # | Train   | Train   | Train  | **Unused** | Unused  |
    # | Train   | Train   | Train  | Train  | **Test** |
    #
    #
    #
    #
    # TRAINING SET (syn)
    #
    # | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
    # |--------|--------|--------|--------|--------|
    # | **Unused** | Train  | Train  | Train  | Train  |
    # | Train   | **Unused** | Train  | Train  | Train  |
    # | Train   | Train   | **Unused** | Train  | Train  |
    # | Train   | Train   | Train  | **Unused** | Unused  |
    # | Train   | Train   | Train  | Train  | **Test** |
    #
    #
    #
    #
    # TEST SET (real)
    #
    # | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
    # |--------|--------|--------|--------|--------|
    # | **Test** | Unused  | Unused  | Unused  | Unused  |
    # | Unused   | **Test** | Unused  | Unused  | Unused  |
    # | Unused   | Unused   | **Test** | Unused  | Unused  |
    # | Unused   | Unused   | Unused  | **Test** | Unused  |
    # | Unused   | Unused   | Unused  | Unused  | **Test** |
    #
    #
    #
    #
    #
    # So, for instance, in the first iteration the ML models will be trained on Fold2+Fold3+Fold4+Fold5 of the Real Training Set and will be also trained on Fold2+Fold3+Fold4+Fold5 of the Syn Training Set. Then, in both cases they will be tested on Fold1 of the Real Test Set. \
    # This is not the usual KFold configuration, but it is a reasonable way to have a confidence interval for the ML models in both the Real and Syn scenarios without using as a test set parts of the data used to train the generative model.

    # Get real data splits
    real_split_data, real_split_labels, real_split_metadata = raw_dataset.get_split('train_val', metadata=True)
    test_split_data, test_split_labels, test_split_metadata = raw_dataset.get_split('test', metadata=True)

    # Stratified k-fold for binary classification
    skf_real = get_stratified_kfold(real_split_labels, n_splits=n_folds, shuffle=True, random_state=seed)
    skf_test = get_stratified_kfold(test_split_labels, n_splits=n_folds, shuffle=True, random_state=seed)

    # Check if GWAS betas are available for PRS computation
    gwas_betas = raw_dataset.metadata.get('betas', None) if raw_dataset.metadata else None
    has_gwas_betas = gwas_betas is not None
    if has_gwas_betas:
        print(f"GWAS betas available: shape {gwas_betas.shape}")
    else:
        print("No GWAS betas found in metadata - skipping GWAS PRS")

    # =============================================================================
    # REAL DATA TRAINING (incremental - only trains missing models)
    # =============================================================================

    # Load existing results if available
    existing_results_real = None
    if check_results_exist(ml_models_path):
        existing_results_real = load_cv_results(ml_models_path)
        print(f"\n{'='*60}")
        print("REAL DATA: Loaded existing results")
        print(f"  Path: {ml_models_path}")
        first_fold = list(existing_results_real.keys())[0]
        existing_model_names = list(existing_results_real[first_fold].get('models', {}).keys())
        print(f"  Existing models in {first_fold}: {existing_model_names}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("REAL DATA: No existing results found. Training all models...")
        print(f"{'='*60}")

    _cv_results_real = existing_results_real if existing_results_real is not None else {}
    any_real_trained = False

    if not skip_real_training:
        for i, ((train_index, _), (_, test_index)) in enumerate(zip(
                skf_real.split(real_split_data, real_split_labels),
                skf_test.split(test_split_data, test_split_labels)
            )):

            fold_key = f'fold_{i}'

            X_train, y_train = real_split_data[train_index], real_split_labels[train_index]
            X_test, y_test = test_split_data[test_index], test_split_labels[test_index]

            fold_metadata = {'train_index': train_index, 'test_index': test_index}

            # Verify indices match if results exist
            if existing_results_real is not None:
                verify_cv_indices(_cv_results_real, fold_key, train_index, test_index)

            # Determine which models are missing
            missing_trainers = get_missing_trainers(_cv_results_real, fold_key, default_models, force_retrain=force_retrain)
            need_gwas = has_gwas_betas and (
                not any(k.startswith('prs gwas') for k in _cv_results_real.get(fold_key, {}).get('models', {}))
                or (force_retrain and 'prs_gwas' in force_retrain)
            )

            if not missing_trainers and not need_gwas:
                print(f"\n*** Fold {i}: All models already exist, skipping ***")
                continue

            any_real_trained = True
            print(f"\n*** Fold {i}: ***")

            if missing_trainers:
                print(f"  Missing trainers: {missing_trainers}")
                print(f"\n++ Training on Real (models: {missing_trainers}) ++")
                real_results = train_models(
                    X_train, y_train, X_test, y_test,
                    metrics_dict=None,
                    save_path=os.path.join(ml_models_path, fold_key),
                    on_gpu=device.type == 'cuda', seed=seed,
                    models=missing_trainers,
                )
                merge_fold_results(_cv_results_real, real_results, fold_key, fold_metadata)
                save_cv_results(_cv_results_real, ml_models_path)  # Save after each fold to ensure progress is not lost

            # Add GWAS PRS if betas available and not already present
            if need_gwas:
                print("\n++ Computing GWAS PRS (fixed betas) ++")
                gwas_prs_results = compute_gwas_prs_results(
                    gwas_betas, X_test, y_test, model_name='prs gwas', verbose=True,
                )
                merge_fold_results(_cv_results_real, gwas_prs_results, fold_key, fold_metadata)

        if any_real_trained:
            print("\nSaving updated real data results...")
            save_cv_results(_cv_results_real, ml_models_path)
            print(f"Real data results saved to: {ml_models_path}")
        else:
            print("\nAll real data models already trained. No changes needed.")

    # =============================================================================
    # SYNTHETIC DATA TRAINING (incremental - only trains missing models)
    # =============================================================================

    all_cv_results_syn = {}

    for syn_dataset_type, syn_data in syn_datasets.items():
        print(f"\n{'='*60}")
        print(f"{syn_dataset_type.upper()} DATA: Starting CV Pipeline")
        print(f"{'='*60}")

        raw_dataset_syn = syn_data['dataset']
        ml_models_path_syn = syn_data['ml_models_path']

        # Load existing results if available
        existing_results_syn = None
        if check_results_exist(ml_models_path_syn):
            existing_results_syn = load_cv_results(ml_models_path_syn)
            first_fold = list(existing_results_syn.keys())[0]
            existing_model_names = list(existing_results_syn[first_fold].get('models', {}).keys())
            print(f"  Loaded existing results from: {ml_models_path_syn}")
            print(f"  Existing models in {first_fold}: {existing_model_names}")
        else:
            print(f"  No existing results. Training all models...")

        _cv_results_syn = existing_results_syn if existing_results_syn is not None else {}

        if raw_dataset_syn.data.shape[1] != raw_dataset.data.shape[1]:
            raise ValueError('Real and generated cohorts must contain the same SNP panel width')
        # Get synthetic split
        if syn_dataset_type == 'reconstructed':
            syn_split_data, syn_split_labels, syn_split_metadata = raw_dataset_syn.get_split('full', metadata=True)
            print(f"  Using 'full' split for reconstructed data (already contains {reconstruction_split} split)")
        else:
            syn_split_data, syn_split_labels, syn_split_metadata = raw_dataset_syn.get_split('train_val', metadata=True)

        skf_syn = get_stratified_kfold(syn_split_labels, n_splits=n_folds, shuffle=True, random_state=seed)

        # Check if any training is needed across all folds
        any_syn_trained = False

        for i, ((train_syn_index, _), (_, test_index)) in enumerate(zip(
                skf_syn.split(syn_split_data, syn_split_labels),
                skf_test.split(test_split_data, test_split_labels)
            )):

            fold_key = f'fold_{i}'

            X_train_syn, y_train_syn = syn_split_data[train_syn_index], syn_split_labels[train_syn_index]
            X_test, y_test = test_split_data[test_index], test_split_labels[test_index]

            fold_metadata = {'train_index': train_syn_index, 'test_index': test_index}

            # Verify indices match if results exist
            if existing_results_syn is not None:
                verify_cv_indices(_cv_results_syn, fold_key, train_syn_index, test_index)

            # Determine which models are missing
            missing_trainers = get_missing_trainers(_cv_results_syn, fold_key, default_models, force_retrain=force_retrain)

            if not missing_trainers:
                print(f"\n*** Fold {i}: All models already exist, skipping ***")
                continue

            any_syn_trained = True
            print(f"\n*** Fold {i}: ***")
            print(f"  Missing trainers: {missing_trainers}")

            print(f"\n++ Training on {syn_dataset_type.capitalize()} (models: {missing_trainers}) ++")
            syn_results = train_models(
                X_train_syn, y_train_syn, X_test, y_test,
                metrics_dict=None,
                save_path=os.path.join(ml_models_path_syn, fold_key),
                on_gpu=device.type == 'cuda', seed=seed,
                models=missing_trainers,
            )
            merge_fold_results(_cv_results_syn, syn_results, fold_key, fold_metadata)
            save_cv_results(_cv_results_syn, ml_models_path_syn)  # Save after each fold to ensure progress is not lost

        if any_syn_trained:
            print(f"\nSaving updated {syn_dataset_type} data results...")
            save_cv_results(_cv_results_syn, ml_models_path_syn)
            print(f"{syn_dataset_type.capitalize()} data results saved to: {ml_models_path_syn}")
        else:
            print(f"\nAll {syn_dataset_type} models already trained. No changes needed.")

        all_cv_results_syn[syn_dataset_type] = _cv_results_syn

    print("\n\n" + "="*60)
    print("Cross-Validation Training Complete")
    print("="*60)
    print(f"\nReal data results: {ml_models_path}")
    for syn_type, syn_data in syn_datasets.items():
        print(f"{syn_type.capitalize()} data results: {syn_data['ml_models_path']}")

    # Plots

    if not args.plot:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    # =============================================================================
    # PLOTTING with AUTO-DETECTION
    # =============================================================================
    # The plot_metrics_with_ci function automatically detects whether the data
    # is from a classification (binary) task and selects appropriate metrics:
    #   - Classification (binary): ['balanced_accuracy', 'roc_auc']
    #
    # You can still override this behavior by specifying metrics_to_plot explicitly.
    # =============================================================================

    load_scaled_results = False

    # Models to include in the plot (leave empty for all models)
    # NOTE: GWAS PRS is shown as horizontal reference line, not as bars
    models_to_keep = args.plot_models

    base_paths = {
        ml_models_path: {
            'pretty_name': 'Real', 'add_prs': False,
            'add_univariate_prs': True, 'color': '#D4EAF2',
        },
    }
    plot_styles = {
        'reconstructed': ('Reconstructed', '#4A90E2'),
        'complete': ('Syn', '#FB9270'),
        'augmented': ('Syn Augmented', '#9C2A20'),
    }
    for dataset_type in ['reconstructed', 'complete', 'augmented']:
        if dataset_type in syn_datasets:
            name, color = plot_styles[dataset_type]
            base_paths[syn_datasets[dataset_type]['ml_models_path']] = {
                'pretty_name': name, 'add_prs': False,
                'add_univariate_prs': True, 'color': color,
            }

    # Optional: Override model display names
    model_pretty_names = {
        'random_forest': 'Random\nForest',
        'catboost': 'CatBoost',
        'xgboost': 'XGBoost',
        'xgboost_balanced': 'XGBoost\n(Balanced)',
        'knn': 'kNN',
        'prs scaled (threshold 0.5)': 'PRS\n(scaled [0-1])',
        'prs univariate scaled (threshold 0.5)': 'PRS Univ\n(scaled [0-1])',
        'prs univariate': 'PRS Univ',
    }

    filename = 'results_scaled.pkl' if load_scaled_results else 'results.pkl'

    results_df = []
    colors = []

    gwas_prs_df = None  # To store GWAS PRS results from real data

    for path, properties in base_paths.items():
        r = load_cv_results(path, filename)
        df = build_multiindex_df(r)

        if 'real' in properties['pretty_name'].lower():
            gwas_prs_df = df.copy()  # Update GWAS PRS df from real data if available

        # Filter models if specified
        if len(models_to_keep) > 0:
            available_models = df.index.get_level_values('model').unique().to_list()
            models_in_df = [m for m in models_to_keep if m in available_models]
            if models_in_df:
                df = df.loc[:, models_in_df, :]

        model_keys = filter_model_list(
            df.index.get_level_values('model').unique().to_list(),
            include_prs=properties.get('add_prs', False),
            include_prs_univariate=properties.get('add_univariate_prs', False),
            prs_to_include=['prs scaled (threshold 0.5)'],
            prs_univariate_to_include=['prs univariate scaled (threshold 0.5)', 'prs univariate'],
        )
        df = df.loc[:, model_keys, :]

        colors.append(properties.get('color', 'black'))
        df = df.assign(name=properties['pretty_name'])
        results_df.append(df)

    combined_df = pd.concat(results_df)
    combined_df = combined_df.set_index('name', append=True).reorder_levels(['name', 'fold', 'model'])

    # Extract trait name from Real data path for plot title
    real_path = [path for path, props in base_paths.items() if props.get('pretty_name') == 'Real'][0]
    if 'ukb_' in real_path:
        trait_name = real_path.split('ukb_')[1].split('/')[0].replace('_', ' ').title()
    else:
        trait_name = None

    # AUTO-DETECTION: No need to specify metrics_to_plot or metric_pretty_names
    # The function will detect the task type and use appropriate defaults
    fig = plot_metrics_with_ci(
        combined_df,
        model_pretty_names=model_pretty_names,
        colors=colors,
        confidence=0.95,
        joint=True,
        orientation='vertical',
        base_height=4,
        base_width=8,
        show_prs_baseline=True,  # Show univariate PRS as horizontal reference line
        show_gwas_prs_baseline=True,  # Show GWAS PRS as horizontal reference line
        gwas_prs_df=gwas_prs_df,  # Pass full df - function selects the right model
        title=trait_name,  # Add trait name as title
        # task_type='auto',  # Default - auto-detects from data
        # metrics_to_plot=['roc_auc'],
        output_path=args.plot_output or os.path.join(args.output_dir or checkpoint_dir, 'cv_comparison.png'),  # Saves the figure and .csv
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
