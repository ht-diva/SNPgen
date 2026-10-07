"""Run paper-style cross-validation evaluation for one ablation checkpoint."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from snpgen.evaluation import (
    check_results_exist,
    compute_gwas_prs_results,
    get_missing_trainers,
    get_stratified_kfold,
    load_cv_results,
    merge_fold_results,
    save_cv_results,
    train_models,
    verify_cv_indices,
)
from ablation.utils import resolve_checkpoint_dir_with_config
from snpgen.utils import instantiate_from_config

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_raw_dataset(config, h5_path: str, seed: int, x_key: str = "data", y_key: str = "labels", metadata=True):
    return instantiate_from_config(
        config.data.raw_dataset,
        file_path=h5_path,
        seed=seed,
        onehot=False,
        data_dtype=None,
        metadata=metadata,
        x_key=x_key,
        y_key=y_key,
    )


def _train_real_baseline(
    raw_dataset,
    ml_models_path: str,
    models: list[str],
    force_retrain: list[str],
    n_folds: int,
    seed: int,
    on_gpu: bool,
    skip_real_training: bool,
) -> dict:
    real_split_data, real_split_labels, _ = raw_dataset.get_split("train_val", metadata=True)
    test_split_data, test_split_labels, _ = raw_dataset.get_split("test", metadata=True)
    skf_real = get_stratified_kfold(real_split_labels, n_splits=n_folds, shuffle=True, random_state=seed)
    skf_test = get_stratified_kfold(test_split_labels, n_splits=n_folds, shuffle=True, random_state=seed)

    gwas_betas = raw_dataset.metadata.get("betas", None) if raw_dataset.metadata else None
    has_gwas_betas = gwas_betas is not None
    print(f"GWAS betas available: {has_gwas_betas}")

    existing_results_real = load_cv_results(ml_models_path) if check_results_exist(ml_models_path) else None
    cv_results_real = existing_results_real if existing_results_real is not None else {}

    if skip_real_training:
        return cv_results_real

    os.makedirs(ml_models_path, exist_ok=True)
    any_real_trained = False
    for i, ((train_index, _), (_, test_index)) in enumerate(zip(
        skf_real.split(real_split_data, real_split_labels),
        skf_test.split(test_split_data, test_split_labels),
    )):
        fold_key = f"fold_{i}"
        X_train, y_train = real_split_data[train_index], real_split_labels[train_index]
        X_test, y_test = test_split_data[test_index], test_split_labels[test_index]
        fold_metadata = {"train_index": train_index, "test_index": test_index}

        if existing_results_real is not None:
            verify_cv_indices(cv_results_real, fold_key, train_index, test_index)

        missing_trainers = get_missing_trainers(cv_results_real, fold_key, models, force_retrain=force_retrain)
        need_gwas = has_gwas_betas and (
            not any(k.startswith("prs gwas") for k in cv_results_real.get(fold_key, {}).get("models", {}))
            or ("prs_gwas" in force_retrain)
        )
        if not missing_trainers and not need_gwas:
            print(f"\n*** Real fold {i}: all models already exist, skipping ***")
            continue

        any_real_trained = True
        if missing_trainers:
            print(f"\n++ Training on Real fold {i}: {missing_trainers} ++")
            real_results = train_models(
                X_train,
                y_train,
                X_test,
                y_test,
                metrics_dict=None,
                save_path=os.path.join(ml_models_path, fold_key),
                on_gpu=on_gpu,
                seed=seed,
                models=missing_trainers,
            )
            merge_fold_results(cv_results_real, real_results, fold_key, fold_metadata)
            save_cv_results(cv_results_real, ml_models_path)

        if need_gwas:
            print("\n++ Computing GWAS PRS (fixed betas) ++")
            gwas_prs_results = compute_gwas_prs_results(
                gwas_betas,
                X_test,
                y_test,
                model_name="prs gwas",
                verbose=True,
            )
            merge_fold_results(cv_results_real, gwas_prs_results, fold_key, fold_metadata)
            save_cv_results(cv_results_real, ml_models_path)

    if any_real_trained:
        save_cv_results(cv_results_real, ml_models_path)
    return cv_results_real


def _train_synthetic_dataset(
    raw_dataset_syn,
    raw_dataset_real,
    syn_dataset_type: str,
    ml_models_path_syn: str,
    models: list[str],
    force_retrain: list[str],
    n_folds: int,
    seed: int,
    on_gpu: bool,
    synthetic_label_strategy: str,
    synthetic_label_seed: int,
) -> dict:
    test_split_data, test_split_labels, _ = raw_dataset_real.get_split("test", metadata=True)
    if syn_dataset_type == "reconstructed":
        syn_split_data, syn_split_labels, _ = raw_dataset_syn.get_split("full", metadata=True)
    else:
        syn_split_data, syn_split_labels, _ = raw_dataset_syn.get_split("train_val", metadata=True)

    syn_split_labels = _synthetic_labels_for_strategy(
        labels=syn_split_labels,
        raw_dataset_real=raw_dataset_real,
        strategy=synthetic_label_strategy,
        seed=synthetic_label_seed,
    )

    skf_syn = get_stratified_kfold(syn_split_labels, n_splits=n_folds, shuffle=True, random_state=seed)
    skf_test = get_stratified_kfold(test_split_labels, n_splits=n_folds, shuffle=True, random_state=seed)

    existing_results_syn = load_cv_results(ml_models_path_syn) if check_results_exist(ml_models_path_syn) else None
    cv_results_syn = existing_results_syn if existing_results_syn is not None else {}

    os.makedirs(ml_models_path_syn, exist_ok=True)
    any_syn_trained = False
    for i, ((train_syn_index, _), (_, test_index)) in enumerate(zip(
        skf_syn.split(syn_split_data, syn_split_labels),
        skf_test.split(test_split_data, test_split_labels),
    )):
        fold_key = f"fold_{i}"
        X_train_syn, y_train_syn = syn_split_data[train_syn_index], syn_split_labels[train_syn_index]
        X_test, y_test = test_split_data[test_index], test_split_labels[test_index]
        fold_metadata = {"train_index": train_syn_index, "test_index": test_index}

        if existing_results_syn is not None:
            verify_cv_indices(cv_results_syn, fold_key, train_syn_index, test_index)

        missing_trainers = get_missing_trainers(cv_results_syn, fold_key, models, force_retrain=force_retrain)
        if not missing_trainers:
            print(f"\n*** {syn_dataset_type} fold {i}: all models already exist, skipping ***")
            continue

        any_syn_trained = True
        print(f"\n++ Training on {syn_dataset_type} fold {i}: {missing_trainers} ++")
        syn_results = train_models(
            X_train_syn,
            y_train_syn,
            X_test,
            y_test,
            metrics_dict=None,
            save_path=os.path.join(ml_models_path_syn, fold_key),
            on_gpu=on_gpu,
            seed=seed,
            models=missing_trainers,
        )
        merge_fold_results(cv_results_syn, syn_results, fold_key, fold_metadata)
        save_cv_results(cv_results_syn, ml_models_path_syn)

    if any_syn_trained:
        save_cv_results(cv_results_syn, ml_models_path_syn)
    return cv_results_syn


def _synthetic_labels_for_strategy(labels: np.ndarray, raw_dataset_real, strategy: str, seed: int) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    if strategy == "matched":
        return labels
    rng = np.random.default_rng(seed)
    if strategy == "permuted":
        return rng.permutation(labels)
    if strategy == "prevalence":
        _real_x, real_labels = raw_dataset_real.get_split("train_val", metadata=False)
        real_labels = np.asarray(real_labels).reshape(-1)
        classes, counts = np.unique(real_labels, return_counts=True)
        return rng.choice(classes, size=labels.shape[0], p=counts / counts.sum()).astype(real_labels.dtype)
    raise ValueError(f"Unknown synthetic label strategy: {strategy}")


def _label_strategy_out_dir(base_out: str, strategy: str, seed: int) -> str:
    if strategy == "matched":
        return base_out
    return f"{base_out}_label_{strategy}_seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--syn-dataset-types", nargs="+", default=["complete", "augmented"], choices=["complete", "augmented", "reconstructed"])
    parser.add_argument("--reconstruction-split", default="train_val")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["xgboost", "xgboost_balanced", "catboost", "prs"])
    parser.add_argument("--force-retrain", nargs="*", default=[])
    parser.add_argument("--skip-real-training", action="store_true")
    parser.add_argument("--cpu", dest="on_gpu", action="store_false")
    parser.add_argument("--gpu", dest="on_gpu", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-label-strategy", choices=["matched", "permuted", "prevalence"], default="matched")
    parser.add_argument("--synthetic-label-seed", type=int, default=42)
    args = parser.parse_args()
    args.checkpoint_dir = resolve_checkpoint_dir_with_config(args.checkpoint_dir)

    _seed_all(args.seed)

    config_path = os.path.join(args.checkpoint_dir, "config.yaml")
    assert os.path.exists(config_path), f"config.yaml not found at {config_path}"
    config = OmegaConf.load(config_path)

    h5_path = config["dataset_path"]
    print(f"Using real dataset path from config: {h5_path}")
    raw_dataset = _load_raw_dataset(config, h5_path, config.get("seed", args.seed), metadata=True)

    real_ml_models_path = os.path.join(os.path.dirname(h5_path), "ml_models", "cv")
    print(f"Real models output path: {real_ml_models_path}")
    _train_real_baseline(
        raw_dataset=raw_dataset,
        ml_models_path=real_ml_models_path,
        models=args.models,
        force_retrain=args.force_retrain,
        n_folds=args.n_folds,
        seed=args.seed,
        on_gpu=args.on_gpu,
        skip_real_training=args.skip_real_training,
    )

    syn_specs = {
        "complete": {
            "h5": os.path.join(args.checkpoint_dir, "syn_complete_dataset.hdf5"),
            "out": _label_strategy_out_dir(os.path.join(args.checkpoint_dir, "ml_models", "cv"), args.synthetic_label_strategy, args.synthetic_label_seed),
        },
        "augmented": {
            "h5": os.path.join(args.checkpoint_dir, "syn_augmented_dataset.hdf5"),
            "out": _label_strategy_out_dir(os.path.join(args.checkpoint_dir, "ml_models", "cv_augmented"), args.synthetic_label_strategy, args.synthetic_label_seed),
        },
        "reconstructed": {
            "h5": os.path.join(args.checkpoint_dir, f"vae_reconstruction_dataset_{args.reconstruction_split}.hdf5"),
            "out": _label_strategy_out_dir(os.path.join(args.checkpoint_dir, "ml_models", f"cv_reconstructed_{args.reconstruction_split}"), args.synthetic_label_strategy, args.synthetic_label_seed),
        },
    }

    for syn_dataset_type in args.syn_dataset_types:
        spec = syn_specs[syn_dataset_type]
        if not os.path.exists(spec["h5"]):
            print(f"Skipping {syn_dataset_type}; dataset not found: {spec['h5']}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating {syn_dataset_type}: {spec['h5']}")
        print(f"Synthetic label strategy: {args.synthetic_label_strategy} (seed={args.synthetic_label_seed})")
        print(f"Output path: {spec['out']}")
        print(f"{'=' * 60}")
        raw_dataset_syn = _load_raw_dataset(
            config,
            spec["h5"],
            config.get("seed", args.seed),
            x_key="syn_samples",
            y_key="targets",
            metadata=True,
        )
        _train_synthetic_dataset(
            raw_dataset_syn=raw_dataset_syn,
            raw_dataset_real=raw_dataset,
            syn_dataset_type=syn_dataset_type,
            ml_models_path_syn=spec["out"],
            models=args.models,
            force_retrain=args.force_retrain,
            n_folds=args.n_folds,
            seed=args.seed,
            on_gpu=args.on_gpu,
            synthetic_label_strategy=args.synthetic_label_strategy,
            synthetic_label_seed=args.synthetic_label_seed,
        )

    print("\nCross-validation evaluation complete")


if __name__ == "__main__":
    main()
