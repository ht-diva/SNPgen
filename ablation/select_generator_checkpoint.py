"""Select a generator checkpoint by quick one-shot downstream TSTR performance."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from typing import Any

import numpy as np
import torch
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from ablation.run_generate import _generate_from_labels, _load_checkpoint_model
from ablation.utils import resolve_checkpoint_dir_with_config
from snpgen.evaluation import train_models
from snpgen.utils import instantiate_from_config

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _load_real_splits(config: Any, h5_path: str, seed: int):
    raw_dataset = instantiate_from_config(
        config.data.raw_dataset,
        file_path=h5_path,
        seed=seed,
        onehot=False,
        data_dtype=None,
        metadata=False,
    )
    _train_data, train_labels = raw_dataset.get_split("train", metadata=False)
    val_data, val_labels = raw_dataset.get_split("val", metadata=False)
    return (
        np.asarray(train_labels).reshape(-1).astype(np.int64),
        np.asarray(val_data),
        np.asarray(val_labels).reshape(-1).astype(np.int64),
    )


def _checkpoint_epoch(path: str) -> int:
    name = os.path.basename(path)
    match = re.search(r"epoch=(\d+)", name)
    if match:
        return int(match.group(1))
    if name == "last.ckpt":
        return 10**12
    return -1


def _candidate_checkpoints(checkpoint_dir: str, pattern: str, include_last: bool) -> list[str]:
    paths = glob.glob(os.path.join(checkpoint_dir, pattern))
    paths.extend(glob.glob(os.path.join(checkpoint_dir, "best-*.ckpt")))
    paths = sorted(set(paths), key=_checkpoint_epoch)
    if include_last:
        last_path = os.path.join(checkpoint_dir, "last.ckpt")
        if os.path.exists(last_path) and last_path not in paths:
            paths.append(last_path)
    return paths


def _metric_from_results(results: dict, metric: str) -> tuple[str, float]:
    preferred = "prs univariate scaled (threshold 0.5)"
    if preferred in results and metric in results[preferred]["metrics"]:
        return preferred, float(results[preferred]["metrics"][metric])

    candidates = []
    for model_name, payload in results.items():
        metrics = payload.get("metrics", {})
        if metric in metrics:
            candidates.append((model_name, float(metrics[metric])))
    if not candidates:
        raise KeyError(f"Metric {metric!r} not found in downstream results: {list(results)}")
    return max(candidates, key=lambda item: item[1])


def _stratified_label_subset(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    if max_samples <= 0 or labels.shape[0] <= max_samples:
        return labels

    rng = np.random.default_rng(seed)
    selected = []
    classes, counts = np.unique(labels, return_counts=True)
    allocation = np.floor(counts / counts.sum() * max_samples).astype(int)
    allocation = np.maximum(allocation, 1)
    while allocation.sum() > max_samples:
        allocation[np.argmax(allocation)] -= 1
    while allocation.sum() < max_samples:
        allocation[np.argmax(counts - allocation)] += 1

    for cls, n_take in zip(classes, allocation):
        cls_idx = np.flatnonzero(labels == cls)
        selected.append(rng.choice(cls_idx, size=min(int(n_take), cls_idx.shape[0]), replace=False))
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return labels[indices]


def _write_outputs(output_dir: str, rows: list[dict], best: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "selection_results.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(output_dir, "selection_results.json")
    with open(json_path, "w") as handle:
        json.dump({"best": best, "results": rows}, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-pattern", default="epoch=*.ckpt")
    parser.add_argument("--include-last", action="store_true")
    parser.add_argument("--batch-size", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+", default=["prs"])
    parser.add_argument("--metric", default="roc_auc")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--output-subdir", default="checkpoint_selection")
    args = parser.parse_args()
    args.checkpoint_dir = resolve_checkpoint_dir_with_config(args.checkpoint_dir)

    seed_everything(args.seed, workers=True)

    config_path = os.path.join(args.checkpoint_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found at {config_path}")
    config = OmegaConf.load(config_path)

    h5_path = config["dataset_path"]
    y_syn_reference, X_val, y_val = _load_real_splits(
        config=config,
        h5_path=h5_path,
        seed=config.get("seed", args.seed),
    )
    y_syn_reference = _stratified_label_subset(y_syn_reference, args.max_train_samples, args.seed)

    checkpoints = _candidate_checkpoints(args.checkpoint_dir, args.checkpoint_pattern, args.include_last)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints matching {args.checkpoint_pattern!r} in {args.checkpoint_dir}"
        )

    output_dir = os.path.join(args.checkpoint_dir, args.output_subdir)
    rows = []
    for checkpoint_path in checkpoints:
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"\n=== Selecting generator checkpoint: {checkpoint_name} ===")
        model = _load_checkpoint_model(config, checkpoint_path, args.device)
        generated = _generate_from_labels(model, y_syn_reference, args.batch_size, args.device)
        X_train_syn = np.asarray(generated["samples"])
        y_train_syn = np.asarray(generated["targets"]).reshape(-1).astype(np.int64)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status = "ok"
        error = ""
        try:
            downstream = train_models(
                X_train_syn,
                y_train_syn,
                X_val,
                y_val,
                metrics_dict=None,
                save_path=None,
                on_gpu=False,
                seed=args.seed,
                models=args.models,
            )
            selected_model, selected_value = _metric_from_results(downstream, args.metric)
            if not np.isfinite(selected_value):
                raise ValueError(f"Non-finite selector metric: {selected_value}")
        except ValueError as exc:
            selected_model = "selector_fallback_no_signal"
            selected_value = 0.5
            status = "fallback"
            error = str(exc)
            print(
                f"WARNING: downstream selector failed for {checkpoint_name}; "
                f"recording {args.metric}=0.5 and continuing. Error: {error}"
            )
        row = {
            "checkpoint": checkpoint_name,
            "checkpoint_path": checkpoint_path,
            "epoch": _checkpoint_epoch(checkpoint_path),
            "model": selected_model,
            "metric": args.metric,
            "value": selected_value,
            "selection_split": "val",
            "n_train_syn": int(X_train_syn.shape[0]),
            "n_holdout_real": int(X_val.shape[0]),
            "status": status,
            "error": error,
        }
        rows.append(row)
        print(f"{checkpoint_name}: {selected_model} {args.metric}={selected_value:.6f}")

    best = max(rows, key=lambda row: (row["value"], row["epoch"]))
    _write_outputs(output_dir, rows, best)

    selected_marker = os.path.join(args.checkpoint_dir, "selected_checkpoint.txt")
    with open(selected_marker, "w") as handle:
        handle.write(best["checkpoint"] + "\n")

    print("\nSelected checkpoint:")
    print(json.dumps(best, indent=2))
    print(f"Wrote marker: {selected_marker}")


if __name__ == "__main__":
    main()
