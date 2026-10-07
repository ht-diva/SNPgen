from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ablation.evaluation.cohort_protocol import matched_class_counts, select_indices_by_class
from ablation.evaluation.run_association import main as association_main
from ablation.evaluation.run_structure import main as structure_main
from snpgen.evaluation.association import benjamini_hochberg, compare_association_effects, fit_binary_associations
from snpgen.evaluation.genotype_structure import (
    allele_frequencies,
    fit_reference_pca,
    hwe_exact_p_value,
    pair_cache,
    pairwise_ld_r2,
    transform_reference_pca,
)


def _write_h5(path, x, y, synthetic=False):
    with h5py.File(path, "w") as hf:
        hf.create_dataset("syn_samples" if synthetic else "data", data=x)
        hf.create_dataset("targets" if synthetic else "labels", data=y)
        if not synthetic:
            hf.create_dataset("snp_ids", data=np.array([f"rs{i}".encode() for i in range(x.shape[1])]))
            hf.create_dataset("chrom", data=np.ones(x.shape[1], dtype=int))
            hf.create_dataset("pos", data=np.arange(1, x.shape[1] + 1) * 1000)


def test_grouped_logistic_matches_direct_likelihood():
    rng = np.random.default_rng(12)
    x = rng.binomial(2, 0.35, size=(3000, 1))
    probability = 1 / (1 + np.exp(-(-1.0 + 0.7 * x[:, 0])))
    y = rng.binomial(1, probability)
    fit = fit_binary_associations(x, y).iloc[0]

    design = np.column_stack([np.ones(x.shape[0]), x[:, 0]])
    objective = lambda theta: np.sum(np.logaddexp(0, design @ theta) - y * (design @ theta))
    reference = minimize(objective, np.zeros(2), method="BFGS")
    assert fit.fit_status == "ok"
    assert np.isclose(fit.beta, reference.x[1], atol=1e-5)
    assert fit.se > 0
    assert 0 <= fit.p_value <= 1


def test_association_flags_and_comparison():
    dosage_block = np.array([0] * 4 + [1] * 4 + [2] * 4)
    label_block = np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
    dosage = np.tile(dosage_block, 10)
    y = np.tile(label_block, 10)
    x = np.column_stack([np.zeros(y.size), dosage])
    fit = fit_binary_associations(x, y)
    assert fit.loc[0, "fit_status"] == "monomorphic"
    comparison = compare_association_effects(fit, fit)
    assert np.isclose(comparison["beta_calibration_slope_origin"], 1)
    assert np.isclose(comparison["sign_concordance"], 1)


def test_benjamini_hochberg_known_values():
    observed = benjamini_hochberg(np.array([0.01, 0.04, 0.03, np.nan]))
    assert np.allclose(observed[:3], [0.03, 0.04, 0.04])
    assert np.isnan(observed[3])


def test_matching_is_exact_and_deterministic():
    labels = {"real": np.array([0] * 80 + [1] * 20), "model": np.array([0] * 60 + [1] * 40)}
    counts = matched_class_counts(labels, max_samples=50)
    assert counts == {0: 40, 1: 10}
    a = select_indices_by_class(labels["model"], counts, seed=3)
    b = select_indices_by_class(labels["model"], counts, seed=3)
    assert np.array_equal(a, b)
    assert np.sum(labels["model"][a] == 0) == 40


def test_structure_primitives():
    x = np.array([[0, 0, 2], [1, 1, 0], [2, 2, 2], [1, 1, 0]], dtype=float)
    af, maf = allele_frequencies(x)
    assert np.allclose(af, [0.5, 0.5, 0.5])
    assert np.allclose(maf, [0.5, 0.5, 0.5])
    ld = pairwise_ld_r2(x)
    assert np.isclose(ld[0, 1], 1.0)
    assert hwe_exact_p_value(10, 20, 10) == hwe_exact_p_value(10, 20, 10)
    assert hwe_exact_p_value(40, 0, 0) == 1.0
    left, right, distance = pair_cache(np.array([1, 1, 2]), np.array([10, 20, 30]))
    assert np.array_equal(left, [0]) and np.array_equal(right, [1]) and np.array_equal(distance, [10])


def test_reference_pca_transform_is_identical():
    rng = np.random.default_rng(4)
    x = rng.binomial(2, 0.4, size=(200, 8))
    state, scores = fit_reference_pca(x, n_components=4, seed=42)
    assert np.allclose(scores, transform_reference_pca(x, state), atol=1e-5)


def test_tiny_end_to_end_outputs(tmp_path):
    rng = np.random.default_rng(7)
    y = np.array([0] * 80 + [1] * 40)
    x = rng.binomial(2, 0.3 + 0.08 * y[:, None], size=(120, 6)).astype(np.int8)
    syn_y = y.copy()
    syn_x = rng.binomial(2, 0.3 + 0.06 * syn_y[:, None], size=(120, 6)).astype(np.int8)
    real_path, syn_path = tmp_path / "real.h5", tmp_path / "syn.h5"
    _write_h5(real_path, x, y)
    _write_h5(syn_path, syn_x, syn_y, synthetic=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"seed: 42\ndataset_path: {real_path}\ndata:\n  raw_dataset:\n    params:\n      test_ratio: 0.2\n"
    )

    association_dir = tmp_path / "association"
    association_main([
        "--trait", "toy", "--reference-config", str(config_path), "--cohort", f"synthetic={syn_path}",
        "--output-dir", str(association_dir), "--max-samples", "80",
    ])
    assert (association_dir / "per_variant.csv.gz").is_file()
    assert pd.read_csv(association_dir / "comparison_summary.csv").shape[0] == 1
    assert json.loads((association_dir / "manifest.json").read_text())["protocol_version"] == "association_v1"

    structure_dir = tmp_path / "structure"
    structure_main([
        "--trait", "toy", "--reference-config", str(config_path), "--cohort", f"synthetic={syn_path}",
        "--output-dir", str(structure_dir), "--max-samples", "20", "--pca-components", "3", "--ld-bins", "3",
    ])
    for filename in ("manifest.json", "summary.csv", "per_snp.csv.gz", "ld_decay.csv", "pca_summary.csv", "pca_scores.npz", "pairwise_ld_r2.npz"):
        assert (structure_dir / filename).is_file()
    per_snp = pd.read_csv(structure_dir / "per_snp.csv.gz")
    assert set(per_snp.hwe_scope) == {"pooled", "controls"}
    assert set(per_snp.frequency_scope) == {"pooled"}
    assert {"pooled_af", "pooled_maf", "real_pooled_af", "delta_pooled_af"}.issubset(per_snp.columns)
