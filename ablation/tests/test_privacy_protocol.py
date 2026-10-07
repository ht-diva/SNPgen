from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from snpgen.evaluation.privacy.metrics import PrivacyEvaluator
from snpgen.evaluation.privacy.distances import onehot_encode_snps
from snpgen.evaluation.privacy.utils import (
    PrivacyDataBundle,
    load_privacy_data_manual,
    load_privacy_results,
    save_privacy_results,
)
from snpgen.evaluation.privacy.yelmen import (
    adversarial_accuracy,
    matched_privacy_cohorts,
    nearest_neighbor_chain_analysis,
    yelmen_privacy_loss,
)


def _brute_knn(query, ref, k=1, metric='hamming', **_kwargs):
    query = np.asarray(query)
    ref = np.asarray(ref)
    if metric == 'hamming':
        distances = np.sum(query[:, None, :] != ref[None, :, :], axis=2)
    elif metric == 'manhattan':
        distances = np.sum(np.abs(query[:, None, :] - ref[None, :, :]), axis=2)
    else:
        raise ValueError(metric)
    indices = np.argsort(distances, axis=1, kind='stable')[:, :k]
    return np.take_along_axis(distances, indices, axis=1), indices


def _manual_aa(truth, synthetic, metric):
    def distance(a, b):
        if metric == 'hamming':
            return np.sum(a != b)
        return np.sum(np.abs(a - b))

    d_ts = np.array([min(distance(row, other) for other in synthetic) for row in truth])
    d_tt = np.array([
        min(distance(row, truth[j]) for j in range(len(truth)) if j != i)
        for i, row in enumerate(truth)
    ])
    d_st = np.array([min(distance(row, other) for other in truth) for row in synthetic])
    d_ss = np.array([
        min(distance(row, synthetic[j]) for j in range(len(synthetic)) if j != i)
        for i, row in enumerate(synthetic)
    ])
    aa_truth = np.mean(d_ts > d_tt)
    aa_syn = np.mean(d_st > d_ss)
    return aa_truth, aa_syn, (aa_truth + aa_syn) / 2


def _write_h5(path, data, labels, synthetic=False):
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('syn_samples' if synthetic else 'data', data=data)
        handle.create_dataset('targets' if synthetic else 'labels', data=labels)


def test_chunked_onehot_encoding_matches_unchunked_encoding():
    data = np.array([[0, 1, 2], [2, 0, 1], [1, 1, 0]], dtype=np.int8)
    unchunked = onehot_encode_snps(data)
    chunked = onehot_encode_snps(data, device='cpu', batch_size=1)
    assert np.array_equal(unchunked.numpy(), chunked.numpy())
    assert tuple(chunked.shape) == (3, 9)


def test_matched_privacy_cohorts_are_equal_stratified_and_deterministic():
    train = np.arange(60).reshape(20, 3)
    test = np.arange(45).reshape(15, 3)
    synthetic = np.arange(75).reshape(25, 3)
    train_y = np.array([0] * 12 + [1] * 8)
    test_y = np.array([0] * 9 + [1] * 6)
    syn_y = np.array([0] * 14 + [1] * 11)

    first = matched_privacy_cohorts(
        train, test, synthetic, train_y, test_y, syn_y, max_samples=12, seed=7
    )
    second = matched_privacy_cohorts(
        train, test, synthetic, train_y, test_y, syn_y, max_samples=12, seed=7
    )
    assert first.train.shape == first.test.shape == first.synthetic.shape == (12, 3)
    assert first.class_counts == {'0': 7, '1': 5}
    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.test, second.test)
    assert np.array_equal(first.synthetic, second.synthetic)


def test_adversarial_accuracy_matches_independent_reference_for_both_distances():
    truth = np.array([[0, 0, 0], [0, 1, 0], [2, 2, 2], [2, 1, 2]])
    synthetic = np.array([[0, 0, 1], [0, 2, 0], [2, 2, 1], [1, 1, 2]])
    for metric in ('hamming', 'manhattan'):
        observed = adversarial_accuracy(
            truth, synthetic, metric=metric, knn_fn=_brute_knn, verbose=False
        )
        assert np.allclose(observed, _manual_aa(truth, synthetic, metric))


def test_yelmen_privacy_loss_has_published_subtraction_direction():
    train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    test = np.array([[2, 2], [2, 1], [1, 2], [2, 0]])
    matched = matched_privacy_cohorts(train, test, train.copy(), max_samples=4, seed=2)
    result = yelmen_privacy_loss(
        matched, metric='manhattan', knn_fn=_brute_knn, verbose=False
    )
    assert result.privacy_loss == result.test_aa_ts - result.train_aa_ts
    assert result.privacy_loss > 0
    assert result.metric == 'manhattan'


def test_nearest_neighbor_chains_cover_every_start_and_pattern_space():
    real = np.array([[0, 0], [0, 1], [0, 2]])
    synthetic = np.array([[2, 2], [2, 1], [2, 0]])
    result = nearest_neighbor_chain_analysis(
        real,
        synthetic,
        min_length=2,
        max_length=4,
        neighbor_k=5,
        knn_fn=_brute_knn,
        verbose=False,
    )
    for length in range(2, 5):
        key = str(length)
        assert len(result.pattern_counts[key]) == 2 ** length
        assert sum(result.pattern_counts[key].values()) == 6
        assert np.isclose(sum(result.pattern_frequencies[key].values()), 1.0)
        assert result.expected_equal_mixture_frequency[key] == 0.5 ** length


def test_nearest_neighbor_chains_skip_previously_visited_nodes():
    real = np.array([[0], [1]])
    synthetic = np.array([[2], [3]])

    def fixed_graph(query, ref, k=1, **_kwargs):
        indices = np.array([
            [0, 2, 1, 3],
            [1, 0, 2, 3],
            [2, 0, 3, 1],
            [3, 2, 1, 0],
        ])[:, :k]
        distances = np.tile(np.arange(k), (4, 1))
        return distances, indices

    result = nearest_neighbor_chain_analysis(
        real,
        synthetic,
        min_length=3,
        max_length=3,
        neighbor_k=4,
        knn_fn=fixed_graph,
        verbose=False,
    )
    # Starting at real node 0 follows 0(T) -> 2(S), then skips visited 0 and
    # selects 3(S), yielding TSS rather than returning to T.
    assert result.pattern_counts['3']['TSS'] >= 1


def test_nearest_neighbor_chains_reject_truncated_knn_results():
    real = np.array([[0], [1]])
    synthetic = np.array([[2], [3]])

    def truncated_knn(query, ref, k=1, **_kwargs):
        return np.zeros((2, k)), np.zeros((2, k), dtype=np.int64)

    with pytest.raises(RuntimeError, match='full-cohort kNN'):
        nearest_neighbor_chain_analysis(
            real,
            synthetic,
            max_length=4,
            neighbor_k=4,
            knn_fn=truncated_knn,
            verbose=False,
        )


def test_protocol_upgrade_removes_only_stale_record_metrics(tmp_path):
    output = tmp_path / 'privacy'
    old = {
        'mi__overall': {'old': True},
        'nnaa__class_1': {'old': True},
        'maf__overall': {'keep': True},
        'case_control_delta__overall': {'keep': True},
    }
    save_privacy_results(old, output)
    bundle = PrivacyDataBundle(
        real_train=np.zeros((6, 2), dtype=np.int8),
        real_holdout=np.zeros((2, 2), dtype=np.int8),
        synthetic=np.zeros((8, 2), dtype=np.int8),
        real_validation=np.zeros((2, 2), dtype=np.int8),
        real_train_val=np.zeros((8, 2), dtype=np.int8),
        dataset_path='/data/real.h5',
        synthetic_path='/data/syn.h5',
    )
    evaluator = PrivacyEvaluator(verbose=False)
    kept, _manifest = evaluator._prepare_synthetic_results(bundle, output)
    assert sorted(kept) == ['case_control_delta__overall', 'maf__overall']
    assert sorted(load_privacy_results(output)) == sorted(kept)
    manifest = json.loads((output / 'privacy_manifest.json').read_text())
    assert manifest['real_training_split'] == 'train'
    assert manifest['fidelity_reference_split'] == 'train_val'


def test_matching_manifest_reuses_corrected_record_metrics(tmp_path):
    output = tmp_path / 'privacy'
    bundle = PrivacyDataBundle(
        real_train=np.zeros((6, 2), dtype=np.int8),
        real_holdout=np.zeros((2, 2), dtype=np.int8),
        synthetic=np.zeros((8, 2), dtype=np.int8),
        real_validation=np.zeros((2, 2), dtype=np.int8),
        real_train_val=np.zeros((8, 2), dtype=np.int8),
        dataset_path='/data/real.h5',
        synthetic_path='/data/syn.h5',
    )
    evaluator = PrivacyEvaluator(verbose=False)
    _results, manifest = evaluator._prepare_synthetic_results(bundle, output)
    save_privacy_results({'mi__overall': {'corrected': True}}, output)
    from snpgen.evaluation.privacy.utils import save_privacy_manifest
    save_privacy_manifest(manifest, output)
    reused, _manifest = evaluator._prepare_synthetic_results(bundle, output)
    assert reused['mi__overall'] == {'corrected': True}


def test_changing_only_chain_settings_keeps_other_corrected_metrics(tmp_path):
    output = tmp_path / 'privacy'
    bundle = PrivacyDataBundle(
        real_train=np.zeros((6, 2), dtype=np.int8),
        real_holdout=np.zeros((2, 2), dtype=np.int8),
        synthetic=np.zeros((8, 2), dtype=np.int8),
        real_validation=np.zeros((2, 2), dtype=np.int8),
        real_train_val=np.zeros((8, 2), dtype=np.int8),
        dataset_path='/data/real.h5',
        synthetic_path='/data/syn.h5',
    )
    first = PrivacyEvaluator(verbose=False, chain_neighbor_k=16)
    _results, manifest = first._prepare_synthetic_results(bundle, output)
    saved = {
        'mi__overall': {'keep': True},
        'yelmen_privacy_loss__overall': {'keep': True},
        'nearest_neighbor_chain__overall': {'remove': True},
    }
    save_privacy_results(saved, output)
    from snpgen.evaluation.privacy.utils import save_privacy_manifest
    save_privacy_manifest(manifest, output)

    second = PrivacyEvaluator(verbose=False, chain_neighbor_k=32)
    observed, _manifest = second._prepare_synthetic_results(bundle, output)
    assert observed['mi__overall'] == {'keep': True}
    assert observed['yelmen_privacy_loss__overall'] == {'keep': True}
    assert 'nearest_neighbor_chain__overall' not in observed


def test_replacing_synthetic_file_invalidates_record_metrics(tmp_path):
    output = tmp_path / 'privacy'
    synthetic_path = tmp_path / 'synthetic.h5'
    synthetic_path.write_bytes(b'first')
    bundle = PrivacyDataBundle(
        real_train=np.zeros((6, 2), dtype=np.int8),
        real_holdout=np.zeros((2, 2), dtype=np.int8),
        synthetic=np.zeros((8, 2), dtype=np.int8),
        real_validation=np.zeros((2, 2), dtype=np.int8),
        real_train_val=np.zeros((8, 2), dtype=np.int8),
        dataset_path='/data/real.h5',
        synthetic_path=str(synthetic_path),
    )
    evaluator = PrivacyEvaluator(verbose=False)
    _results, manifest = evaluator._prepare_synthetic_results(bundle, output)
    save_privacy_results(
        {'mi__overall': {'remove': True}, 'maf__overall': {'keep': True}}, output
    )
    from snpgen.evaluation.privacy.utils import save_privacy_manifest
    save_privacy_manifest(manifest, output)

    synthetic_path.write_bytes(b'a replacement with a different size')
    observed, _manifest = evaluator._prepare_synthetic_results(bundle, output)
    assert 'mi__overall' not in observed
    assert observed['maf__overall'] == {'keep': True}


def test_manual_loader_exposes_true_train_validation_test_and_train_val(tmp_path):
    rng = np.random.default_rng(3)
    data = rng.integers(0, 3, size=(100, 5), dtype=np.int8)
    labels = np.array([0, 1] * 50)
    real_path = tmp_path / 'real.h5'
    syn_dir = tmp_path / 'generator'
    syn_dir.mkdir()
    _write_h5(real_path, data, labels)
    _write_h5(syn_dir / 'syn_complete_dataset.hdf5', data, labels, synthetic=True)

    bundle = load_privacy_data_manual(
        dataset_path=real_path,
        ddpm_checkpoint_dir=syn_dir,
        seed=42,
        val_ratio=0.2,
        test_ratio=0.1,
        verbose=False,
    )
    assert len(bundle.real_train) == 72
    assert len(bundle.real_validation) == 18
    assert len(bundle.real_holdout) == 10
    assert len(bundle.real_train_val) == 90
    assert len(bundle.labels_train) == len(bundle.real_train)
    assert len(bundle.labels_validation) == len(bundle.real_validation)


def test_manual_loader_rejects_validation_as_membership_holdout(tmp_path):
    rng = np.random.default_rng(4)
    data = rng.integers(0, 3, size=(40, 5), dtype=np.int8)
    labels = np.array([0, 1] * 20)
    real_path = tmp_path / 'real.h5'
    syn_dir = tmp_path / 'generator'
    syn_dir.mkdir()
    _write_h5(real_path, data, labels)
    _write_h5(syn_dir / 'syn_complete_dataset.hdf5', data, labels, synthetic=True)

    with pytest.raises(ValueError, match="holdout_split='test'"):
        load_privacy_data_manual(
            dataset_path=real_path,
            ddpm_checkpoint_dir=syn_dir,
            holdout_split='val',
            verbose=False,
        )


def test_tiny_corrected_evaluator_writes_new_metrics_and_manifest(tmp_path):
    rng = np.random.default_rng(8)
    train = rng.integers(0, 3, size=(20, 4), dtype=np.int8)
    validation = rng.integers(0, 3, size=(6, 4), dtype=np.int8)
    test = rng.integers(0, 3, size=(8, 4), dtype=np.int8)
    synthetic = rng.integers(0, 3, size=(24, 4), dtype=np.int8)
    bundle = PrivacyDataBundle(
        real_train=train,
        real_holdout=test,
        synthetic=synthetic,
        labels_train=np.array([0, 1] * 10),
        labels_holdout=np.array([0, 1] * 4),
        labels_syn=np.array([0, 1] * 12),
        real_validation=validation,
        real_train_val=np.concatenate((train, validation)),
        labels_validation=np.array([0, 1] * 3),
        labels_train_val=np.array([0, 1] * 13),
        dataset_path='/data/toy.h5',
        synthetic_path='/data/toy_syn.h5',
    )
    output = tmp_path / 'privacy'
    evaluator = PrivacyEvaluator(
        device='cpu',
        verbose=False,
        per_class=False,
        nnaa_n_samples=8,
        matched_n_samples=8,
        chain_neighbor_k=6,
        cpu_max_samples=100,
    )
    results = evaluator.evaluate(bundle, output, eval_target='synthetic')
    assert 'mi__overall' in results
    assert 'yelmen_privacy_loss__overall' in results
    assert 'nearest_neighbor_chain__overall' in results
    assert 'maf__overall' in results
    manifest = json.loads((output / 'privacy_manifest.json').read_text())
    assert manifest['protocol_version'] == evaluator.SYNTHETIC_PROTOCOL_VERSION
    assert manifest['completed_metric_keys'] == sorted(results)
