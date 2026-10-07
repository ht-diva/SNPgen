import h5py
import numpy as np
import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset

import ablation.select_generator_checkpoint as selector
import ablation.run_train as run_train
from ablation.run_generate import _fixed_prevalence_labels
from ablation.baselines.af_sampler import ClassConditionalAFSampler
from ablation.evaluation.schema import load_standard_hdf5, save_standard_hdf5, validate_standard_hdf5
from ablation.models.conditional_vae import ConditionalAutoencoder, ConditionalAutoencoderTrainingWrapper
from ablation.models.crbm import ConditionalCRBMModule, _dosage_to_random_phase_haplotypes
from ablation.models.wgan_gp import ConditionalWGAN_GPModule
from snpgen.training.callbacks.checkpointing import ValidationAndPeriodicCheckpoint
from snpgen.utils import instantiate_from_config


def test_run_train_cli_parses_cpu_smoke_arguments():
    args = run_train.build_parser().parse_args(
        [
            "--config", "ablation/configs/conditional_crbm.yaml",
            "--output-dir", "/tmp/crbm-smoke",
            "--no-wandb",
            "--accelerator", "cpu",
            "--devices", "1",
            "--limit-train-batches", "2",
            "--limit-val-batches", "0.5",
            "dataset_path=/tmp/cohort.hdf5",
        ]
    )
    assert args.accelerator == "cpu"
    assert args.devices == 1
    assert args.limit_train_batches == 2
    assert args.limit_val_batches == 0.5
    assert args.overrides == ["dataset_path=/tmp/cohort.hdf5"]
    single_batch = run_train.build_parser().parse_args(
        ["--config", "unused.yaml", "--limit-val-batches", "1"]
    )
    assert type(single_batch.limit_val_batches) is int
    full_validation = run_train.build_parser().parse_args(
        ["--config", "unused.yaml", "--limit-val-batches", "1.0"]
    )
    assert type(full_validation.limit_val_batches) is float


def test_run_train_trait_registry_works_without_checkpoint_config(tmp_path):
    from omegaconf import OmegaConf

    trait_path = tmp_path / "traits.yaml"
    trait_path.write_text(
        "traits:\n  example:\n    dataset_path: /tmp/cohort.hdf5\n    seq_len: 16\n"
    )
    config = OmegaConf.create(
        {
            "seq_len": 8,
            "data": {
                "raw_dataset": {"params": {"file_path": "unused"}},
                "dataset": {"params": {"seq_len": 8}},
            },
            "model": {"params": {"seq_len": 8}},
            "experiment": {"base_scratch_dir": "./runs"},
        }
    )

    run_train._apply_trait_overrides(config, "example", str(trait_path))

    assert config.dataset_path == "/tmp/cohort.hdf5"
    assert config.data.raw_dataset.params.file_path == "/tmp/cohort.hdf5"
    assert config.seq_len == 16
    assert config.data.dataset.params.seq_len == 16
    assert config.experiment.proj_name == "example"


def test_standard_hdf5_roundtrip(tmp_path):
    samples = np.array([[0, 1, 2, 0], [2, 1, 0, 1]], dtype=np.int8)
    labels = np.array([0, 1], dtype=np.int32)
    path = tmp_path / "syn.hdf5"

    save_standard_hdf5(str(path), samples, labels, attrs={"model_name": "test"})
    loaded = validate_standard_hdf5(str(path))

    assert np.array_equal(loaded.samples, samples)
    assert np.array_equal(loaded.labels, labels)
    assert loaded.attrs["model_name"] == "test"
    with h5py.File(path, "r") as hf:
        assert {"syn_samples", "targets", "data", "labels"}.issubset(hf.keys())


def test_af_sampler_generates_valid_genotypes():
    x = np.array(
        [
            [0, 0, 1, 2],
            [0, 1, 1, 2],
            [2, 2, 1, 0],
            [2, 1, 0, 0],
        ],
        dtype=np.int8,
    )
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    sampler = ClassConditionalAFSampler(alpha=0.5).fit(x, y)
    out = sampler.sample([0, 1, 0, 1], seed=123)

    assert out.shape == (4, 4)
    assert out.dtype == np.int8
    assert set(np.unique(out)).issubset({0, 1, 2})


def _tiny_conditional_vae_config():
    return {
        "target": "ablation.models.conditional_vae.ConditionalAutoencoder",
        "params": {
            "n_classes": 2,
            "input_ch": 3,
            "z_channels": 1,
            "z_dim": 8,
            "condition_encoder": True,
            "condition_decoder": True,
            "encoder_config": {
                "target": "ablation.tests.stubs.TinyEncoder",
                "params": {
                    "input_ch": 3,
                    "z_channels": 1,
                },
            },
            "decoder_config": {
                "target": "ablation.tests.stubs.TinyDecoder",
                "params": {
                    "out_ch": 3,
                    "z_channels": 1,
                },
            },
        },
    }


def test_conditional_vae_forward_and_sample():
    model = instantiate_from_config(_tiny_conditional_vae_config())
    assert isinstance(model, ConditionalAutoencoder)

    x_idx = torch.randint(0, 3, (2, 8))
    x = torch.nn.functional.one_hot(x_idx, num_classes=3).permute(0, 2, 1).float()
    y = torch.tensor([0, 1])
    mu, logvar = model.encode(x, y)
    logits = model.decode(mu, y, argmax=False)
    samples = model.sample(y, argmax=True)

    assert mu.shape == (2, 1, 8)
    assert logvar.shape == (2, 1, 8)
    assert logits.shape == (2, 3, 8)
    assert samples.shape == (2, 8)
    assert set(torch.unique(samples).tolist()).issubset({0, 1, 2})


def test_wgan_sample_shape_and_range():
    model = ConditionalWGAN_GPModule(seq_len=8, latent_dim=4, generator_channels=8, critic_channels=4)
    labels = torch.tensor([0, 1, 1])
    samples = model.sample(labels, argmax=True)

    assert samples.shape == (3, 8)
    assert set(torch.unique(samples).tolist()).issubset({0, 1, 2})


def test_crbm_sample_shape_and_range():
    model = ConditionalCRBMModule(seq_len=8, num_hidden=4, gibbs_steps=2, generation_gibbs_steps=2)
    labels = torch.tensor([0, 1, 1])
    samples = model.sample(labels, argmax=True)

    assert samples.shape == (3, 8)
    assert set(torch.unique(samples).tolist()).issubset({0, 1, 2})


def test_crbm_random_phase_haplotypes_preserve_dosage_and_are_symmetric():
    torch.manual_seed(7)
    dosage = torch.ones(4000, 1, dtype=torch.long)
    haplotypes = _dosage_to_random_phase_haplotypes(dosage).reshape(-1, 2, 1)
    assert torch.equal(haplotypes.sum(dim=1), dosage.float())
    first_allele_frequency = haplotypes[:, 0].mean().item()
    assert 0.47 < first_allele_frequency < 0.53


def test_crbm_unconditional_random_phase_mode():
    model = ConditionalCRBMModule(
        seq_len=8,
        num_hidden=4,
        gibbs_steps=2,
        generation_gibbs_steps=2,
        condition_on_label=False,
        genotype_encoding="random_phase_haplotypes",
    )
    assert model.stage0.fixed_nodes == 0
    assert model.stage1.fixed_nodes == model.first_segment_len
    samples = model.sample(labels=None, num_samples=3)
    assert samples.shape == (3, 8)
    assert set(torch.unique(samples).tolist()).issubset({0, 1, 2})


def test_wgan_unconditional_has_no_label_embeddings_and_accepts_no_labels():
    model = ConditionalWGAN_GPModule(
        seq_len=8,
        channels=8,
        condition_on_label=False,
        pack_m=3,
    )
    assert not any("by_label" in key for key in model.state_dict())
    samples = model.sample(labels=None, num_samples=3)
    assert samples.shape == (3, 8)


def test_crbm_training_step_updates_weights():
    model = ConditionalCRBMModule(seq_len=8, num_hidden=4, gibbs_steps=2, generation_gibbs_steps=2)
    x_idx = torch.randint(0, 3, (6, 8))
    x = torch.nn.functional.one_hot(x_idx, num_classes=3).permute(0, 2, 1).float()
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    model.initialize_from_dataloader(DataLoader(TensorDataset(x, y), batch_size=3))
    before0 = model.stage0.W.clone()
    before1 = model.stage1.W.clone()
    model.training_step({"x": x, "y": y}, 0)

    assert not torch.equal(before0, model.stage0.W)
    assert not torch.equal(before1, model.stage1.W)


def test_validation_and_periodic_checkpoint_keeps_single_best(tmp_path):
    class DummyTrainer:
        def __init__(self):
            self.current_epoch = 0
            self.global_step = 0
            self.callback_metrics = {}
            self.sanity_checking = False

        def save_checkpoint(self, path):
            with open(path, "w") as handle:
                handle.write("checkpoint")

    trainer = DummyTrainer()
    callback = ValidationAndPeriodicCheckpoint(
        dirpath=str(tmp_path),
        every_n_epochs=2,
        save_last=True,
    )

    trainer.callback_metrics["val/loss"] = torch.tensor(0.5)
    callback.on_validation_end(trainer, None)
    first_best = list(tmp_path.glob("best-*-val_loss=0.5000.ckpt"))
    assert len(first_best) == 1

    trainer.current_epoch = 1
    trainer.global_step = 3
    callback.on_train_epoch_end(trainer, None)
    assert (tmp_path / "epoch=1-step=3.ckpt").exists()

    trainer.callback_metrics["val/loss"] = torch.tensor(0.25)
    callback.on_validation_end(trainer, None)
    assert not first_best[0].exists()
    assert len(list(tmp_path.glob("best-*.ckpt"))) == 1
    assert list(tmp_path.glob("best-*-val_loss=0.2500.ckpt"))

    callback.on_train_end(trainer, None)
    assert (tmp_path / "last.ckpt").exists()


def test_validation_checkpoint_supports_val_recon_acc_filename(tmp_path):
    class DummyTrainer:
        current_epoch = 4
        global_step = 12
        callback_metrics = {"val/metrics/recons/accuracy": torch.tensor(0.8125)}
        sanity_checking = False

        def save_checkpoint(self, path):
            with open(path, "w") as handle:
                handle.write("checkpoint")

    callback = ValidationAndPeriodicCheckpoint(
        dirpath=str(tmp_path),
        monitor="val/metrics/recons/accuracy",
        mode="max",
        best_filename="best-epoch={epoch}-step={step}-val_recon_acc={val_recon_acc:.4f}",
        metric_filename_key="val_recon_acc",
    )
    callback.on_validation_end(DummyTrainer(), None)

    assert (tmp_path / "best-epoch=4-step=12-val_recon_acc=0.8125.ckpt").exists()


def test_wgan_validation_step_logs_val_loss(monkeypatch):
    model = ConditionalWGAN_GPModule(seq_len=8, latent_dim=4, generator_channels=8, critic_channels=4, pack_m=1)
    x_idx = torch.randint(0, 3, (4, 8))
    x = torch.nn.functional.one_hot(x_idx, num_classes=3).permute(0, 2, 1).float()
    y = torch.tensor([0, 1, 0, 1])
    logged = {}
    monkeypatch.setattr(model, "log_dict", lambda metrics, **kwargs: logged.update(metrics))

    loss = model.validation_step({"x": x, "y": y}, 0)

    assert torch.isfinite(loss)
    assert "val/loss" in logged
    assert torch.isfinite(logged["val/loss"])


def test_crbm_validation_step_logs_val_loss(monkeypatch):
    model = ConditionalCRBMModule(seq_len=8, num_hidden=4, gibbs_steps=2, generation_gibbs_steps=2)
    x_idx = torch.randint(0, 3, (6, 8))
    x = torch.nn.functional.one_hot(x_idx, num_classes=3).permute(0, 2, 1).float()
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    model.initialize_from_dataloader(DataLoader(TensorDataset(x, y), batch_size=3))
    logged = {}
    monkeypatch.setattr(model, "log_dict", lambda metrics, **kwargs: logged.update(metrics))

    loss = model.validation_step({"x": x, "y": y}, 0)

    assert torch.isfinite(loss)
    assert "val/loss" in logged
    assert torch.isfinite(logged["val/loss"])


def test_cvae_validation_step_logs_reconstruction_accuracy(monkeypatch):
    model = ConditionalAutoencoderTrainingWrapper(
        autoencoder_config=_tiny_conditional_vae_config(),
        loss_config={
            "target": "snpgen.models.modules.vae.losses.GeneralLoss",
            "params": {"kl_weight": 1.0},
        },
        optimizer_config={"autoencoder": {"target": "torch.optim.Adam", "params": {"lr": 1.0e-4}}},
        convert_to_onehot=True,
        channel_first=True,
    )
    x_idx = torch.randint(0, 3, (4, 8))
    x = torch.nn.functional.one_hot(x_idx, num_classes=3).permute(0, 2, 1).float()
    y = torch.tensor([0, 1, 0, 1])
    logged = {}
    monkeypatch.setattr(model, "log_dict", lambda metrics, **kwargs: logged.update(metrics))

    loss = model.validation_step({"x": x, "y": y}, 0)

    assert torch.isfinite(loss)
    assert "val/metrics/recons/accuracy" in logged
    assert "val/loss" not in logged
    assert torch.isfinite(logged["val/metrics/recons/accuracy"])


def test_selector_loads_train_labels_and_validation_holdout(monkeypatch):
    class FakeRawDataset:
        def get_split(self, split, metadata=False):
            assert metadata is False
            if split == "train":
                return np.zeros((3, 4)), np.array([0, 1, 0])
            if split == "val":
                return np.ones((2, 4)), np.array([1, 0])
            if split == "test":
                raise AssertionError("selector must not load test split")
            raise AssertionError(f"unexpected split: {split}")

    monkeypatch.setattr(selector, "instantiate_from_config", lambda *args, **kwargs: FakeRawDataset())
    config = SimpleNamespace(data=SimpleNamespace(raw_dataset={"target": "unused", "params": {}}))

    train_labels, val_data, val_labels = selector._load_real_splits(config, "unused.h5", seed=42)

    assert train_labels.tolist() == [0, 1, 0]
    assert val_data.shape == (2, 4)
    assert val_labels.tolist() == [1, 0]


def test_generated_label_design_uses_reference_prevalence():
    reference = np.array([0, 0, 0, 1])
    labels = _fixed_prevalence_labels(reference, n_samples=100, seed=42)
    assert labels.shape == (100,)
    assert int(labels.sum()) == 25


def test_load_standard_hdf5_accepts_raw_keys(tmp_path):
    path = tmp_path / "raw.hdf5"
    with h5py.File(path, "w") as hf:
        hf.create_dataset("data", data=np.array([[0, 1, 2]], dtype=np.int8))
        hf.create_dataset("labels", data=np.array([1], dtype=np.int32))

    loaded = load_standard_hdf5(str(path))
    assert loaded.samples.shape == (1, 3)
    assert loaded.labels.tolist() == [1]
