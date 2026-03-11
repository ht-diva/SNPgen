"""
Main generator class for synthetic SNP dataset generation.
"""

import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .label_samplers import LabelSampler


@dataclass
class GenerationResult:
    """Container for generation results."""
    targets: np.ndarray
    samples: np.ndarray
    # Additional metadata fields for syn_recon mode
    orig_samples: Optional[np.ndarray] = None
    mu: Optional[np.ndarray] = None
    logvar: Optional[np.ndarray] = None
    z_recon: Optional[np.ndarray] = None
    z_recon_scaled: Optional[np.ndarray] = None
    reconstructions: Optional[np.ndarray] = None
    z_syn: Optional[np.ndarray] = None
    z_recon_ddpm: Optional[np.ndarray] = None
    reconstructions_ddpm: Optional[np.ndarray] = None


class SyntheticDatasetGenerator:
    """
    Generator for synthetic SNP datasets from trained DDPM models.

    This class provides a unified interface for generating synthetic datasets
    with different conditioning strategies (complete, syn_recon, augmented).

    Args:
        model: The DDPM training wrapper (DiffusionEngine)
        config: OmegaConf configuration object
        decoder_config: Decoder configuration dict (for z_shape)
        device: Device to run generation on (default: 'cuda')

    Example:
        >>> generator = SyntheticDatasetGenerator(ddpm_model, config, decoder_config)
        >>> result = generator.generate_complete(dataloader, batch_size=2048)
        >>> save_synthetic_dataset(result, 'syn_complete.hdf5')
    """

    def __init__(
        self,
        model,
        config,
        decoder_config: Dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.decoder_config = decoder_config
        self.device = device

        # Compute z_shape from decoder config
        self.z_shape = (decoder_config['z_channels'], decoder_config['z_dim'])

        # Setup UCG keys for conditioning
        self._setup_ucg_keys()

    def _setup_ucg_keys(self):
        """Setup unconditional guidance keys from model conditioner."""
        self.conditioner_input_keys = [
            e.input_key for e in self.model.conditioner.embedders
        ]
        self.ucg_keys = self.conditioner_input_keys

    def _get_denoiser(self, sampling_kwargs: Dict = None):
        """Create denoiser function for sampling."""
        sampling_kwargs = sampling_kwargs or {}
        return lambda input, sigma, c: self.model.denoiser(
            self.model.model, input, sigma, c, **sampling_kwargs
        )

    def _compute_conditioning(
        self,
        batch: Dict[str, torch.Tensor],
        batch_size: int
    ) -> Tuple[Dict, Dict]:
        """Compute conditional and unconditional embeddings."""
        c, uc = self.model.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=self.ucg_keys
            if len(self.model.conditioner.embedders) > 0
            else [],
        )

        for k in c:
            if k != "crossattn":
                c[k], uc[k] = map(
                    lambda y: y[k][:batch_size].to(self.device),
                    (c, uc)
                )

        return c, uc

    def _generate_batch(
        self,
        batch: Dict[str, torch.Tensor],
        sampling_kwargs: Dict = None,
        return_latent: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generate samples for a single batch."""
        sampling_kwargs = sampling_kwargs or {}
        batch_size = batch['x'].shape[0] if 'x' in batch else batch['y'].shape[0]

        # Move batch to device
        # TODO: possible optimization: move only necessary keys (or do not move 'x' if not used)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Compute conditioning
        c, uc = self._compute_conditioning(batch, batch_size)

        # Generate synthetic samples
        z_syn = self.model.sample(
            c, shape=self.z_shape, uc=uc, batch_size=batch_size, **sampling_kwargs
        )
        samples = self.model.decode_first_stage(z_syn, argmax=True)

        if return_latent:
            return samples, z_syn
        return samples, None

    def prepare_model(self):
        """Prepare model for inference (move to device and eval mode)."""
        self.model.to(self.device)
        self.model.eval()

    def generate_complete(
        self,
        dataloader: DataLoader,
        sampling_kwargs: Dict = None,
        verbose: bool = True
    ) -> GenerationResult:
        """
        Generate synthetic dataset with same size and labels as the original.

        Args:
            dataloader: DataLoader with original data
            sampling_kwargs: Additional kwargs for sampling
            verbose: Whether to show progress bar

        Returns:
            GenerationResult with targets and samples
        """
        sampling_kwargs = sampling_kwargs or {}

        _targets = []
        _syn_samples = []

        iterator = tqdm(dataloader, desc="Generating complete dataset") if verbose else dataloader

        for _batch in iterator:
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in _batch.items()}

                samples, _ = self._generate_batch(batch, sampling_kwargs)

                _targets.append(batch['y'].detach().cpu())
                _syn_samples.append(samples.detach().cpu())

        targets = torch.cat(_targets).numpy()
        syn_samples = torch.cat(_syn_samples).numpy()

        return GenerationResult(targets=targets, samples=syn_samples)

    def generate_syn_recon(
        self,
        dataloader: DataLoader,
        sampling_kwargs: Dict = None,
        onehot: bool = True,
        verbose: bool = True
    ) -> GenerationResult:
        """
        Generate synthetic dataset with reconstructions and latent space info.

        This mode generates synthetic samples and also computes reconstructions
        through the VAE encoder-decoder pipeline, useful for analysis.

        Args:
            dataloader: DataLoader with original data
            sampling_kwargs: Additional kwargs for sampling
            onehot: Whether original data is one-hot encoded
            verbose: Whether to show progress bar

        Returns:
            GenerationResult with full reconstruction metadata
        """
        sampling_kwargs = sampling_kwargs or {}
        denoiser = self._get_denoiser(sampling_kwargs)

        _orig_samples = []
        _targets = []
        _mu = []
        _logvar = []
        _z_recon = []
        _z_recon_scaled = []
        _reconstructions = []
        _z_syn = []
        _syn_samples = []
        _z_recon_ddpm = []
        _reconstructions_ddpm = []

        iterator = tqdm(dataloader, desc="Generating syn_recon dataset") if verbose else dataloader

        for _batch in iterator:
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in _batch.items()}
                batch_size = batch['x'].shape[0]

                # Compute conditioning
                c, uc = self._compute_conditioning(batch, batch_size)

                # Compute reconstructions: encoder -> decoder

                #z = ddpm_training_wrapper.encode_first_stage(x, sample=True) # -> we could use this but we also want mu and logvar
                
                x = self.model.get_input(batch)
                mu, logvar = self.model.first_stage_model.encode(x, sample=False)
                z = self.model.first_stage_model.reparameterize(mu, logvar)
                z_scaled = self.model.scale_factor * z

                reconstructions = self.model.decode_first_stage(z, argmax=True)

                # Generate synthetic samples
                z_syn = self.model.sample(
                    c, shape=self.z_shape, uc=uc, batch_size=batch_size, **sampling_kwargs
                )
                samples = self.model.decode_first_stage(z_syn, argmax=True)

                # Generate reconstructions: encoder -> ddpm -> decoder
                z_recon_ddpm = self.model.sampler(denoiser, z_scaled, c, uc=uc)
                reconstructions_ddpm = self.model.decode_first_stage(z_recon_ddpm, argmax=True)

                _orig_samples.append(batch['x'].detach().cpu())
                _targets.append(batch['y'].detach().cpu())
                _z_syn.append(z_syn.detach().cpu())
                _syn_samples.append(samples.detach().cpu())
                _mu.append(mu.detach().cpu())
                _logvar.append(logvar.detach().cpu())
                _z_recon.append(z.detach().cpu())
                if self.model.scale_factor != 1:
                    _z_recon_scaled.append(z_scaled.detach().cpu())
                _reconstructions.append(reconstructions.detach().cpu())
                _z_recon_ddpm.append(z_recon_ddpm.detach().cpu())
                _reconstructions_ddpm.append(reconstructions_ddpm.detach().cpu())

        # Concatenate results
        orig_samples = torch.cat(_orig_samples).numpy()
        if onehot:
            orig_samples = np.argmax(orig_samples, 1)

        targets = torch.cat(_targets).numpy()
        z_syn = torch.cat(_z_syn).numpy()
        syn_samples = torch.cat(_syn_samples).numpy()
        mu = torch.cat(_mu).numpy()
        logvar = torch.cat(_logvar).numpy()
        z_recon = torch.cat(_z_recon).numpy()
        z_recon_scaled = torch.cat(_z_recon_scaled).numpy() if _z_recon_scaled else None
        reconstructions = torch.cat(_reconstructions).numpy()
        z_recon_ddpm = torch.cat(_z_recon_ddpm).numpy()
        reconstructions_ddpm = torch.cat(_reconstructions_ddpm).numpy()

        # Squeeze z arrays if they have single channel
        if z_syn.shape[1] == 1:
            z_syn = np.squeeze(z_syn, axis=1)
            z_recon = np.squeeze(z_recon, axis=1)
            z_recon_ddpm = np.squeeze(z_recon_ddpm, axis=1)
            if z_recon_scaled is not None and z_recon_scaled.size > 0:
                z_recon_scaled = np.squeeze(z_recon_scaled, axis=1)

        return GenerationResult(
            targets=targets,
            samples=syn_samples,
            orig_samples=orig_samples,
            mu=mu,
            logvar=logvar,
            z_recon=z_recon,
            z_recon_scaled=z_recon_scaled,
            reconstructions=reconstructions,
            z_syn=z_syn,
            z_recon_ddpm=z_recon_ddpm,
            reconstructions_ddpm=reconstructions_ddpm
        )

    def generate_augmented(
        self,
        label_sampler: LabelSampler,
        total_samples: int,
        batch_size: int,
        seq_len: int,
        sampling_kwargs: Dict = None,
        verbose: bool = True
    ) -> GenerationResult:
        """
        Generate augmented synthetic dataset with custom label distribution.

        Args:
            label_sampler: LabelSampler instance that generates labels
            total_samples: Total number of samples to generate
            batch_size: Batch size for generation
            seq_len: Sequence length for dummy input
            sampling_kwargs: Additional kwargs for sampling
            verbose: Whether to show progress bar

        Returns:
            GenerationResult with targets and samples
        """
        sampling_kwargs = sampling_kwargs or {}

        # Generate all labels
        aug_labels = label_sampler.sample(total_samples)

        if verbose:
            print(f"Generated labels stats: min={aug_labels.min():.3f}, max={aug_labels.max():.3f}, "
                  f"mean={aug_labels.mean():.3f}, std={aug_labels.std():.3f}")

        _targets = []
        _syn_samples = []

        num_batches = math.ceil(len(aug_labels) / batch_size)
        label_idx = 0

        iterator = range(num_batches)
        if verbose:
            iterator = tqdm(iterator, desc=f"Generating augmented dataset ({label_sampler.name})")

        for i in iterator:
            with torch.no_grad():
                current_batch_size = min(batch_size, len(aug_labels) - label_idx)
                batch_labels = aug_labels[label_idx:label_idx + current_batch_size]
                label_idx += current_batch_size

                # Create batch with sampled labels
                batch = {
                    'x': torch.zeros((current_batch_size, seq_len)), # Dummy input
                    'y': torch.from_numpy(batch_labels).to(label_sampler.dtype)
                }

                samples, _ = self._generate_batch(batch, sampling_kwargs)

                _targets.append(batch['y'].detach().cpu())
                _syn_samples.append(samples.detach().cpu())

        targets = torch.cat(_targets).numpy()
        syn_samples = torch.cat(_syn_samples).numpy()

        return GenerationResult(
            targets=targets,
            samples=syn_samples,
        )

    def generate_from_labels(
        self,
        labels: np.ndarray,
        batch_size: int,
        seq_len: int,
        sampling_kwargs: Dict = None,
        verbose: bool = True
    ) -> GenerationResult:
        """
        Generate synthetic samples from explicit labels.

        Args:
            labels: Array of labels to condition on
            batch_size: Batch size for generation
            seq_len: Sequence length for dummy input
            sampling_kwargs: Additional kwargs for sampling
            verbose: Whether to show progress bar

        Returns:
            GenerationResult with targets and samples
        """
        sampling_kwargs = sampling_kwargs or {}
        labels = np.asarray(labels, dtype=np.int64)

        _targets = []
        _syn_samples = []

        num_batches = math.ceil(len(labels) / batch_size)
        label_idx = 0

        iterator = range(num_batches)
        if verbose:
            iterator = tqdm(iterator, desc="Generating from labels")

        for i in iterator:
            with torch.no_grad():
                current_batch_size = min(batch_size, len(labels) - label_idx)
                batch_labels = labels[label_idx:label_idx + current_batch_size]
                label_idx += current_batch_size

                dtype = torch.long
                batch = {
                    'x': torch.zeros((current_batch_size, seq_len)), # Dummy input
                    'y': torch.from_numpy(batch_labels).to(dtype)
                }

                samples, _ = self._generate_batch(batch, sampling_kwargs)

                _targets.append(batch['y'].detach().cpu())
                _syn_samples.append(samples.detach().cpu())

        targets = torch.cat(_targets).numpy()
        syn_samples = torch.cat(_syn_samples).numpy()

        return GenerationResult(
            targets=targets,
            samples=syn_samples,
        )
