"""
VAE Reconstruction generator for computing reconstructions without DDPM.

This module provides utilities for generating reconstructions from a trained VAE,
useful for evaluating how well the VAE preserves information from real data
without involving the diffusion model in the generation process.

The reconstructed dataset can then be used to train ML and PRS models to
evaluate whether the VAE preserves the predictive information in the data.
"""

import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class ReconstructionResult:
    """Container for VAE reconstruction results.

    Attributes:
        reconstructions: The reconstructed samples (argmax decoded)
        targets: The original targets (phenotype labels) - real targets since VAE is not conditioned
        orig_samples: The original input samples (optional, for analysis)
        mu: Encoder mean output (optional, for latent space analysis)
        logvar: Encoder log-variance output (optional, for latent space analysis)
        z: Sampled latent vectors (optional, for latent space analysis)
        labels: Binary/categorical labels (optional, for flexible downstream training)
    """
    reconstructions: np.ndarray
    targets: np.ndarray
    orig_samples: Optional[np.ndarray] = None
    mu: Optional[np.ndarray] = None
    logvar: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    eids: Optional[np.ndarray] = None


def decode_first_stage(
    z: torch.Tensor,
    autoencoder,
    scale_factor: float = 1.0,
    disable_first_stage_autocast: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    Decode latent vectors through the VAE decoder.

    Args:
        z: Latent vectors to decode
        autoencoder: The VAE autoencoder model
        scale_factor: Scaling factor applied during encoding (will be inverted)
        disable_first_stage_autocast: Whether to disable autocast
        **kwargs: Additional arguments passed to decoder (e.g., argmax=True)

    Returns:
        Decoded samples
    """
    z = 1.0 / scale_factor * z
    n_samples = z.shape[0]

    n_rounds = math.ceil(z.shape[0] / n_samples)
    all_out = []
    with torch.autocast("cuda", enabled=not disable_first_stage_autocast):
        for n in range(n_rounds):
            out = autoencoder.decode(
                z[n * n_samples : (n + 1) * n_samples], **kwargs
            )
            all_out.append(out)
    out = torch.cat(all_out, dim=0)
    return out


class ReconstructionGenerator:
    """
    Generator for VAE reconstructions without DDPM involvement.

    This class provides utilities for computing reconstructions from a trained VAE,
    saving the results in a format compatible with ML/PRS training pipelines.

    Unlike DDPM-based generation, this directly passes data through the VAE
    encoder-decoder to evaluate information preservation.

    Args:
        model: The VAE training wrapper (with autoencoder attribute)
        device: Device to run on (default: 'cuda')

    Example:
        >>> generator = ReconstructionGenerator(vae_training_wrapper, device='cuda')
        >>> generator.prepare_model()
        >>> result = generator.generate_reconstructions(dataloader)
        >>> save_reconstruction_dataset(result, 'vae_reconstructions.hdf5')
    """

    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device

        # Get the autoencoder from the training wrapper
        if hasattr(model, 'autoencoder'):
            self.autoencoder = model.autoencoder
        elif hasattr(model, 'first_stage_model'):
            self.autoencoder = model.first_stage_model
        else:
            # Assume model is the autoencoder itself
            self.autoencoder = model

    def prepare_model(self):
        """Move model to device and set to eval mode."""
        self.model.to(self.device)
        self.model.eval()

    def generate_reconstructions(
        self,
        dataloader: DataLoader,
        sample_posterior: bool = True,
        store_latents: bool = False,
        store_originals: bool = True,
        onehot: bool = True,
        pad_mask: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> ReconstructionResult:
        """
        Generate reconstructions for a dataset through the VAE.

        Args:
            dataloader: DataLoader containing original data
            sample_posterior: If True, sample from posterior (mu, logvar).
                            If False, use mu directly (deterministic).
            store_latents: Whether to store mu, logvar, and z (increases memory)
            store_originals: Whether to store original samples in result
            onehot: Whether input data is one-hot encoded
            pad_mask: Boolean mask for padded features to remove from output
            verbose: Show progress bar

        Returns:
            ReconstructionResult containing reconstructions and targets
        """
        _reconstructions = []
        _targets = []
        _orig_samples = [] if store_originals else None
        _mu = [] if store_latents else None
        _logvar = [] if store_latents else None
        _z = [] if store_latents else None

        iterator = tqdm(dataloader, desc="Computing reconstructions") if verbose else dataloader

        for batch in iterator:
            with torch.no_grad():
                # Handle both dict and tuple batch formats
                if isinstance(batch, dict):
                    x = batch['x'].to(self.device)
                    y = batch['y']
                else:
                    x, y = batch
                    x = x.to(self.device)

                # Check shape if model has the method
                if hasattr(self.model, 'check_shape'):
                    x, _ = self.model.check_shape(x)

                # Encode: get mu and logvar
                mu, logvar = self.autoencoder.encode(x, sample=False)

                # Sample from posterior or use mean
                if sample_posterior:
                    z = self.autoencoder.reparameterize(mu, logvar)
                else:
                    z = mu

                # Decode
                recons = decode_first_stage(
                    z, self.autoencoder,
                    disable_first_stage_autocast=False,
                    argmax=True
                )

                # Save memory by using int8 for reconstructions
                if not torch.is_floating_point(recons):
                    recons = recons.to(torch.int8)

                _reconstructions.append(recons.detach().cpu())
                _targets.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))

                if store_originals:
                    _orig_samples.append(x.detach().cpu())

                if store_latents:
                    _mu.append(mu.detach().cpu())
                    _logvar.append(logvar.detach().cpu())
                    _z.append(z.detach().cpu())

        # Concatenate all batches
        reconstructions = torch.cat(_reconstructions).numpy()
        targets = torch.cat(_targets).numpy()

        # Process original samples if stored
        orig_samples = None
        if store_originals:
            orig_samples = torch.cat(_orig_samples).numpy()
            if onehot and orig_samples.ndim == 3:
                orig_samples = orig_samples.argmax(axis=1).astype(np.int8)

        # Apply pad mask if provided
        if pad_mask is not None:
            reconstructions = reconstructions[:, ~pad_mask]
            if orig_samples is not None:
                orig_samples = orig_samples[:, ~pad_mask]

        # Process latents
        mu_arr = torch.cat(_mu).numpy() if store_latents else None
        logvar_arr = torch.cat(_logvar).numpy() if store_latents else None
        z_arr = torch.cat(_z).numpy() if store_latents else None

        # Squeeze single-channel latents
        if store_latents and z_arr is not None and z_arr.shape[1] == 1:
            z_arr = np.squeeze(z_arr, axis=1)
            if mu_arr is not None:
                mu_arr = np.squeeze(mu_arr, axis=1)
            if logvar_arr is not None:
                logvar_arr = np.squeeze(logvar_arr, axis=1)

        return ReconstructionResult(
            reconstructions=reconstructions,
            targets=targets,
            orig_samples=orig_samples,
            mu=mu_arr,
            logvar=logvar_arr,
            z=z_arr
        )

    def generate_for_split(
        self,
        dataset,
        split: str = 'train_val',
        batch_size: int = 768,
        num_workers: int = 4,
        **kwargs
    ) -> ReconstructionResult:
        """
        Convenience method to generate reconstructions for a dataset split.

        Args:
            dataset: Dataset object with get_split method
            split: Split name ('train', 'val', 'test', 'train_val', 'full')
            batch_size: Batch size for processing
            num_workers: Number of dataloader workers
            **kwargs: Additional arguments passed to generate_reconstructions

        Returns:
            ReconstructionResult
        """
        from torch.utils.data import TensorDataset

        # Get split data
        if hasattr(dataset, 'get_split'):
            X, y = dataset.get_split(split)
        else:
            print("Warning: Dataset has no get_split method, using full data.")
            X, y = dataset.data, dataset.targets

        # Create simple dataloader
        tensor_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # Check for pad mask
        pad_mask = getattr(dataset, 'pad_mask', None)

        return self.generate_reconstructions(
            dataloader,
            pad_mask=pad_mask,
            **kwargs
        )
