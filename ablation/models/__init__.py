"""Model targets for ablation configs."""

from .conditional_vae import (
    ConditionalAutoencoder,
    ConditionalAutoencoderDiscriminatorTrainingWrapper,
    ConditionalAutoencoderTrainingWrapper,
)
from .wgan_gp import ConditionalWGAN_GPModule
from .crbm import ConditionalCRBMModule

__all__ = [
    "ConditionalAutoencoder",
    "ConditionalAutoencoderDiscriminatorTrainingWrapper",
    "ConditionalAutoencoderTrainingWrapper",
    "ConditionalWGAN_GPModule",
    "ConditionalCRBMModule",
]
