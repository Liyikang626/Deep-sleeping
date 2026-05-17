"""Standalone discriminator package for non-primary energy ratio regression."""

from .config import ModelConfig, TrainConfig
from .model import SimpleTransformerDiscriminator

__all__ = ["ModelConfig", "TrainConfig", "SimpleTransformerDiscriminator"]
