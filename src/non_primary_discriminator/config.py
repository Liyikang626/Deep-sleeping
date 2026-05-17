from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the root of the standalone discriminator project."""

    return Path(__file__).resolve().parents[2]


def default_dataset_root() -> str:
    """
    Return the local default dataset path.

    The user asked to train on:
    C:\\Users\\fison\\Desktop\\class\\DL\\discriminator\\deep_sleeping\\dataset generator\\generated
    so we resolve that location relative to the current workspace.
    """

    return str(project_root().parent / "deep_sleeping" / "dataset generator" / "generated")


@dataclass
class ModelConfig:
    """
    Hyper-parameters for the audio Transformer.

    The model turns the mixture into a log-magnitude STFT sequence and uses
    a small Transformer encoder to regress the non-primary energy ratio.
    """

    sample_rate: int = 44_100
    n_fft: int = 1_024
    hop_length: int = 256
    win_length: int = 1_024
    d_model: int = 192
    num_heads: int = 4
    num_layers: int = 4
    feedforward_dim: int = 384
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """
    Training configuration for the standalone discriminator.

    `samples_per_epoch` controls how many random crops are drawn per epoch.
    This is useful when the dataset has only a few long songs, because one
    song can yield many training segments.
    """

    dataset_root: str = field(default_factory=default_dataset_root)
    output_dir: str = "runs"
    seed: int = 42
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    segment_seconds: float = 6.0
    train_split: float = 0.8
    samples_per_epoch: int = 256
    val_samples: int = 64
    device: str = ""
    amp: bool = True
    log_every: int = 10
    cache_audio: bool = True


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    """Convert a dataclass into a plain dictionary for checkpoint metadata."""

    return asdict(value)
