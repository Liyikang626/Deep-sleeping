from __future__ import annotations

import wave
from pathlib import Path
from typing import Iterable

import torch


def load_wav(path: Path, expected_sample_rate: int | None = None) -> torch.Tensor:
    """
    Load a 16-bit PCM WAV file into a float tensor shaped as (channels, samples).

    The deep_sleeping generator writes standard PCM WAV files, so the standard
    library is enough here and we do not need an extra dependency such as
    torchaudio or librosa.
    """

    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
        raise ValueError(f"{path} sample rate {sample_rate} != expected {expected_sample_rate}")
    if sample_width != 2:
        raise ValueError(f"{path} uses {sample_width * 8}-bit samples; only 16-bit PCM is supported")

    audio = torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768.0
    return audio.view(-1, channels).transpose(0, 1).contiguous()


def resolve_generated_path(dataset_root: Path, raw_path: str) -> Path:
    """
    Resolve artifact paths stored in metadata.json.

    Metadata entries look like `generated/project_000000/mixture.wav`.
    If `dataset_root` already points at the `generated/` directory, then
    `dataset_root / raw_path` would duplicate the folder name. We therefore
    try a small set of common resolutions and keep the first existing file.
    """

    path = Path(raw_path)
    candidates = [
        dataset_root / path,
        dataset_root.parent / path,
        dataset_root / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not resolve metadata path: {raw_path}")


def match_channels(audio: torch.Tensor, channels: int) -> torch.Tensor:
    """Convert mono to stereo or stereo to mono when needed."""

    if audio.shape[0] == channels:
        return audio
    if audio.shape[0] == 1:
        return audio.expand(channels, -1)
    if channels == 1:
        return audio.mean(dim=0, keepdim=True)
    raise ValueError(f"cannot convert {audio.shape[0]} channels to {channels}")


def crop_or_pad(audio: torch.Tensor, target_samples: int, start: int = 0) -> torch.Tensor:
    """
    Return a fixed-length segment from `audio`.

    If the clip is longer than the target, we crop.
    If the clip is shorter, we pad with zeros at the end.
    """

    if target_samples <= 0:
        raise ValueError("target_samples must be positive")
    if audio.shape[-1] >= target_samples:
        return audio[..., start : start + target_samples].contiguous()

    padded = torch.zeros(audio.shape[0], target_samples, dtype=audio.dtype)
    padded[..., : audio.shape[-1]] = audio
    return padded


def align_to_shortest(items: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    """Trim a list of audio tensors so they all share the shortest length."""

    tensors = list(items)
    if not tensors:
        raise ValueError("expected at least one tensor")
    shortest = min(item.shape[-1] for item in tensors)
    return [item[..., :shortest].contiguous() for item in tensors]


def non_primary_energy_ratio(stems: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the label used by the discriminator.

    The ratio is:
        (sum(all stem energies) - max(single stem energy)) / sum(all stem energies)

    This means:
    - 0.0  => one stem fully dominates the segment
    - 1.0  => no single stem dominates, the segment is highly mixed
    """

    if stems.ndim != 3:
        raise ValueError(f"expected stems shape (num_stems, channels, samples), got {tuple(stems.shape)}")

    stem_energy = stems.pow(2).sum(dim=(1, 2))
    total_energy = stem_energy.sum().clamp_min(eps)
    primary_energy = stem_energy.max()
    return ((total_energy - primary_energy) / total_energy).clamp(0.0, 1.0)
