from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .audio import align_to_shortest, crop_or_pad, load_wav, match_channels, non_primary_energy_ratio, resolve_generated_path


@dataclass
class ProjectRecord:
    """One generated project: one mixture plus a list of rendered stems."""

    project_id: str
    mixture_path: Path
    stem_paths: list[Path]


def discover_project_records(dataset_root: str | Path) -> list[ProjectRecord]:
    """
    Discover generated projects from `manifest.jsonl` or from project folders.

    Using the manifest is nice when it exists because it records which projects
    actually finished rendering.
    """

    root = Path(dataset_root).resolve()
    manifest = root / "manifest.jsonl"

    project_ids: list[str] = []
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "rendered" and record.get("project_id"):
                project_ids.append(record["project_id"])
    else:
        project_ids = [path.name for path in sorted(root.glob("project_*")) if path.is_dir()]

    records: list[ProjectRecord] = []
    for project_id in project_ids:
        metadata_path = root / project_id / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        render = metadata["render"]
        mixture_path = resolve_generated_path(root, render["mixture_path"])
        stem_paths = [resolve_generated_path(root, item["stem_path"]) for item in render["stems"]]
        if stem_paths:
            records.append(ProjectRecord(project_id=project_id, mixture_path=mixture_path, stem_paths=stem_paths))

    if not records:
        raise FileNotFoundError(f"no rendered projects found under {root}")
    return records


def split_records(records: list[ProjectRecord], train_split: float, seed: int) -> tuple[list[ProjectRecord], list[ProjectRecord]]:
    """
    Create a reproducible train/validation split at the project level.

    Splitting by project instead of by crop avoids leaking nearly identical
    waveform segments from the same song into both training and validation.
    """

    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * train_split))
    cut = min(cut, len(shuffled) - 1) if len(shuffled) > 1 else 1
    train_records = shuffled[:cut]
    val_records = shuffled[cut:] if len(shuffled) > 1 else shuffled[:]
    return train_records, val_records


class GeneratedMixDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Dataset of mixture segments paired with non-primary energy ratio labels.

    Each item returns:
    - mixture segment: (channels, samples)
    - ratio label: scalar tensor in [0, 1]
    """

    def __init__(
        self,
        records: list[ProjectRecord],
        sample_rate: int,
        segment_seconds: float,
        samples_per_epoch: int,
        random_crop: bool,
        cache_audio: bool = True,
    ) -> None:
        if not records:
            raise ValueError("records must not be empty")
        if segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")

        self.records = records
        self.sample_rate = sample_rate
        self.segment_samples = int(round(segment_seconds * sample_rate))
        self.samples_per_epoch = samples_per_epoch
        self.random_crop = random_crop
        self.cache_audio = cache_audio
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index % len(self.records)]
        mixture, stems = self._load_project(record)

        total_samples = mixture.shape[-1]
        if total_samples > self.segment_samples:
            max_start = total_samples - self.segment_samples
            start = random.randint(0, max_start) if self.random_crop else max_start // 2
        else:
            start = 0

        mixture_segment = crop_or_pad(mixture, self.segment_samples, start=start)
        stem_segments = torch.stack([crop_or_pad(stem, self.segment_samples, start=start) for stem in stems], dim=0)
        target_ratio = non_primary_energy_ratio(stem_segments)
        return mixture_segment.float(), target_ratio.float()

    def _load_project(self, record: ProjectRecord) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_audio and record.project_id in self._cache:
            return self._cache[record.project_id]

        mixture = load_wav(record.mixture_path, expected_sample_rate=self.sample_rate)
        stems = [load_wav(path, expected_sample_rate=self.sample_rate) for path in record.stem_paths]

        # We align every signal to the shortest available length so that one
        # broken stem cannot shift the labels relative to the mixture.
        aligned = align_to_shortest([mixture, *stems])
        mixture = aligned[0]
        channels = mixture.shape[0]
        aligned_stems = torch.stack([match_channels(item, channels) for item in aligned[1:]], dim=0)

        if self.cache_audio:
            self._cache[record.project_id] = (mixture, aligned_stems)
        return mixture, aligned_stems
