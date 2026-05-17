from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig, dataclass_to_dict, project_root
from .dataset import GeneratedMixDataset, discover_project_records, split_records
from .model import SimpleTransformerDiscriminator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Transformer discriminator for non-primary energy ratio regression.")
    parser.add_argument("--dataset-root", default=None, help="Path to the generated dataset root.")
    parser.add_argument("--output-dir", default="runs", help="Directory where checkpoints and config snapshots are written.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Python and PyTorch.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--segment-seconds", type=float, default=6.0, help="Length of each training crop.")
    parser.add_argument("--train-split", type=float, default=0.8, help="Project-level train split.")
    parser.add_argument("--samples-per-epoch", type=int, default=256, help="Number of random training crops per epoch.")
    parser.add_argument("--val-samples", type=int, default=64, help="Number of validation crops per epoch.")
    parser.add_argument("--device", default="", help="Training device, e.g. cuda or cpu. Empty means auto-detect.")
    parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--disable-cache-audio", action="store_true", help="Disable in-memory WAV caching.")
    parser.add_argument("--log-every", type=int, default=10, help="Print every N optimizer steps.")

    parser.add_argument("--sample-rate", type=int, default=44_100, help="Expected WAV sample rate.")
    parser.add_argument("--n-fft", type=int, default=1_024, help="STFT n_fft.")
    parser.add_argument("--hop-length", type=int, default=256, help="STFT hop length.")
    parser.add_argument("--win-length", type=int, default=1_024, help="STFT window length.")
    parser.add_argument("--d-model", type=int, default=192, help="Transformer hidden size.")
    parser.add_argument("--num-heads", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer encoder depth.")
    parser.add_argument("--feedforward-dim", type=int, default=384, help="Transformer feed-forward width.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    return parser


def parse_configs(args: argparse.Namespace) -> tuple[TrainConfig, ModelConfig]:
    train_config = TrainConfig(
        dataset_root=args.dataset_root or TrainConfig().dataset_root,
        output_dir=args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        num_workers=args.num_workers,
        segment_seconds=args.segment_seconds,
        train_split=args.train_split,
        samples_per_epoch=args.samples_per_epoch,
        val_samples=args.val_samples,
        device=args.device,
        amp=not args.disable_amp,
        log_every=args.log_every,
        cache_audio=not args.disable_cache_audio,
    )
    model_config = ModelConfig(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    )
    return train_config, model_config


def main() -> None:
    args = build_parser().parse_args()
    train_config, model_config = parse_configs(args)
    train(train_config, model_config)


def train(train_config: TrainConfig, model_config: ModelConfig) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)

    device = torch.device(train_config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = (project_root() / train_config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_project_records(train_config.dataset_root)
    train_records, val_records = split_records(records, train_config.train_split, train_config.seed)

    train_dataset = GeneratedMixDataset(
        records=train_records,
        sample_rate=model_config.sample_rate,
        segment_seconds=train_config.segment_seconds,
        samples_per_epoch=train_config.samples_per_epoch,
        random_crop=True,
        cache_audio=train_config.cache_audio,
    )
    val_dataset = GeneratedMixDataset(
        records=val_records,
        sample_rate=model_config.sample_rate,
        segment_seconds=train_config.segment_seconds,
        samples_per_epoch=train_config.val_samples,
        random_crop=False,
        cache_audio=train_config.cache_audio,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SimpleTransformerDiscriminator(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    use_amp = train_config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    config_snapshot = {
        "train_config": dataclass_to_dict(train_config),
        "model_config": dataclass_to_dict(model_config),
        "train_records": [item.project_id for item in train_records],
        "val_records": [item.project_id for item in val_records],
    }
    (output_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    print(f"device={device}")
    print(f"dataset_root={Path(train_config.dataset_root).resolve()}")
    print(f"train_projects={len(train_records)} val_projects={len(val_records)}")
    print(f"output_dir={output_dir}")

    best_val_mae = float("inf")
    global_step = 0
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_signed_error_sum = 0.0
        train_samples = 0

        for mixture, target in train_loader:
            mixture = mixture.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                prediction = model(mixture)
                residual = target - prediction

                # The user explicitly asked for "ground-truth minus output".
                # A raw signed mean can cancel out positive and negative errors,
                # so we keep that residual but optimize its absolute value.
                loss = residual.abs().mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            batch_size = mixture.shape[0]
            mae = residual.abs().detach()
            signed_error = residual.detach()
            train_loss_sum += float(loss.detach().cpu()) * batch_size
            train_mae_sum += float(mae.mean().cpu()) * batch_size
            train_signed_error_sum += float(signed_error.mean().cpu()) * batch_size
            train_samples += batch_size
            global_step += 1

            if global_step % train_config.log_every == 0:
                print(
                    f"epoch={epoch:03d} step={global_step:05d} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"pred_mean={float(prediction.detach().mean().cpu()):.4f} "
                    f"target_mean={float(target.detach().mean().cpu()):.4f}"
                )

        train_metrics = {
            "loss": train_loss_sum / max(train_samples, 1),
            "mae": train_mae_sum / max(train_samples, 1),
            "signed_error": train_signed_error_sum / max(train_samples, 1),
        }
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_mae={train_metrics['mae']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} "
            f"val_signed_error={val_metrics['signed_error']:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_config": dataclass_to_dict(train_config),
            "model_config": dataclass_to_dict(model_config),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            torch.save(checkpoint, output_dir / "best.pt")


@torch.no_grad()
def evaluate(model: SimpleTransformerDiscriminator, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    mae_sum = 0.0
    signed_error_sum = 0.0
    samples = 0

    for mixture, target in loader:
        mixture = mixture.to(device)
        target = target.to(device)
        prediction = model(mixture)
        residual = target - prediction
        loss = residual.abs().mean()

        batch_size = mixture.shape[0]
        loss_sum += float(loss.detach().cpu()) * batch_size
        mae_sum += float(residual.abs().mean().cpu()) * batch_size
        signed_error_sum += float(residual.mean().cpu()) * batch_size
        samples += batch_size

    return {
        "loss": loss_sum / max(samples, 1),
        "mae": mae_sum / max(samples, 1),
        "signed_error": signed_error_sum / max(samples, 1),
    }


if __name__ == "__main__":
    main()
