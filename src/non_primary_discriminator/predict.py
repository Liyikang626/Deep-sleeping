from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .audio import load_wav
from .config import ModelConfig
from .model import SimpleTransformerDiscriminator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict the non-primary energy ratio of one mixture WAV file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint such as best.pt.")
    parser.add_argument("--mixture", required=True, help="Path to the mixture WAV file.")
    parser.add_argument("--device", default="", help="Inference device, e.g. cuda or cpu. Empty means auto-detect.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])

    model = SimpleTransformerDiscriminator(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    mixture_path = Path(args.mixture).resolve()
    audio = load_wav(mixture_path, expected_sample_rate=model_config.sample_rate).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(audio).item()

    print(f"mixture={mixture_path}")
    print(f"predicted_non_primary_ratio={prediction:.6f}")
    print(f"predicted_non_primary_percent={prediction * 100.0:.2f}%")


if __name__ == "__main__":
    main()
