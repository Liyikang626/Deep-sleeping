from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig


class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic sine/cosine positional encoding for Transformer inputs.

    We generate the encoding on the fly so the model can handle variable
    sequence lengths at inference time without needing a fixed max length.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = d_model

    def forward(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype) * (-math.log(10_000.0) / self.d_model)
        )
        encoding = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding.unsqueeze(0)


class SimpleTransformerDiscriminator(nn.Module):
    """
    A small Transformer regressor for non-primary energy ratio prediction.

    Input:
        audio tensor with shape (batch, channels, samples)

    Output:
        tensor with shape (batch,) and values in [0, 1]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.freq_bins = config.n_fft // 2 + 1

        self.input_proj = nn.Linear(self.freq_bins, config.d_model)
        self.position = SinusoidalPositionalEncoding(config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3:
            raise ValueError(f"expected audio shape (batch, channels, samples), got {tuple(audio.shape)}")

        # This baseline averages channels before the STFT. It keeps the model
        # simple while still letting stereo mixtures contribute both channels.
        mono = audio.mean(dim=1)
        window = torch.hann_window(self.config.win_length, device=audio.device, dtype=audio.dtype)
        spec = torch.stft(
            mono,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
            return_complex=True,
        )

        # Shape change:
        #   (batch, freq, frames) -> (batch, frames, freq)
        log_mag = torch.log1p(spec.abs()).transpose(1, 2)
        tokens = self.input_proj(log_mag)
        tokens = tokens + self.position(tokens.shape[1], device=tokens.device, dtype=tokens.dtype)
        encoded = self.encoder(tokens)

        # Mean pooling is enough for this small regression task and keeps the
        # code easy to inspect.
        pooled = encoded.mean(dim=1)
        logits = self.head(pooled).squeeze(-1)
        return torch.sigmoid(logits)
