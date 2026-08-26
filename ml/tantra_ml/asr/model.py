from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AsrConfig:
    vocab_size: int
    language_count: int = 11
    model_dim: int = 256
    layers: int = 8
    heads: int = 4
    feed_forward_dim: int = 1024
    convolution_kernel: int = 15
    dropout: float = 0.1
    conv_channels: tuple[int, ...] = (64, 128, 192, 256)
    conv_kernels: tuple[int, ...] = (10, 8, 4, 4)
    conv_strides: tuple[int, ...] = (5, 4, 2, 2)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["conv_channels"] = list(self.conv_channels)
        value["conv_kernels"] = list(self.conv_kernels)
        value["conv_strides"] = list(self.conv_strides)
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "AsrConfig":
        value = dict(value)
        for key in ("conv_channels", "conv_kernels", "conv_strides"):
            if key in value:
                value[key] = tuple(value[key])
        return cls(**value)


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class ConvolutionModule(nn.Module):
    def __init__(self, dim: int, kernel: int, dropout: float) -> None:
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("Conformer convolution kernel must be odd")
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, dim * 2, 1)
        self.depthwise = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise_out = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        value = self.norm(value).transpose(1, 2)
        value = F.glu(self.pointwise_in(value), dim=1)
        value = F.silu(self.batch_norm(self.depthwise(value)))
        return self.dropout(self.pointwise_out(value)).transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, hidden, dropout)
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.convolution = ConvolutionModule(dim, kernel, dropout)
        self.ffn2 = FeedForwardModule(dim, hidden, dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, value: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        value = value + 0.5 * self.ffn1(value)
        normalized = self.attention_norm(value)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        value = value + self.attention_dropout(attended)
        value = value + self.convolution(value)
        value = value + 0.5 * self.ffn2(value)
        return self.final_norm(value)


class ConvFeatureEncoder(nn.Module):
    def __init__(self, config: AsrConfig) -> None:
        super().__init__()
        channels = (1,) + config.conv_channels
        layers: list[nn.Module] = []
        for index, (kernel, stride) in enumerate(zip(config.conv_kernels, config.conv_strides, strict=True)):
            layers += [
                nn.Conv1d(channels[index], channels[index + 1], kernel, stride=stride, bias=False),
                nn.GroupNorm(1, channels[index + 1]),
                nn.SiLU(),
            ]
        self.net = nn.Sequential(*layers)
        self.projection = nn.Linear(config.conv_channels[-1], config.model_dim)
        self.kernels = config.conv_kernels
        self.strides = config.conv_strides

    def output_lengths(self, lengths: Tensor) -> Tensor:
        for kernel, stride in zip(self.kernels, self.strides, strict=True):
            lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
        return lengths.clamp_min(1)

    def forward(self, samples: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        value = self.net(samples.unsqueeze(1)).transpose(1, 2)
        return self.projection(value), self.output_lengths(lengths)


class SinusoidalPosition(nn.Module):
    def __init__(self, dim: int, maximum: int = 8192) -> None:
        super().__init__()
        positions = torch.arange(maximum, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / dim))
        table = torch.zeros(maximum, dim)
        table[:, 0::2] = torch.sin(positions * divisor)
        table[:, 1::2] = torch.cos(positions * divisor)
        self.register_buffer("table", table, persistent=False)

    def forward(self, value: Tensor) -> Tensor:
        return value + self.table[: value.shape[1]].unsqueeze(0).to(value.dtype)


class TantraAsr(nn.Module):
    """Compact raw-waveform multilingual CTC model with shared language conditioning."""

    def __init__(self, config: AsrConfig) -> None:
        super().__init__()
        self.config = config
        self.features = ConvFeatureEncoder(config)
        self.language_embedding = nn.Embedding(config.language_count, config.model_dim)
        self.position = SinusoidalPosition(config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            ConformerBlock(
                config.model_dim,
                config.heads,
                config.feed_forward_dim,
                config.convolution_kernel,
                config.dropout,
            )
            for _ in range(config.layers)
        )
        self.classifier = nn.Linear(config.model_dim, config.vocab_size)

    def forward(self, samples: Tensor, sample_lengths: Tensor, language_id: Tensor) -> tuple[Tensor, Tensor]:
        value, output_lengths = self.features(samples, sample_lengths)
        value = value + self.language_embedding(language_id).unsqueeze(1)
        value = self.dropout(self.position(value))
        positions = torch.arange(value.shape[1], device=value.device).unsqueeze(0)
        padding_mask = positions >= output_lengths.unsqueeze(1)
        for block in self.blocks:
            value = block(value, padding_mask)
        return self.classifier(value), output_lengths

    def log_probabilities(self, samples: Tensor, sample_lengths: Tensor, language_id: Tensor) -> tuple[Tensor, Tensor]:
        logits, lengths = self(samples, sample_lengths, language_id)
        return logits.log_softmax(dim=-1), lengths

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
