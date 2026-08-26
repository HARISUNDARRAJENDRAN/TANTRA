from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TtsConfig:
    vocab_size: int
    language_count: int = 11
    speaker_count: int = 1
    hidden: int = 192
    mel_bins: int = 80
    encoder_layers: int = 4
    decoder_layers: int = 4
    heads: int = 4
    max_token_duration: int = 32
    vocoder_channels: int = 256
    hop_length: int = 256
    sample_rate: int = 22_050

    def to_dict(self) -> dict:
        return asdict(self)


class DurationPredictor(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
        )
        self.projection = nn.Linear(hidden, 1)

    def forward(self, value: Tensor) -> Tensor:
        first = self.net[0](value.transpose(1, 2)).transpose(1, 2)
        first = self.net[1](first)
        first = self.net[2](first)
        second = self.net[3](first.transpose(1, 2)).transpose(1, 2)
        second = self.net[4](second)
        return self.projection(second).squeeze(-1)


class ResidualVocoderBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.first = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.second = nn.Conv1d(channels, channels, 1)

    def forward(self, value: Tensor) -> Tensor:
        return value + self.second(F.leaky_relu(self.first(value), 0.1))


class TinyVocoder(nn.Module):
    def __init__(self, mel_bins: int, channels: int) -> None:
        super().__init__()
        self.pre = nn.Conv1d(mel_bins, channels, 7, padding=3)
        factors = (8, 8, 2, 2)
        stages: list[nn.Module] = []
        current = channels
        for factor in factors:
            next_channels = max(32, current // 2)
            stages.append(nn.ConvTranspose1d(current, next_channels, factor * 2, stride=factor, padding=factor // 2))
            stages.append(ResidualVocoderBlock(next_channels, 1))
            stages.append(ResidualVocoderBlock(next_channels, 3))
            current = next_channels
        self.stages = nn.Sequential(*stages)
        self.post = nn.Conv1d(current, 1, 7, padding=3)

    def forward(self, mel: Tensor) -> Tensor:
        value = self.pre(mel.transpose(1, 2))
        for layer in self.stages:
            value = F.leaky_relu(layer(value), 0.1) if isinstance(layer, nn.ConvTranspose1d) else layer(value)
        return torch.tanh(self.post(F.leaky_relu(value, 0.1))).squeeze(1)


class TantraTts(nn.Module):
    """Compact non-autoregressive multilingual acoustic model plus neural vocoder.

    Ground-truth token durations are used during training. Inference predicts durations,
    length-regulates hidden states, and emits waveform samples in one graph.
    """

    def __init__(self, config: TtsConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden)
        self.language_embedding = nn.Embedding(config.language_count, config.hidden)
        self.speaker_embedding = nn.Embedding(config.speaker_count, config.hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            config.hidden, config.heads, config.hidden * 4, 0.1, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.encoder_layers)
        self.duration = DurationPredictor(config.hidden)
        decoder_layer = nn.TransformerEncoderLayer(
            config.hidden, config.heads, config.hidden * 4, 0.1, batch_first=True, activation="gelu", norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, config.decoder_layers)
        self.mel_projection = nn.Linear(config.hidden, config.mel_bins)
        self.vocoder = TinyVocoder(config.mel_bins, config.vocoder_channels)

    def encode(self, tokens: Tensor, language_id: Tensor, speaker_id: Tensor) -> Tensor:
        value = self.token_embedding(tokens)
        value = value + self.language_embedding(language_id).unsqueeze(1)
        value = value + self.speaker_embedding(speaker_id).unsqueeze(1)
        return self.encoder(value)

    def predict_durations(self, encoded: Tensor, speed: Tensor) -> tuple[Tensor, Tensor]:
        log_duration = self.duration(encoded)
        duration = (torch.exp(log_duration) - 1.0).clamp(1.0, float(self.config.max_token_duration))
        duration = torch.round(duration / speed.view(-1, 1).clamp(0.5, 2.0)).to(torch.long)
        return log_duration, duration

    @staticmethod
    def length_regulate(encoded: Tensor, durations: Tensor) -> tuple[Tensor, Tensor]:
        expanded = [torch.repeat_interleave(row, repeat, dim=0) for row, repeat in zip(encoded, durations, strict=True)]
        lengths = torch.tensor([row.shape[0] for row in expanded], device=encoded.device, dtype=torch.long)
        return torch.nn.utils.rnn.pad_sequence(expanded, batch_first=True), lengths

    def acoustic(self, tokens: Tensor, language_id: Tensor, speaker_id: Tensor, speed: Tensor, durations: Tensor | None = None):
        encoded = self.encode(tokens, language_id, speaker_id)
        log_duration, predicted = self.predict_durations(encoded, speed)
        regulated, frame_lengths = self.length_regulate(encoded, predicted if durations is None else durations)
        mel = self.mel_projection(self.decoder(regulated))
        return mel, log_duration, frame_lengths

    def forward(
        self,
        tokens: Tensor,
        token_lengths: Tensor,
        language_id: Tensor,
        speaker_id: Tensor,
        speed: Tensor,
    ) -> Tensor:
        del token_lengths
        mel, _, _ = self.acoustic(tokens, language_id, speaker_id, speed)
        return self.vocoder(mel)

    def forward_train(
        self,
        tokens: Tensor,
        language_id: Tensor,
        speaker_id: Tensor,
        durations: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        speed = torch.ones(tokens.shape[0], device=tokens.device)
        mel, log_duration, _ = self.acoustic(tokens, language_id, speaker_id, speed, durations)
        audio = self.vocoder(mel)
        return mel, log_duration, audio
