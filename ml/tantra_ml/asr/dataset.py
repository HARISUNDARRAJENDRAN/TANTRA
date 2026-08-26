from __future__ import annotations

from collections import Counter
import random
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler
import torchaudio

from ..manifest import Utterance
from ..text.vocabulary import Vocabulary, normalize_text


class AsrDataset(Dataset[dict]):
    def __init__(
        self,
        utterances: Sequence[Utterance],
        vocabulary: Vocabulary,
        sample_rate: int = 16_000,
        max_seconds: float = 15.0,
        training: bool = True,
    ) -> None:
        self.utterances = list(utterances)
        self.vocabulary = vocabulary
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self.training = training

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, index: int) -> dict:
        row = self.utterances[index]
        waveform, source_rate = torchaudio.load(str(row.audio))
        waveform = waveform.mean(dim=0)
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, self.sample_rate)
        if waveform.numel() > self.max_samples:
            raise ValueError(
                f"Utterance exceeds {self.max_samples / self.sample_rate:.1f}s; segment it during data preparation: {row.audio}"
            )
        if self.training:
            gain = 10 ** (random.uniform(-3.0, 3.0) / 20.0)
            waveform = (waveform * gain).clamp(-1.0, 1.0)
        tokens = torch.tensor(self.vocabulary.encode(normalize_text(row.training_text)), dtype=torch.long)
        if tokens.numel() == 0:
            raise ValueError(f"Transcript became empty: {row.audio}")
        return {
            "samples": waveform.float(),
            "tokens": tokens,
            "language": int(row.language),
            "text": normalize_text(row.training_text),
            "audio": str(row.audio),
        }


def collate_asr(batch: Sequence[dict]) -> dict:
    sample_lengths = torch.tensor([item["samples"].numel() for item in batch], dtype=torch.long)
    token_lengths = torch.tensor([item["tokens"].numel() for item in batch], dtype=torch.long)
    samples = torch.nn.utils.rnn.pad_sequence([item["samples"] for item in batch], batch_first=True)
    tokens = torch.cat([item["tokens"] for item in batch])
    return {
        "samples": samples,
        "sample_lengths": sample_lengths,
        "tokens": tokens,
        "token_lengths": token_lengths,
        "language_id": torch.tensor([item["language"] for item in batch], dtype=torch.long),
        "texts": [item["text"] for item in batch],
        "audio": [item["audio"] for item in batch],
    }


def language_balanced_sampler(rows: Sequence[Utterance]) -> WeightedRandomSampler:
    counts = Counter(row.language for row in rows)
    weights = [1.0 / counts[row.language] for row in rows]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
