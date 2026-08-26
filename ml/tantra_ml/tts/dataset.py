from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import Dataset
import torchaudio

from ..manifest import Utterance
from ..text.vocabulary import Vocabulary, normalize_text


@dataclass(frozen=True)
class TtsDataConfig:
    sample_rate: int = 22_050
    mel_bins: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    max_seconds: float = 15.0


class TtsDataset(Dataset[dict]):
    def __init__(
        self,
        utterances: Sequence[Utterance],
        vocabulary: Vocabulary,
        speaker_map: dict[str, int],
        config: TtsDataConfig = TtsDataConfig(),
    ) -> None:
        self.rows = list(utterances)
        self.vocabulary = vocabulary
        self.speaker_map = speaker_map
        self.config = config
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.mel_bins,
            power=1.0,
            center=True,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        if row.durations is None:
            raise ValueError(f"TTS row lacks token durations: {row.audio}")
        waveform, source_rate = torchaudio.load(str(row.audio))
        waveform = waveform.mean(dim=0)
        if source_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, self.config.sample_rate)
        maximum = int(self.config.max_seconds * self.config.sample_rate)
        if waveform.numel() > maximum:
            waveform = waveform[:maximum]
        tokens = torch.tensor(self.vocabulary.encode(normalize_text(row.training_text), boundaries=True), dtype=torch.long)
        durations = torch.tensor(row.durations, dtype=torch.long)
        if durations.numel() != tokens.numel():
            raise ValueError(
                f"Duration count {durations.numel()} != token count {tokens.numel()} for {row.audio}"
            )
        mel = torch.log(self.mel(waveform).clamp_min(1e-5)).transpose(0, 1)
        difference = int(mel.shape[0] - durations.sum())
        if difference:
            durations[-1] = max(1, int(durations[-1]) + difference)
        target_samples = int(durations.sum()) * self.config.hop_length
        waveform = torch.nn.functional.pad(waveform, (0, max(0, target_samples - waveform.numel())))[:target_samples]
        return {
            "tokens": tokens,
            "durations": durations,
            "mel": mel,
            "audio": waveform,
            "language": int(row.language),
            "speaker": self.speaker_map[row.speaker_id],
            "text": row.training_text,
        }


def build_speaker_map(rows: Sequence[Utterance]) -> dict[str, int]:
    counts = Counter(row.speaker_id for row in rows)
    return {speaker: index for index, (speaker, _) in enumerate(sorted(counts.items()))}


def collate_tts(batch: Sequence[dict]) -> dict:
    token_lengths = torch.tensor([item["tokens"].numel() for item in batch], dtype=torch.long)
    frame_lengths = torch.tensor([item["mel"].shape[0] for item in batch], dtype=torch.long)
    audio_lengths = torch.tensor([item["audio"].numel() for item in batch], dtype=torch.long)
    return {
        "tokens": torch.nn.utils.rnn.pad_sequence([item["tokens"] for item in batch], batch_first=True),
        "token_lengths": token_lengths,
        "durations": torch.nn.utils.rnn.pad_sequence([item["durations"] for item in batch], batch_first=True),
        "mel": torch.nn.utils.rnn.pad_sequence([item["mel"] for item in batch], batch_first=True),
        "frame_lengths": frame_lengths,
        "audio": torch.nn.utils.rnn.pad_sequence([item["audio"] for item in batch], batch_first=True),
        "audio_lengths": audio_lengths,
        "language_id": torch.tensor([item["language"] for item in batch], dtype=torch.long),
        "speaker_id": torch.tensor([item["speaker"] for item in batch], dtype=torch.long),
        "texts": [item["text"] for item in batch],
    }
