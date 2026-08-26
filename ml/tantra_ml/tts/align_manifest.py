from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .ctc_alignment import ctc_viterbi_alignment, token_durations_from_path
from ..manifest import read_manifest
from ..text.vocabulary import Vocabulary


def linear_resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32)
    target_size = round(len(samples) * target_rate / source_rate)
    return np.interp(
        np.linspace(0, len(samples), target_size, endpoint=False),
        np.arange(len(samples)),
        samples,
    ).astype(np.float32)


def align_manifest(asr_model: str, source_manifest: str, vocabulary_path: str, output_manifest: str,
                   tts_sample_rate: int = 22_050, tts_hop: int = 256) -> None:
    import onnxruntime as ort

    vocabulary = Vocabulary.load(vocabulary_path)
    session = ort.InferenceSession(asr_model, providers=["CPUExecutionProvider"])
    output = Path(output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in read_manifest(source_manifest):
            samples, rate = sf.read(row.audio, dtype="float32", always_2d=False)
            if samples.ndim == 2:
                samples = samples.mean(axis=1)
            samples = linear_resample(samples, rate, 16_000)
            logits = session.run(["logits"], {
                "samples": samples[None, :],
                "sample_lengths": np.array([len(samples)], dtype=np.int64),
                "language_id": np.array([int(row.language)], dtype=np.int64),
            })[0][0]
            targets = torch.tensor(vocabulary.encode(row.training_text, boundaries=True), dtype=torch.long)
            path = ctc_viterbi_alignment(torch.from_numpy(logits).log_softmax(-1), targets)
            asr_durations = token_durations_from_path(path, targets.numel()).float()
            total_tts_frames = round(len(samples) / 16_000 * tts_sample_rate / tts_hop)
            scaled = torch.round(asr_durations / asr_durations.sum() * total_tts_frames).long().clamp_min(1)
            difference = total_tts_frames - int(scaled.sum())
            scaled[-1] = max(1, int(scaled[-1]) + difference)
            value = {
                "audio": str(row.audio), "text": row.training_text,
                "language": row.language.name.lower(), "speaker_id": row.speaker_id,
                "duration": row.duration, "durations": scaled.tolist(),
            }
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tts-sample-rate", type=int, default=22_050)
    parser.add_argument("--tts-hop", type=int, default=256)
    args = parser.parse_args()
    align_manifest(args.asr_model, args.manifest, args.vocabulary, args.output, args.tts_sample_rate, args.tts_hop)


if __name__ == "__main__":
    main()
