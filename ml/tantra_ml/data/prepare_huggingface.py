from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..languages import Language
from ..text.vocabulary import normalize_text


def stable_id(dataset: str, config: str | None, split: str, index: int) -> str:
    value = f"{dataset}|{config or ''}|{split}|{index}".encode()
    return hashlib.sha256(value).hexdigest()[:20]


def prepare(
    dataset_name: str,
    config_name: str | None,
    split: str,
    output_root: str,
    language: str | None,
    language_field: str | None,
    audio_field: str,
    text_field: str,
    speaker_field: str | None,
    revision: str | None,
    minimum_seconds: float,
    maximum_seconds: float,
    maximum_rows: int | None,
) -> Path:
    from datasets import Audio, load_dataset

    root = Path(output_root)
    audio_root = root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        dataset_name,
        config_name,
        split=split,
        revision=revision,
        trust_remote_code=False,
    )
    dataset = dataset.cast_column(audio_field, Audio(sampling_rate=16_000, decode=True))
    manifest = root / f"{split}.jsonl"
    accepted = 0
    rejected = 0
    with manifest.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            if maximum_rows is not None and accepted >= maximum_rows:
                break
            text = normalize_text(str(row.get(text_field, "")))
            if not text:
                rejected += 1
                continue
            code = language or (str(row.get(language_field, "")) if language_field else "")
            try:
                parsed_language = Language.parse(code)
            except ValueError:
                rejected += 1
                continue
            audio: dict[str, Any] = row[audio_field]
            samples = np.asarray(audio["array"], dtype=np.float32)
            if samples.ndim > 1:
                samples = samples.mean(axis=-1)
            duration = samples.shape[0] / 16_000
            if not minimum_seconds <= duration <= maximum_seconds:
                rejected += 1
                continue
            identifier = stable_id(dataset_name, config_name, split, index)
            audio_path = audio_root / f"{identifier}.flac"
            sf.write(audio_path, samples, 16_000, format="FLAC", subtype="PCM_16")
            value = {
                "audio": str(audio_path.resolve()),
                "text": text,
                "language": parsed_language.name.lower(),
                "speaker_id": str(row.get(speaker_field, "unknown")) if speaker_field else "unknown",
                "duration": duration,
                "source": {
                    "dataset": dataset_name,
                    "config": config_name,
                    "split": split,
                    "revision": revision,
                    "source_index": index,
                },
            }
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            accepted += 1
    summary = {
        "dataset": dataset_name, "config": config_name, "split": split, "revision": revision,
        "accepted": accepted, "rejected": rejected, "manifest": str(manifest),
        "license_review_required": True,
    }
    (root / f"{split}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Hugging Face speech dataset into a TANTRA manifest")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-root", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--language")
    group.add_argument("--language-field")
    parser.add_argument("--audio-field", default="audio")
    parser.add_argument("--text-field", default="transcription")
    parser.add_argument("--speaker-field")
    parser.add_argument("--revision")
    parser.add_argument("--minimum-seconds", type=float, default=0.35)
    parser.add_argument("--maximum-seconds", type=float, default=15.0)
    parser.add_argument("--maximum-rows", type=int)
    args = parser.parse_args()
    print(prepare(
        args.dataset, args.config, args.split, args.output_root, args.language, args.language_field,
        args.audio_field, args.text_field, args.speaker_field, args.revision,
        args.minimum_seconds, args.maximum_seconds, args.maximum_rows,
    ))


if __name__ == "__main__":
    main()
