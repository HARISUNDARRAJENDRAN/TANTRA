from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from ..languages import Language
from ..manifest import read_manifest
from ..text.vocabulary import Vocabulary, normalize_text


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int = 16_000) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32)
    duration = len(samples) / source_rate
    source_x = np.linspace(0.0, duration, len(samples), endpoint=False)
    target_count = round(duration * target_rate)
    target_x = np.linspace(0.0, duration, target_count, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def greedy_decode(logits: np.ndarray, vocabulary: Vocabulary, blank_id: int = 0) -> str:
    ids: list[int] = []
    previous = -1
    for token in logits.argmax(axis=-1).tolist():
        if token != blank_id and token != previous:
            ids.append(token)
        previous = token
    return vocabulary.decode(ids)


def evaluate(model_path: str | Path, manifest_path: str | Path, vocabulary_path: str | Path, output: str | Path) -> dict:
    import onnxruntime as ort
    from jiwer import cer, wer

    vocabulary = Vocabulary.load(vocabulary_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    rows = read_manifest(manifest_path)
    per_language: dict[str, list[tuple[str, str, float, float]]] = defaultdict(list)
    all_rows: list[dict] = []
    for row in rows:
        samples, rate = sf.read(row.audio, dtype="float32", always_2d=False)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = _resample_linear(samples, rate)
        started = time.perf_counter()
        logits = session.run(
            ["logits"],
            {
                "samples": samples[None, :],
                "sample_lengths": np.array([samples.shape[0]], dtype=np.int64),
                "language_id": np.array([int(row.language)], dtype=np.int64),
            },
        )[0][0]
        elapsed = time.perf_counter() - started
        reference = normalize_text(row.training_text)
        hypothesis = normalize_text(greedy_decode(logits, vocabulary))
        duration = samples.shape[0] / 16_000
        item = {
            "audio": str(row.audio),
            "language": row.language.name.lower(),
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": wer(reference, hypothesis),
            "cer": cer(reference, hypothesis),
            "latency_ms": elapsed * 1000,
            "rtf": elapsed / max(duration, 1e-6),
        }
        all_rows.append(item)
        per_language[item["language"]].append((reference, hypothesis, item["latency_ms"], item["rtf"]))
    summary = {}
    for language, values in per_language.items():
        references = [value[0] for value in values]
        hypotheses = [value[1] for value in values]
        summary[language] = {
            "utterances": len(values),
            "wer": wer(references, hypotheses),
            "cer": cer(references, hypotheses),
            "latency_p50_ms": float(np.percentile([value[2] for value in values], 50)),
            "latency_p95_ms": float(np.percentile([value[2] for value in values], 95)),
            "rtf_p50": float(np.percentile([value[3] for value in values], 50)),
            "rtf_p95": float(np.percentile([value[3] for value in values], 95)),
        }
    report = {"model": str(model_path), "manifest": str(manifest_path), "languages": summary, "utterances": all_rows}
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate(args.model, args.manifest, args.vocabulary, args.output)
    print(json.dumps(report["languages"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
