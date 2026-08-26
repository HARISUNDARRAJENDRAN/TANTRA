from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

from ..languages import Language
from ..text.vocabulary import Vocabulary


def evaluate(model_path: str, vocabulary_path: str, prompts_path: str, output_dir: str, sample_rate: int = 22_050) -> dict:
    vocabulary = Vocabulary.load(vocabulary_path)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    prompts = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, prompt in enumerate(prompts):
        language = Language.parse(prompt["language"])
        token_ids = np.array([vocabulary.encode(prompt["text"], boundaries=True)], dtype=np.int64)
        started = time.perf_counter()
        audio = session.run(["audio"], {
            "tokens": token_ids,
            "token_lengths": np.array([token_ids.shape[1]], dtype=np.int64),
            "language_id": np.array([int(language)], dtype=np.int64),
            "speaker_id": np.array([int(prompt.get("speaker_id", 0))], dtype=np.int64),
            "speed": np.array([float(prompt.get("speed", 1.0))], dtype=np.float32),
        })[0][0]
        elapsed = time.perf_counter() - started
        path = root / f"{index:04d}-{language.name.lower()}.wav"
        sf.write(path, audio, sample_rate)
        duration = len(audio) / sample_rate
        rows.append({
            "id": index, "language": language.name.lower(), "text": prompt["text"], "audio": str(path),
            "latency_ms": elapsed * 1000, "duration_seconds": duration, "rtf": elapsed / max(duration, 1e-6),
            "intelligibility_transcription": "", "naturalness_mos_1_to_5": "", "reviewer": "",
        })
    with (root / "listening-sheet.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    report = {"utterances": len(rows), "rtf_p50": float(np.percentile([row["rtf"] for row in rows], 50)),
              "rtf_p95": float(np.percentile([row["rtf"] for row in rows], 95))}
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--prompts", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=22_050)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model, args.vocabulary, args.prompts, args.output_dir, args.sample_rate), indent=2))


if __name__ == "__main__":
    main()
