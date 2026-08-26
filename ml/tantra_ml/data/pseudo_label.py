from __future__ import annotations

import argparse
import json
from pathlib import Path


def pseudo_label(
    source_manifest: str,
    output_manifest: str,
    model_id: str,
    revision: str | None,
    assumed_confidence: float,
    trust_remote_code: bool,
    batch_size: int,
) -> None:
    import torch
    from transformers import pipeline

    recognizer = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        revision=revision,
        device=0 if torch.cuda.is_available() else -1,
        trust_remote_code=trust_remote_code,
    )
    rows = [json.loads(line) for line in Path(source_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    output = Path(output_manifest); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            predictions = recognizer([row["audio"] for row in batch], batch_size=batch_size)
            if isinstance(predictions, dict):
                predictions = [predictions]
            for row, prediction in zip(batch, predictions, strict=True):
                row["teacher_text"] = str(prediction["text"]).strip()
                row["teacher_confidence"] = assumed_confidence
                row["teacher"] = {"model": model_id, "revision": revision, "trust_remote_code": trust_remote_code}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequence-level ASR distillation with explicit teacher provenance")
    parser.add_argument("--manifest", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True); parser.add_argument("--revision")
    parser.add_argument("--assumed-confidence", type=float, required=True,
                        help="Explicit calibration value; never silently infer confidence from pipeline text")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.assumed_confidence <= 1:
        raise SystemExit("Confidence must be between zero and one")
    pseudo_label(args.manifest, args.output, args.model, args.revision, args.assumed_confidence,
                 args.trust_remote_code, args.batch_size)


if __name__ == "__main__":
    main()
