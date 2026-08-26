from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ..languages import LANGUAGE_CODES
from ..text.vocabulary import normalize_text


def validate(named_paths: dict[str, str], require_all_languages: bool = True) -> dict:
    seen_audio: dict[str, str] = {}
    seen_speakers: dict[tuple[str, str], str] = {}
    report = {}
    errors: list[str] = []
    for split, path_value in named_paths.items():
        counts = Counter(); seconds = Counter(); speakers = set()
        with Path(path_value).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                row = json.loads(line)
                language = str(row.get("language", ""))
                audio = Path(row.get("audio", ""))
                text = normalize_text(str(row.get("text") or row.get("teacher_text") or ""))
                if language not in LANGUAGE_CODES:
                    errors.append(f"{split}:{number}: unsupported language {language}")
                if not audio.is_file():
                    errors.append(f"{split}:{number}: missing audio {audio}")
                    continue
                if not text:
                    errors.append(f"{split}:{number}: empty transcript")
                digest = hashlib.sha256(audio.read_bytes()).hexdigest()
                if digest in seen_audio and seen_audio[digest] != split:
                    errors.append(f"audio leakage: {audio} appears in {seen_audio[digest]} and {split}")
                seen_audio[digest] = split
                speaker = str(row.get("speaker_id", "unknown"))
                key = (language, speaker)
                if speaker != "unknown" and key in seen_speakers and seen_speakers[key] != split:
                    errors.append(f"speaker leakage: {language}/{speaker} in {seen_speakers[key]} and {split}")
                seen_speakers[key] = split
                counts[language] += 1
                seconds[language] += float(row.get("duration", 0.0))
                speakers.add(key)
        report[split] = {
            "utterances": sum(counts.values()), "hours": sum(seconds.values()) / 3600,
            "languages": dict(counts), "speakers": len(speakers),
        }
        if require_all_languages:
            missing = sorted(set(LANGUAGE_CODES) - set(counts))
            if missing:
                errors.append(f"{split}: missing languages {missing}")
    report["errors"] = errors
    if errors:
        raise ValueError("\n".join(errors[:100]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True); parser.add_argument("--dev", required=True); parser.add_argument("--test", required=True)
    parser.add_argument("--allow-missing-languages", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate({"train": args.train, "dev": args.dev, "test": args.test}, not args.allow_missing_languages), indent=2))


if __name__ == "__main__":
    main()
