from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

from .languages import Language


@dataclass(frozen=True)
class Utterance:
    audio: Path
    text: str
    language: Language
    speaker_id: str = "unknown"
    duration: float | None = None
    teacher_text: str | None = None
    teacher_confidence: float | None = None
    durations: tuple[int, ...] | None = None

    @property
    def training_text(self) -> str:
        if self.text.strip():
            return self.text
        if self.teacher_text and (self.teacher_confidence or 0.0) >= 0.75:
            return self.teacher_text
        raise ValueError(f"Utterance {self.audio} has no accepted transcript")


def read_manifest(path: str | Path) -> list[Utterance]:
    path = Path(path)
    rows: list[Utterance] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            audio = Path(value["audio"])
            if not audio.is_absolute():
                audio = (path.parent / audio).resolve()
            try:
                row = Utterance(
                    audio=audio,
                    text=str(value.get("text", "")),
                    language=Language.parse(value["language"]),
                    speaker_id=str(value.get("speaker_id", "unknown")),
                    duration=float(value["duration"]) if value.get("duration") is not None else None,
                    teacher_text=value.get("teacher_text"),
                    teacher_confidence=float(value["teacher_confidence"]) if value.get("teacher_confidence") is not None else None,
                    durations=tuple(map(int, value["durations"])) if value.get("durations") is not None else None,
                )
                row.training_text
            except Exception as exc:
                raise ValueError(f"Invalid manifest row {path}:{number}: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def write_manifest(path: str | Path, rows: Iterator[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
