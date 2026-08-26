from __future__ import annotations

from enum import IntEnum


class Language(IntEnum):
    UNKNOWN = 0
    HI = 1
    GU = 2
    MR = 3
    KN = 4
    ML = 5
    TA = 6
    TE = 7
    OR = 8
    BN = 9
    EN = 10

    @classmethod
    def parse(cls, value: str | int) -> "Language":
        if isinstance(value, int):
            return cls(value)
        normalized = value.lower().replace("_", "-").split("-", 1)[0]
        if normalized == "od":
            normalized = "or"
        try:
            return cls[normalized.upper()]
        except KeyError as exc:
            raise ValueError(f"Unsupported language: {value}") from exc


LANGUAGE_CODES = tuple(language.name.lower() for language in Language if language is not Language.UNKNOWN)
