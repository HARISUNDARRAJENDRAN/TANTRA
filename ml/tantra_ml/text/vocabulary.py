from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import unicodedata
from typing import Iterable, Iterator

import regex

SPECIAL_TOKENS = ("<blank>", "<unk>", "<bos>", "<eos>")
_GRAPHEME = regex.compile(r"\X")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = regex.sub(r"[\p{Cc}\p{Cf}&&[^\n\t]]", "", text)
    text = regex.sub(r"\s+", " ", text).strip()
    return text


def graphemes(text: str) -> Iterator[str]:
    yield from _GRAPHEME.findall(normalize_text(text))


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...]
    unknown_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    def __post_init__(self) -> None:
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Vocabulary has duplicate tokens")
        if not self.tokens or self.tokens[0] != "<blank>":
            raise ValueError("CTC blank must be token zero")

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def encode(self, text: str, boundaries: bool = False) -> list[int]:
        mapping = self.token_to_id
        encoded = [mapping.get(token, self.unknown_id) for token in graphemes(text)]
        if boundaries:
            encoded.insert(0, self.bos_id)
            encoded.append(self.eos_id)
        return encoded

    def decode(self, ids: Iterable[int], strip_special: bool = True) -> str:
        pieces: list[str] = []
        for index in ids:
            if not 0 <= index < len(self.tokens):
                continue
            token = self.tokens[index]
            if strip_special and token.startswith("<"):
                continue
            pieces.append(token)
        return "".join(pieces).replace("▁", " ").strip()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "tokens": list(self.tokens),
                    "unknown_id": self.unknown_id,
                    "bos_id": self.bos_id,
                    "eos_id": self.eos_id,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(value["tokens"]), value.get("unknown_id", 1), value.get("bos_id", 2), value.get("eos_id", 3))


def iter_manifest_text(manifest_paths: Iterable[str | Path]) -> Iterator[str]:
    for manifest_path in manifest_paths:
        with Path(manifest_path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text") or row.get("teacher_text")
                if not isinstance(text, str):
                    raise ValueError(f"{manifest_path}:{line_number} has no text")
                yield text


def build_vocabulary(
    texts: Iterable[str],
    max_tokens: int = 4096,
    minimum_count: int = 1,
) -> Vocabulary:
    if max_tokens < len(SPECIAL_TOKENS) + 2:
        raise ValueError("max_tokens is too small")
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(graphemes(text))
    ordered = sorted(
        (item for item in counts.items() if item[1] >= minimum_count and item[0] not in SPECIAL_TOKENS),
        key=lambda item: (-item[1], item[0]),
    )
    selected = [token for token, _ in ordered[: max_tokens - len(SPECIAL_TOKENS)]]
    if " " not in selected and counts[" "]:
        if len(selected) == max_tokens - len(SPECIAL_TOKENS):
            selected[-1] = " "
        else:
            selected.append(" ")
    return Vocabulary(tuple(SPECIAL_TOKENS + tuple(selected)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a shared Unicode-grapheme vocabulary")
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--minimum-count", type=int, default=1)
    args = parser.parse_args()
    vocab = build_vocabulary(iter_manifest_text(args.manifests), args.max_tokens, args.minimum_count)
    vocab.save(args.output)
    print(json.dumps({"output": args.output, "tokens": len(vocab.tokens)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
