from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def split(rows: list[dict], seed: int, dev_fraction: float, test_fraction: float) -> dict[str, list[dict]]:
    if dev_fraction + test_fraction >= 0.5:
        raise ValueError("Dev + test fraction is unexpectedly high")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        speaker = str(row.get("speaker_id") or hashlib.sha256(str(row["audio"]).encode()).hexdigest())
        groups[(str(row["language"]), speaker)].append(row)
    by_language: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for (language, speaker), items in groups.items():
        by_language[language].append((speaker, items))
    result = {"train": [], "dev": [], "test": []}
    generator = random.Random(seed)
    for language, speakers in sorted(by_language.items()):
        generator.shuffle(speakers)
        total = sum(len(items) for _, items in speakers)
        target_dev = max(1, round(total * dev_fraction))
        target_test = max(1, round(total * test_fraction))
        counts = {"dev": 0, "test": 0}
        for _, items in speakers:
            if counts["test"] < target_test:
                target = "test"
            elif counts["dev"] < target_dev:
                target = "dev"
            else:
                target = "train"
            result[target].extend(items)
            if target in counts:
                counts[target] += len(items)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--dev-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    result = split(load_rows(args.manifests), args.seed, args.dev_fraction, args.test_fraction)
    for name, rows in result.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({name: len(rows) for name, rows in result.items()}, indent=2))


if __name__ == "__main__":
    main()
