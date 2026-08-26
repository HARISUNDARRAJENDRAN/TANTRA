from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from .languages import LANGUAGE_CODES
from .text.vocabulary import Vocabulary


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pack(
    output: str | Path,
    pack_id: str,
    languages: list[str],
    vocabulary: str | Path,
    asr_model: str | Path,
    tts_model: str | Path | None,
    model_card: str | Path,
    licenses: list[str | Path],
    license_spdx: list[str],
    sample_rate: int = 16_000,
    tts_sample_rate: int = 22_050,
) -> Path:
    if not pack_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in pack_id):
        raise ValueError("pack_id contains unsafe characters")
    invalid = sorted(set(languages) - set(LANGUAGE_CODES))
    if invalid:
        raise ValueError(f"Unsupported language codes: {invalid}")
    if not licenses or not license_spdx:
        raise ValueError("At least one license file and SPDX identifier is required")
    vocabulary = Path(vocabulary)
    asr_model = Path(asr_model)
    model_card = Path(model_card)
    for required in (vocabulary, asr_model, model_card, *(Path(item) for item in licenses)):
        if not required.is_file():
            raise FileNotFoundError(required)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tantra-pack-") as temporary:
        root = Path(temporary)
        shutil.copy2(vocabulary, root / "vocab.json")
        shutil.copy2(asr_model, root / "asr.onnx")
        shutil.copy2(model_card, root / "MODEL_CARD.md")
        license_dir = root / "LICENSES"
        license_dir.mkdir()
        for index, license_path in enumerate(map(Path, licenses), 1):
            shutil.copy2(license_path, license_dir / f"{index:02d}-{license_path.name}")
        files = {"asr.onnx": sha256(root / "asr.onnx")}
        tts_contract = None
        if tts_model:
            shutil.copy2(tts_model, root / "tts.onnx")
            files["tts.onnx"] = sha256(root / "tts.onnx")
            tts_contract = {
                "tokens_input": "tokens", "lengths_input": "token_lengths",
                "language_input": "language_id", "speaker_input": "speaker_id",
                "speed_input": "speed", "audio_output": "audio",
            }
        manifest = {
            "format_version": 1,
            "pack_id": pack_id,
            "languages": languages,
            "sample_rate": sample_rate,
            "tts_sample_rate": tts_sample_rate,
            "vocab_sha256": sha256(root / "vocab.json"),
            "files": files,
            "license_spdx": license_spdx,
            "asr": {
                "samples_input": "samples", "lengths_input": "sample_lengths",
                "language_input": "language_id", "logits_output": "logits", "blank_id": 0,
            },
            "tts": tts_contract,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
    return output


def verify_pack(path: str | Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="tantra-verify-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                target = (root / info.filename).resolve()
                if root.resolve() not in target.parents and target != root.resolve():
                    raise ValueError("Unsafe ZIP member")
                archive.extract(info, root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["format_version"] != 1:
            raise ValueError("Unsupported pack version")
        if sha256(root / "vocab.json") != manifest["vocab_sha256"]:
            raise ValueError("Vocabulary hash mismatch")
        for relative, expected in manifest["files"].items():
            if sha256(root / relative) != expected:
                raise ValueError(f"Hash mismatch: {relative}")
        Vocabulary.load(root / "vocab.json")
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--pack-id", required=True)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--tts-model")
    parser.add_argument("--model-card", required=True)
    parser.add_argument("--license", action="append", required=True)
    parser.add_argument("--license-spdx", action="append", required=True)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--tts-sample-rate", type=int, default=22_050)
    args = parser.parse_args()
    output = build_pack(
        args.output, args.pack_id, args.languages, args.vocabulary, args.asr_model,
        args.tts_model, args.model_card, args.license, args.license_spdx,
        args.sample_rate, args.tts_sample_rate,
    )
    print(json.dumps(verify_pack(output), indent=2))


if __name__ == "__main__":
    main()
