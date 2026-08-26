from pathlib import Path

from tantra_ml.modelpack import build_pack, verify_pack
from tantra_ml.text.vocabulary import build_vocabulary


def test_pack_round_trip(tmp_path: Path):
    vocabulary = build_vocabulary(["hello", "नमस्ते"])
    vocab_path = tmp_path / "vocab.json"
    vocabulary.save(vocab_path)
    asr = tmp_path / "asr.onnx"
    asr.write_bytes(b"placeholder-model-for-pack-format-test")
    card = tmp_path / "MODEL_CARD.md"
    card.write_text("# Test model\n")
    license_path = tmp_path / "LICENSE"
    license_path.write_text("test-only")
    output = tmp_path / "test.tantra-pack"
    build_pack(output, "test-pack", ["hi", "en"], vocab_path, asr, None, card, [license_path], ["Apache-2.0"])
    manifest = verify_pack(output)
    assert manifest["pack_id"] == "test-pack"
