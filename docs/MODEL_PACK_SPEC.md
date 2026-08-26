# `.tantra-pack` model-pack format

A pack is a ZIP archive with no executable code:

```text
manifest.json
vocab.json
asr.onnx
# optional
tts.onnx
speaker_map.json
LICENSES/*
MODEL_CARD.md
```

`manifest.json` contains:

```json
{
  "format_version": 1,
  "pack_id": "tantra-hi-v1",
  "languages": ["hi"],
  "sample_rate": 16000,
  "tts_sample_rate": 22050,
  "vocab_sha256": "...",
  "files": {"asr.onnx": "sha256", "tts.onnx": "sha256"},
  "license_spdx": ["Apache-2.0"],
  "asr": {
    "samples_input": "samples",
    "lengths_input": "sample_lengths",
    "language_input": "language_id",
    "logits_output": "logits",
    "blank_id": 0
  },
  "tts": {
    "tokens_input": "tokens",
    "lengths_input": "token_lengths",
    "language_input": "language_id",
    "speaker_input": "speaker_id",
    "speed_input": "speed",
    "audio_output": "audio"
  }
}
```

The Android importer rejects path traversal, missing license/model-card files, unknown format versions, hash mismatch, and unsupported language identifiers.
