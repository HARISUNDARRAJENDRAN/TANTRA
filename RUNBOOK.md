# TANTRA execution runbook

## 1. Validate source

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e 'protocol/python[dev]'
pip install -e 'ml[dev]'
./scripts/check.sh
```

## 2. Prepare licensed data

Convert each approved corpus to the JSONL schema documented in `docs/DATA_GOVERNANCE.md`, record immutable source revisions, then run:

```bash
python -m tantra_ml.data.validate_manifest --manifest data/train.jsonl
python -m tantra_ml.data.split_manifest --manifest data/all.jsonl --out-dir data/splits
```

Speaker-disjoint splits are mandatory. Human transcripts are preferred. Teacher-generated labels must retain provenance and confidence.

## 3. Train ASR on Modal

After authenticating Modal using a fresh token stored outside the repository:

```bash
modal run ml/modal_app.py --action gpu-smoke
modal run ml/modal_app.py --action train-asr --config ml/configs/mobile-small.yaml
modal run ml/modal_app.py --action export-asr --config ml/configs/mobile-small.yaml
modal run ml/modal_app.py --action benchmark-asr --config ml/configs/mobile-small.yaml
```

## 4. Train TTS

```bash
modal run ml/modal_app.py --action align-tts --config ml/configs/tts-baseline.yaml
modal run ml/modal_app.py --action train-tts --config ml/configs/tts-baseline.yaml
modal run ml/modal_app.py --action export-tts --config ml/configs/tts-baseline.yaml
```

## 5. Build and verify a model pack

```bash
python -m tantra_ml.pack.build --help
python -m tantra_ml.pack.verify path/to/model.tantra-pack
```

A pack is rejected by Android when any declared hash, vocabulary ABI, language map, ONNX signature, model card, or license file is missing.

## 6. Run two-phone loop

Follow `docs/ANDROID_DEPLOYMENT.md`, then export telemetry for the evaluation harness in `docs/EVALUATION.md`.
