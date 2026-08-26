# TANTRA ML

This directory trains and exports the models consumed by the Android model-pack ABI.

## Data manifest

One JSON object per line:

```json
{"audio":"/data/audio/example.wav","text":"मदद चाहिए","language":"hi","speaker_id":"speaker-17","duration":1.42}
```

For sequence-level distillation, an unlabeled row may provide `teacher_text` and `teacher_confidence`; pseudo-labels below the configured acceptance threshold are rejected. TTS rows additionally include per-token `durations`, generated with the shared-vocabulary CTC Viterbi aligner.

## ASR

`TantraAsr` is a compact raw-waveform CTC Conformer. The raw waveform front end is part of ONNX, avoiding a second Android feature-extraction implementation. Language-balanced sampling prevents Hindi/English volume from dominating lower-resource languages.

```bash
pip install -e '.[train,dev]'
tantra-build-vocab /data/manifests/train.jsonl --output /data/manifests/vocab.json
tantra-train-asr --config configs/baseline.yaml
tantra-export-asr --checkpoint /artifacts/asr-runs/<id>/best.pt \
  --output /artifacts/asr.onnx --quantized-output /artifacts/asr-int8.onnx
tantra-eval-asr --model /artifacts/asr-int8.onnx --manifest /data/manifests/test.jsonl \
  --vocabulary /data/manifests/vocab.json --output /artifacts/asr-report.json
```

## TTS

`TantraTts` is a compact non-autoregressive multilingual acoustic model with predicted durations and a small neural vocoder. It shares the exact grapheme vocabulary and language IDs with ASR/AksharaLink. CTC forced alignment creates duration supervision without a proprietary aligner.

The class and export contract are implemented; high-quality training still requires licensed multi-speaker recordings, filtering, duration manifests, and measured listening tests. Do not label an unvalidated checkpoint as production quality.

## Modal

`modal_app.py` defines GPU smoke, ASR training, export, and benchmark jobs with persistent `tantra-data` and `tantra-artifacts` volumes. Configure credentials outside the repository.
