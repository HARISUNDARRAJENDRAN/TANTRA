# Project status

## Implemented in this checkpoint

- Novel end-to-end architecture and packet protocol.
- Python reference protocol with CRC validation, token varints, incremental prefix commits, acknowledgements, and corruption tests.
- Android application shell, microphone/VAD endpointing, foreground session service, ONNX ASR/TTS interfaces, pack validation/import, LAN and Bluetooth transports, PTT/continuous modes, alert-priority playback, and telemetry schema.
- Trainable raw-waveform multilingual CTC student and shared grapheme vocabulary tooling.
- ONNX export, dynamic int8 quantization, pack builder, WER/CER and RTF benchmark entry points.
- Modal GPU job definitions using persistent volumes.
- CI and reproducible documentation.

## Artifact gate before claiming competition readiness

A release is competition-ready only after all ten language packs pass the device matrix and the measured benchmark JSON files are committed. This repository deliberately does not label untrained or unmeasured models as complete.
