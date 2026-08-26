# TANTRA project report

TANTRA is an offline, low-bitrate speech transceiver for ten Indian languages. It avoids transmitting audio. A sender performs local streaming recognition, commits only stable text prefixes, encodes the shared Indic grapheme IDs into compact delta packets, and sends them over Wi-Fi, Bluetooth, or a transparent embedded relay. A receiver begins local synthesis at clause boundaries. This makes intelligible voice communication possible even when the link cannot carry a conventional audio stream.

## Implemented engineering surface

- Android application shell, foreground microphone runtime, PTT and continuous modes.
- Adaptive endpoint detection, platform AEC/NS/AGC integration, streaming ASR/TTS interfaces.
- ONNX Runtime model-pack ABI with integrity, provenance, license, and compatibility validation.
- Binary AksharaLink framing, CRC, sequencing, acknowledgements, retries, alert priority, stable-prefix and token-delta coding.
- LAN TCP, Bluetooth RFCOMM, loopback transport, and ESP32 pass-through bridge.
- Compact multilingual CTC-Conformer ASR training/export/evaluation pipeline.
- Compact non-autoregressive multilingual TTS training/alignment/export/evaluation pipeline.
- Modal GPU functions with persistent volumes and explicit train/export/benchmark actions.
- Protocol and model unit tests, Android unit tests, CI, security checks, and evaluation schemas.

## Honest completion boundary

The repository is a complete reproducible implementation baseline, but a competition-ready claim requires trained model weights and physical-device measurements on properly licensed corpora. Model accuracy, naturalness, memory, CPU, battery, and end-to-end latency must be measured rather than inferred. The release gate is defined in `docs/EVALUATION.md`.
