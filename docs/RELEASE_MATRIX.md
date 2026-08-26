# Competition release matrix

A release may be called **competition-ready** only after every required row is backed by raw artifacts and exact hashes.

| Area | Required evidence | Pass rule |
|---|---|---|
| Language coverage | 10 speaker-disjoint test manifests | Every required language present; no train/test speaker overlap |
| STT accuracy | WER and CER by language, gender where licensed, clean/noisy, low/mid device | Pre-declared per-language threshold met; no macro-average hiding a failed language |
| TTS intelligibility | Human transcription accuracy and listening ratings | Blind evaluators; at least 20 speakers/listeners across language groups |
| TTS flow | MOS-style naturalness and pause/prosody rubric | Confidence intervals reported; alert voice tested separately |
| ASR latency | first partial and endpoint p50/p90/p95/p99 | Measured on physical low- and mid-range phones |
| TTS latency | receive-to-first-audio and RTF percentiles | RTF below 1.0 for every required device/language |
| Full loop | sender speech-end to remote first audio | Corrected clock method from `TELEMETRY.md` |
| Efficiency | APK, each model pack, RSS/PSS, idle/listening CPU, battery drain | Exact device/Android/build hashes included |
| Link efficiency | bytes per utterance, bytes/s, retransmits under controlled loss | Compare against Opus baselines at multiple bitrates |
| Offline guarantee | airplane-mode two-phone test | No network permission path or remote inference dependency |
| Licensing | source, revision, license and attribution inventory | Every data/model component redistributable for the intended submission |
| Security | malformed pack, CRC corruption, replay/session tests | No crash, path traversal, silent model substitution, or stale-alert replay |

Target numbers in configuration or documentation are design goals. Only results under `benchmarks/device-runs/<release-id>/` count as measurements.
