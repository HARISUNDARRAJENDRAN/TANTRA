# Research and design notes

## Sources inspected

- Modal's low-latency voice-bot engineering article for concurrent pipeline structure, streaming boundaries, and latency measurement.
- AI4Bharat multilingual ASR and TTS model families as teacher/baseline candidates.
- sherpa-onnx and ONNX Runtime Mobile for offline Android deployment patterns.
- Whisper/whisper.cpp as a compact multilingual baseline.
- Mozilla Common Voice and FLEURS as reproducible evaluation sources where their release terms permit the intended use.
- Android `AudioRecord`, audio effects, foreground-service, Bluetooth, Wi-Fi, audio focus, and alarm behavior documentation.

Exact upstream revisions and licenses belong in `THIRD_PARTY_MODELS.md` when weights are selected. Public availability is not treated as permission to redistribute.

## Decision matrix

| Option | Coverage | Mobile footprint | Streaming | Main risk | Decision |
|---|---|---:|---|---|---|
| Large multilingual Whisper/MMS | broad | high | limited/adapter dependent | low-end CPU/RAM | teacher/baseline only |
| Per-language large IndicConformer | strong Indic accuracy | high across 10 packs | good variants exist | storage | teacher |
| Shared compact CTC student | all required languages | controllable | sliding/stateful | must train/distill | primary ASR target |
| Ten independent neural TTS voices | straightforward | high storage | clause-level | pack management | baseline |
| Shared multilingual TTS + adapters | compact active RAM | controllable | clause-level | training complexity | primary TTS target |
| Send compressed audio | language independent | link still kilobits/s | excellent | violates core low-bitrate advantage | fallback only |
| Send token deltas | bytes/s to low hundreds | tiny | excellent | ASR errors become semantic errors | primary radio payload |

## Engineering interpretation of “fully offline”

Training and model-pack preparation may use Internet-connected GPU infrastructure. The submitted Android runtime performs capture, ASR, framing, transport, TTS, and playback without an Internet service. Model packs are bundled or side-loaded before deployment.
