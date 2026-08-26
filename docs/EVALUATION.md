# Evaluation protocol

## Principle

Every number is tied to a model hash, app commit, device, Android version, language, dataset/noise condition, and run count. Results are never copied from upstream model cards.

## Accuracy (40%)

### STT

Report word error rate and character error rate per language on:

1. clean read speech;
2. conversational speech;
3. 5, 10, and 20 dB SNR mixtures;
4. device-recorded far-field speech; and
5. code-switched utterances where available.

Normalization is language-specific and version-controlled. Keep raw and normalized hypotheses. Aggregate with both macro average (equal language weight) and micro average (utterance weight).

### TTS

Use randomized, blinded human tests for intelligibility and naturalness. At minimum record word transcription accuracy, 1–5 naturalness MOS, and alert comprehension in noise. Objective measures such as speaker-independent ASR round-trip WER, DNSMOS, or MCD are diagnostics, not replacements for human listening.

## Efficiency (20%)

Measure:

- APK/AAB size excluding and including each model pack;
- model pack size and active resident model bytes;
- peak and steady-state RSS;
- idle-listening CPU after a five-minute warm-up;
- active ASR and TTS CPU;
- battery drain over a 30-minute scripted session;
- thermal throttling state.

Use Android Studio profiler, `dumpsys meminfo`, `simpleperf`, and Perfetto. Record at least one low-range and one mid-range device.

## Latency (20%)

- audio chunk arrival to partial transcript;
- detected end-of-speech to final transcript;
- packet enqueue to peer receive;
- peer receive to first synthesized sample;
- TTS real-time factor;
- sender speech end to receiver audio start.

Clock offset is estimated during session handshake with repeated ping/pong samples. End-to-end values carry uncertainty; same-device loopback is measured separately.

## Robustness and product behavior (remaining score)

Test Bluetooth reconnects, Wi-Fi handoff, duplicate/lost/corrupt frames, language changes, screen-off foreground operation, incoming phone calls, DND policy, low storage, low battery, and model-pack hash failures.

## Result schema

Each run writes the JSON schema in `benchmarks/result.schema.json`; CI validates it before accepting a benchmark report.
