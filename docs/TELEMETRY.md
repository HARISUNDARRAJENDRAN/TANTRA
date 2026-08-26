# Telemetry and clock-correct latency measurement

TANTRA uses monotonic clocks for every duration. Wall-clock timestamps are metadata only and must never be subtracted for latency calculations.

## Event chain

Each utterance receives a random 64-bit `utterance_id`. Implementations emit these events:

1. `speech_start` — endpoint detector enters speech.
2. `audio_frame_captured` — optional sampled event, not one event per frame in production.
3. `asr_partial` — decoder emits a hypothesis.
4. `stable_prefix_commit` — text becomes immutable and eligible for transmission.
5. `speech_end` — endpoint detector closes the utterance.
6. `frame_tx` — AksharaLink frame is passed to the transport.
7. `frame_rx` — the peer validates the frame CRC.
8. `tts_enqueue` — a speakable clause is queued.
9. `tts_inference_start` and `tts_inference_end`.
10. `first_audio_sample` — the player submits the first non-silent PCM sample.
11. `playback_complete`.

The app stores telemetry in a bounded in-memory ring and exports JSONL only after an explicit evaluator action. Raw microphone audio and transcript text are excluded by default; a salted utterance hash and token count are sufficient for latency analysis.

## Cross-device offset estimation

A sender clock and receiver clock cannot be compared directly. Before an evaluation session, peers exchange a four-timestamp probe:

- sender sends at `t0`;
- receiver receives at `t1` and replies at `t2`;
- sender receives at `t3`.

For low-symmetry error, estimate:

```text
round_trip = (t3 - t0) - (t2 - t1)
offset = ((t1 - t0) + (t2 - t3)) / 2
```

Run at least 20 probes and retain the offset from the lowest-round-trip quartile. Repeat every 60 seconds during long tests. Report both corrected one-way latency and uncorrected sender-end-to-receiver-audio wall duration. Never silently mix samples from different offset epochs.

## Required derived metrics

- ASR first-partial latency: `asr_partial - speech_start`.
- ASR endpoint latency: final `stable_prefix_commit - speech_end`.
- Network delivery latency: corrected `frame_rx - frame_tx`.
- TTS compute RTF: `(tts_inference_end - tts_inference_start) / generated_audio_duration`.
- Receiver start latency: `first_audio_sample - frame_rx`.
- Full loop latency: corrected `first_audio_sample(receiver) - speech_end(sender)`.
- Streaming loop latency: corrected first remote audio associated with a stable clause minus the sender's clause commit.

Report median, p90, p95, and p99; a mean alone hides endpoint and retransmission failures.
