# TANTRA architecture

## 1. Design objective

The product must preserve spoken interaction over links where sending conventional audio is expensive or unreliable. TANTRA therefore separates the *human interface* (speech at each endpoint) from the *radio payload* (compact linguistic tokens).

## 2. Runtime components

### SvaraGate — capture and endpointing

Android `AudioRecord` captures 16 kHz mono PCM. Platform acoustic echo cancellation, noise suppression, and automatic gain control are enabled when available. A causal adaptive detector estimates the local noise floor, applies hysteresis, and combines minimum-speech, hangover, endpoint-silence, and maximum-utterance rules. PTT release is an explicit final boundary; continuous mode uses pause detection.

The detector is intentionally independent of a proprietary wake-word SDK. A small neural VAD can later replace the energy front end without changing the state machine.

### Shruti-ASR — local multilingual recognition

The deployment contract is a quantized raw-waveform CTC ONNX model:

- input `samples`: float32 `[1, N]`, 16 kHz normalized waveform;
- input `sample_lengths`: int64 `[1]`;
- optional input `language_id`: int64 `[1]`;
- output `logits`: float32 `[1, T, V]`.

The reference student uses a strided causal convolutional front end, compact Conformer blocks, a shared grapheme vocabulary, and a learned language embedding. Training supports hard-label supervision and teacher pseudo-labels. The Android engine uses overlapping windows and longest-common-prefix stabilization; a stateful streaming export can be substituted behind the same interface.

### StablePrefix — safe early commitment

Raw partial transcripts fluctuate. TANTRA tracks consecutive hypotheses and commits only a prefix that:

1. appears unchanged in at least `k` updates or for a stability duration;
2. ends at a safe word/clause boundary; and
3. has not already been transmitted.

The final endpoint flushes the remaining suffix. The receiver can therefore synthesize committed clauses before the speaker finishes, while never speaking text that a later ASR update retracts.

### AksharaLink — low-bitrate linguistic framing

The wire protocol carries:

- session and sequence IDs;
- language and priority;
- partial/final/alert/control kind;
- stable-prefix replacement index;
- compact token IDs or UTF-8 fallback;
- timestamp for end-to-end latency;
- CRC-32 for corruption detection;
- optional acknowledgement and AES-GCM envelope.

Shared Indic grapheme tokens are encoded as unsigned varints. Delta packets replace only the suffix after the longest common prefix. Partial updates may be dropped; final and alert packets are retried until acknowledged.

### Nadi — transport abstraction

`DuplexTransport` exposes the same byte stream over:

- Wi-Fi LAN TCP, with a small UDP/NSD discovery layer;
- Bluetooth Classic RFCOMM for broad Android compatibility;
- loopback for deterministic tests.

The speech pipeline never depends on an Internet route. TCP/RFCOMM are length-prefixed. Transport reconnects do not reset the ASR session; sequence IDs suppress duplicates.

### Vaani-TTS — receiver-side synthesis

The deployment contract is a single ONNX graph:

- `tokens`: int64 `[1, L]`;
- `token_lengths`: int64 `[1]`;
- `language_id`: int64 `[1]`;
- `speaker_id`: int64 `[1]`;
- `speed`: float32 `[1]`;
- output `audio`: float32 `[1, S]`.

This standard signature allows a distilled VITS, FastSpeech-style model plus vocoder, or another open architecture to be swapped without Android changes. A receiver queue starts synthesis at committed clause boundaries and cross-fades adjacent PCM segments.

### Dhwani — priority and duplex policy

Normal messages use voice-communication audio attributes. Alert messages request exclusive transient focus, use alarm attributes, move to the front of the queue, and optionally retry playback until acknowledged by the user. Android OEM/user policy remains authoritative; the app surfaces missing notification/DND permissions rather than claiming an impossible absolute guarantee.

In PTT mode capture and playback are half-duplex. Continuous mode enables AEC where available and temporarily gates capture during local TTS on devices whose echo canceller is inadequate.

## 3. Novel aspects

### Token-native radio access

Most voice systems optimize an audio codec. TANTRA instead makes the ASR vocabulary the link alphabet. The same token IDs are consumed by the receiver TTS front end, avoiding repeated Unicode normalization and enabling incremental suffix updates.

### Confidence-aware link budget

Final/alert information receives acknowledgements and retries. Low-confidence partials are not retransmitted. When the link degrades, the sender increases the prefix stability threshold and packet coalescing window rather than increasing audio distortion.

### Language adapters under a shared memory budget

The target model uses a shared encoder and low-rank language adapters. Only the active adapter and TTS voice are resident. This preserves multilingual coverage without loading ten full networks at once.

### Measurable glass-to-glass latency

Sender endpoint timestamps are carried on the wire. Receiver audio-start callbacks close the measurement loop, producing p50/p95/p99 values for microphone-to-speaker latency without manual stopwatch measurements.

## 4. Failure behavior

- **No model pack:** UI remains usable for pairing/settings but PTT is disabled with an explicit diagnostic.
- **Corrupt packet:** CRC failure drops the frame; reliable kinds are recovered by timeout/retry.
- **Duplicate final:** sequence cache suppresses repeated speech.
- **Lost partial:** next delta is based on the last acknowledged committed prefix or falls back to a full final payload.
- **TTS failure:** text remains visible, the failure is logged locally, and an optional open eSpeak-NG emergency fallback can be integrated as a separately licensed module.
- **Echo loop:** capture gates while local TTS plays when platform AEC is unavailable or fails a calibration check.
