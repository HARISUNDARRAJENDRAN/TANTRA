# Security and privacy

## Threats

- nearby peer impersonation;
- replayed alert frames;
- malicious/corrupt model packs;
- packet corruption on embedded links;
- unintended audio retention;
- denial of service through repeated high-priority messages.

## Controls

- Pair peers with a user-confirmed out-of-band key; derive session keys with HKDF-SHA256.
- Protect payloads with AES-256-GCM and bind header fields as associated data.
- Include monotonically increasing sequence numbers and reject old/replayed frames.
- Verify model-pack hashes and license manifest before activation.
- Keep audio buffers in memory and zero/drop them after finalization.
- Rate-limit alert frames and require a locally visible paired identity.
- Do not log transcripts by default.

The reference protocol implements CRC and sequence handling. The Android crypto envelope is modular so a security review can be completed independently before field deployment.
