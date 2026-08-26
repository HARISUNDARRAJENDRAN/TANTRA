# AksharaLink v1 wire protocol

All multi-byte integers are big-endian.

## Frame

| Field | Bytes | Meaning |
|---|---:|---|
| Magic | 2 | ASCII `TN` |
| Version/flags | 1 | high 3 bits version, low 5 bits flags |
| Kind/language | 1 | high 4 bits kind, low 4 bits language |
| Priority | 1 | 0 normal, 1 urgent, 2 alert |
| Session ID | 4 | random unsigned ID |
| Sequence | 4 | monotonically increasing per session |
| Sender timestamp | 8 | monotonic milliseconds |
| Payload length | 2 | 0–65535 |
| Payload | variable | kind-specific bytes |
| CRC-32 | 4 | header and payload |

Version 1 frame overhead is 27 bytes including CRC. A length prefix is added by stream transports and is not part of the frame.

## Flags

- bit 0 `FINAL`
- bit 1 `TOKEN_PAYLOAD`
- bit 2 `ACK_REQUIRED`
- bit 3 `ENCRYPTED`
- bit 4 `FULL_SNAPSHOT`

## Kinds

1. `HELLO`
2. `PARTIAL`
3. `CLAUSE`
4. `FINAL`
5. `ALERT`
6. `ACK`
7. `HEARTBEAT`
8. `CONTROL`

## Language IDs

0 unknown, 1 Hindi, 2 Gujarati, 3 Marathi, 4 Kannada, 5 Malayalam, 6 Tamil, 7 Telugu, 8 Odia, 9 Bengali, 10 English.

## Token-delta payload

```text
base_sequence: u32
replace_from_token_index: varuint
suffix_token_count: varuint
suffix_token_ids: repeated varuint
```

A full snapshot sets `FULL_SNAPSHOT` and uses `replace_from=0`. A receiver missing `base_sequence` requests a snapshot rather than applying a potentially incorrect delta.

## ACK payload

`acked_sequence` as `u32` followed by one status byte (`0=accepted`, `1=duplicate`, `2=bad-base`, `3=unsupported`).
