from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
import struct
import time
import zlib
from typing import Iterable, Sequence

MAGIC = b"TN"
VERSION = 1
_HEADER = struct.Struct(">2sBBBIIQH")
_CRC = struct.Struct(">I")
_MAX_PAYLOAD = 65535


class ProtocolError(ValueError):
    pass


class Flags(IntFlag):
    NONE = 0
    FINAL = 1 << 0
    TOKEN_PAYLOAD = 1 << 1
    ACK_REQUIRED = 1 << 2
    ENCRYPTED = 1 << 3
    FULL_SNAPSHOT = 1 << 4


class FrameKind(IntEnum):
    HELLO = 1
    PARTIAL = 2
    CLAUSE = 3
    FINAL = 4
    ALERT = 5
    ACK = 6
    HEARTBEAT = 7
    CONTROL = 8


class Language(IntEnum):
    UNKNOWN = 0
    HINDI = 1
    GUJARATI = 2
    MARATHI = 3
    KANNADA = 4
    MALAYALAM = 5
    TAMIL = 6
    TELUGU = 7
    ODIA = 8
    BENGALI = 9
    ENGLISH = 10


class Priority(IntEnum):
    NORMAL = 0
    URGENT = 1
    ALERT = 2


class AckStatus(IntEnum):
    ACCEPTED = 0
    DUPLICATE = 1
    BAD_BASE = 2
    UNSUPPORTED = 3


@dataclass(frozen=True, slots=True)
class Frame:
    kind: FrameKind
    language: Language
    priority: Priority
    session_id: int
    sequence: int
    payload: bytes = b""
    flags: Flags = Flags.NONE
    sender_timestamp_ms: int = 0

    def with_clock(self) -> "Frame":
        if self.sender_timestamp_ms:
            return self
        return Frame(
            kind=self.kind,
            language=self.language,
            priority=self.priority,
            session_id=self.session_id,
            sequence=self.sequence,
            payload=self.payload,
            flags=self.flags,
            sender_timestamp_ms=time.monotonic_ns() // 1_000_000,
        )


@dataclass(frozen=True, slots=True)
class TextDelta:
    base_sequence: int
    replace_from: int
    suffix_tokens: tuple[int, ...]


def _check_uint(name: str, value: int, bits: int) -> None:
    if not 0 <= value < (1 << bits):
        raise ProtocolError(f"{name} is outside uint{bits}: {value}")


def encode_varuint(value: int) -> bytes:
    if value < 0:
        raise ProtocolError("varuint cannot encode a negative value")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def decode_varuint(data: bytes, offset: int = 0) -> tuple[int, int]:
    value = 0
    shift = 0
    for index in range(offset, min(len(data), offset + 10)):
        byte = data[index]
        value |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            return value, index + 1
        shift += 7
    raise ProtocolError("truncated or oversized varuint")


def encode_frame(frame: Frame) -> bytes:
    frame = frame.with_clock()
    payload = bytes(frame.payload)
    if len(payload) > _MAX_PAYLOAD:
        raise ProtocolError(f"payload exceeds {_MAX_PAYLOAD} bytes")
    _check_uint("session_id", frame.session_id, 32)
    _check_uint("sequence", frame.sequence, 32)
    _check_uint("sender_timestamp_ms", frame.sender_timestamp_ms, 64)
    if int(frame.flags) & ~0x1F:
        raise ProtocolError("flags exceed five-bit field")
    version_flags = (VERSION << 5) | int(frame.flags)
    kind_language = (int(frame.kind) << 4) | int(frame.language)
    header = _HEADER.pack(
        MAGIC,
        version_flags,
        kind_language,
        int(frame.priority),
        frame.session_id,
        frame.sequence,
        frame.sender_timestamp_ms,
        len(payload),
    )
    body = header + payload
    return body + _CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def decode_frame(data: bytes) -> Frame:
    minimum = _HEADER.size + _CRC.size
    if len(data) < minimum:
        raise ProtocolError("frame is shorter than minimum header")
    magic, version_flags, kind_language, priority, session_id, sequence, timestamp, size = _HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ProtocolError("bad frame magic")
    version = version_flags >> 5
    if version != VERSION:
        raise ProtocolError(f"unsupported version {version}")
    expected = _HEADER.size + size + _CRC.size
    if len(data) != expected:
        raise ProtocolError(f"frame length mismatch: expected {expected}, received {len(data)}")
    expected_crc = _CRC.unpack_from(data, expected - _CRC.size)[0]
    actual_crc = zlib.crc32(data[:-_CRC.size]) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ProtocolError("CRC mismatch")
    try:
        kind = FrameKind(kind_language >> 4)
        language = Language(kind_language & 0x0F)
        parsed_priority = Priority(priority)
    except ValueError as exc:
        raise ProtocolError(f"unknown enum value: {exc}") from exc
    return Frame(
        kind=kind,
        language=language,
        priority=parsed_priority,
        session_id=session_id,
        sequence=sequence,
        sender_timestamp_ms=timestamp,
        payload=data[_HEADER.size : _HEADER.size + size],
        flags=Flags(version_flags & 0x1F),
    )


def encode_text_delta(delta: TextDelta) -> bytes:
    _check_uint("base_sequence", delta.base_sequence, 32)
    out = bytearray(struct.pack(">I", delta.base_sequence))
    out += encode_varuint(delta.replace_from)
    out += encode_varuint(len(delta.suffix_tokens))
    for token in delta.suffix_tokens:
        out += encode_varuint(token)
    if len(out) > _MAX_PAYLOAD:
        raise ProtocolError("encoded token delta exceeds frame payload limit")
    return bytes(out)


def decode_text_delta(data: bytes) -> TextDelta:
    if len(data) < 4:
        raise ProtocolError("token delta is missing base sequence")
    base_sequence = struct.unpack_from(">I", data)[0]
    replace_from, offset = decode_varuint(data, 4)
    count, offset = decode_varuint(data, offset)
    tokens: list[int] = []
    for _ in range(count):
        token, offset = decode_varuint(data, offset)
        tokens.append(token)
    if offset != len(data):
        raise ProtocolError("token delta has trailing bytes")
    return TextDelta(base_sequence, replace_from, tuple(tokens))


def make_delta(previous: Sequence[int], current: Sequence[int], base_sequence: int) -> TextDelta:
    common = 0
    for left, right in zip(previous, current):
        if left != right:
            break
        common += 1
    return TextDelta(base_sequence, common, tuple(current[common:]))


def apply_delta(previous: Sequence[int], delta: TextDelta) -> tuple[int, ...]:
    if delta.replace_from > len(previous):
        raise ProtocolError("delta replace index exceeds current token count")
    return tuple(previous[: delta.replace_from]) + delta.suffix_tokens


def encode_ack(sequence: int, status: AckStatus = AckStatus.ACCEPTED) -> bytes:
    _check_uint("sequence", sequence, 32)
    return struct.pack(">IB", sequence, int(status))


def decode_ack(payload: bytes) -> tuple[int, AckStatus]:
    if len(payload) != 5:
        raise ProtocolError("ACK payload must be five bytes")
    sequence, status = struct.unpack(">IB", payload)
    try:
        return sequence, AckStatus(status)
    except ValueError as exc:
        raise ProtocolError(f"unknown ACK status {status}") from exc
