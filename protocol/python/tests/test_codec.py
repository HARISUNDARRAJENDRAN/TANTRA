import pytest

from tantra_protocol import (
    Flags,
    Frame,
    FrameKind,
    Language,
    Priority,
    ProtocolError,
    StablePrefixCommitter,
    apply_delta,
    decode_frame,
    decode_text_delta,
    encode_frame,
    encode_text_delta,
    make_delta,
)


def test_frame_round_trip() -> None:
    frame = Frame(
        kind=FrameKind.ALERT,
        language=Language.HINDI,
        priority=Priority.ALERT,
        session_id=123,
        sequence=7,
        sender_timestamp_ms=456789,
        payload="सहायता चाहिए".encode(),
        flags=Flags.FINAL | Flags.ACK_REQUIRED,
    )
    assert decode_frame(encode_frame(frame)) == frame


def test_corruption_is_detected() -> None:
    encoded = bytearray(
        encode_frame(Frame(FrameKind.FINAL, Language.ENGLISH, Priority.NORMAL, 1, 1, b"help", sender_timestamp_ms=1))
    )
    encoded[-5] ^= 0xAA
    with pytest.raises(ProtocolError, match="CRC"):
        decode_frame(bytes(encoded))


def test_token_delta_round_trip() -> None:
    previous = (11, 12, 13, 14)
    current = (11, 12, 99, 100, 101)
    delta = make_delta(previous, current, base_sequence=8)
    assert apply_delta(previous, decode_text_delta(encode_text_delta(delta))) == current


def test_invalid_delta_rejected() -> None:
    from tantra_protocol.codec import TextDelta

    with pytest.raises(ProtocolError):
        apply_delta((1, 2), TextDelta(1, 3, (4,)))


def test_stable_prefix_commits_only_boundary() -> None:
    committer = StablePrefixCommitter(required_observations=2, minimum_stability_ms=10_000)
    assert committer.update("we need med", now_ms=0) == ""
    assert committer.update("we need medical", now_ms=10) == ""
    assert committer.update("we need medical", now_ms=20) == "we need"
    assert committer.update("we need medical help", now_ms=30) == ""
    assert committer.update("we need medical help.", now_ms=40, final=True) == "medical help."
