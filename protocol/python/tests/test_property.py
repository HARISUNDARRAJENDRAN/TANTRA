from hypothesis import given, strategies as st

from tantra_protocol.codec import decode_varuint, encode_varuint


@given(st.integers(min_value=0, max_value=(1 << 63) - 1))
def test_varuint_round_trip(value: int) -> None:
    encoded = encode_varuint(value)
    decoded, offset = decode_varuint(encoded)
    assert decoded == value
    assert offset == len(encoded)
