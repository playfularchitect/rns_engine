import numpy as np
import pytest

import rns_engine as rns


def test_signed_constants_match_four_rail_modulus():
    assert rns.M == 35_742_890_181_197_824
    assert rns.HALF_M == rns.M // 2
    assert rns.SIGNED_MIN == -rns.HALF_M
    assert rns.SIGNED_MAX == rns.HALF_M - 1
    assert rns.UNIQUE_SIGNED_MIN == -rns.HALF_M + 1
    assert rns.UNIQUE_SIGNED_MAX == rns.HALF_M - 1


def test_signed_round_trip_including_canonical_boundaries():
    values = np.array(
        [rns.SIGNED_MIN, -1_000_003, -1, 0, 1, 1_000_003, rns.SIGNED_MAX],
        dtype=np.int64,
    )

    decoded = rns.decode_signed(*rns.encode_signed(values))

    np.testing.assert_array_equal(decoded, values)
    assert decoded.dtype == np.int64


def test_negative_one_uses_the_same_residue_as_m_minus_one():
    signed_rails = rns.encode_signed(np.array([-1], dtype=np.int64))
    unsigned_rails = rns.encode(np.array([rns.M - 1], dtype=np.uint64))

    for signed_rail, unsigned_rail in zip(signed_rails, unsigned_rails):
        np.testing.assert_array_equal(signed_rail, unsigned_rail)


def test_half_modulus_residue_has_canonical_negative_interpretation():
    rails = rns.encode(np.array([rns.HALF_M], dtype=np.uint64))

    decoded = rns.decode_signed(*rails)

    np.testing.assert_array_equal(decoded, np.array([rns.SIGNED_MIN], dtype=np.int64))


def test_signed_encoder_refuses_silent_wraparound():
    with pytest.raises(OverflowError, match="signed RNS input"):
        rns.encode_signed([rns.SIGNED_MIN - 1])

    with pytest.raises(OverflowError, match="signed RNS input"):
        rns.encode_signed([rns.SIGNED_MAX + 1])


def test_signed_encoder_rejects_non_integer_and_non_vector_input():
    with pytest.raises(TypeError, match="must contain integers"):
        rns.encode_signed([1.5, 2.5])

    with pytest.raises(ValueError, match="1D"):
        rns.encode_signed([[1, 2], [3, 4]])


def test_signed_session_uses_existing_residue_arithmetic():
    session = rns.SignedSession()
    a = session.encode_signed([-5, 7])
    b = session.encode_signed([3, -10])

    np.testing.assert_array_equal(
        session.decode_signed(session.add(a, b)),
        np.array([-2, -3], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        session.decode_signed(session.mul(a, b)),
        np.array([-15, -70], dtype=np.int64),
    )


def test_strict_signed_bound_certificate():
    safe = rns.certify_signed_bound(rns.HALF_M - 1)
    assert safe.unique is True
    assert safe.headroom == 0
    assert safe.minimum_required_modulus == rns.M - 1
    assert safe.require_unique() is safe

    ambiguous = rns.certify_signed_bound(rns.HALF_M)
    assert ambiguous.unique is False
    assert ambiguous.headroom == -1
    assert ambiguous.minimum_required_modulus == rns.M + 1
    with pytest.raises(OverflowError, match="modular-only"):
        ambiguous.require_unique()


def test_signed_bound_certificate_validates_the_bound():
    with pytest.raises(ValueError, match=">= 0"):
        rns.certify_signed_bound(-1)

    with pytest.raises(TypeError, match="must be an integer"):
        rns.certify_signed_bound(1.5)
