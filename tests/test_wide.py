from math import gcd

import numpy as np
import pytest

import rns_engine as rns


def test_seven_rail_profiles_are_pairwise_coprime_and_close_target():
    target = rns.plan_digit_plane_gemm_capacity(
        5120,
        128,
        [127] * 8,
        [127] * 8,
    ).capacity

    for config in (
        rns.WideRNSConfig.smallest_product_seven_rail(),
        rns.WideRNSConfig.balanced_seven_rail(),
    ):
        running = 1
        for modulus in config.moduli:
            assert gcd(running, modulus) == 1
            running *= modulus
        assert config.rail_count == 7
        assert config.product >= target.minimum_required_modulus
        assert config.product_bits >= 126
        config.require_unique_bound(target.max_abs_bound)


def test_wide_signed_round_trip_covers_boundaries_and_big_values():
    config = rns.WideRNSConfig.balanced_seven_rail()
    values = np.asarray(
        [
            config.signed_min,
            -2**80 + 7,
            -1,
            0,
            1,
            2**80 - 9,
            config.signed_max,
        ],
        dtype=object,
    )

    rails = config.encode(values)
    decoded = config.decode(rails, signed=True)

    assert np.array_equal(decoded, values)
    assert [rail.dtype for rail in rails[:4]] == [
        np.dtype(np.uint16),
        np.dtype(np.uint16),
        np.dtype(np.uint32),
        np.dtype(np.uint32),
    ]


def test_wide_reference_add_and_mul_are_exact_when_bound_is_unique():
    config = rns.WideRNSConfig.smallest_product_seven_rail()
    left = np.asarray([2**60 + 3, -(2**55) + 7], dtype=object)
    right = np.asarray([2**58 - 5, 2**54 + 11], dtype=object)

    added = config.decode(config.add(config.encode(left), config.encode(right)), signed=True)
    multiplied = config.decode(config.mul(config.encode(left), config.encode(right)), signed=True)

    assert np.array_equal(added, left + right)
    assert np.array_equal(multiplied, left * right)


def test_wide_config_rejects_non_coprime_moduli_and_modular_only_claims():
    with pytest.raises(ValueError, match="not coprime"):
        rns.WideRNSConfig((7, 21))

    config = rns.WideRNSConfig.current()
    assert config.uniquely_represents_bound(config.product // 2 - 1)
    assert not config.uniquely_represents_bound(config.product // 2)
    with pytest.raises(OverflowError, match="modular-only"):
        config.require_unique_bound(config.product // 2)


def test_wide_weighted_int32_matches_python_big_integer_sum():
    config = rns.WideRNSConfig.balanced_seven_rail()
    partials = np.asarray(
        [
            [[2**30 - 1, -(2**30)], [17, -19]],
            [[-777, 333], [2**29, -(2**28)]],
            [[9, -11], [13, -15]],
        ],
        dtype=np.int32,
    )
    weights = (1, 128**4, -(128**9))

    result = rns.accumulate_weighted_int32_wide(
        partials,
        weights,
        config,
        require_unique=True,
    )
    expected = sum(
        partial.astype(object) * weight
        for partial, weight in zip(partials, weights)
    )

    assert result.unique
    assert np.array_equal(result.decode_signed(), expected)
    assert result.max_abs_bound == sum(
        abs(weight) * max(abs(int(term.min())), abs(int(term.max())))
        for term, weight in zip(partials, weights)
    )
