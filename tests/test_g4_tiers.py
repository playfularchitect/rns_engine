from fractions import Fraction

import numpy as np
import pytest

from rns_engine.tiers import (
    ExactRangeError,
    G416,
    G432,
    G464,
    G4X,
    PromotionPolicy,
    current_promotion_policy,
    promote_exact,
    scalar,
    tensor,
    tier_table,
)


def test_public_range_envelopes_match_ieee_binary_tiers():
    assert G416.max_finite == 65504
    assert G416.min_positive == Fraction(1, 2**24)
    assert G432.max_finite == (2**24 - 1) * 2**104
    assert G432.min_positive == Fraction(1, 2**149)
    assert G464.max_finite == (2**53 - 1) * 2**971
    assert G464.min_positive == Fraction(1, 2**1074)


def test_strict_is_the_default_and_never_silently_promotes():
    assert current_promotion_policy() is PromotionPolicy.STRICT
    with pytest.raises(ExactRangeError):
        _ = G416(G416.max_finite) + G416(1)
    assert current_promotion_policy() is PromotionPolicy.STRICT


def test_explicit_expert_promotion_selects_the_smallest_sufficient_tier():
    with promote_exact():
        result = G416(G416.max_finite) + G416(1)
    assert result.tier is G432
    assert result.value == 65505
    assert current_promotion_policy() is PromotionPolicy.STRICT


def test_values_beyond_g464_promote_to_unbounded_g4x_only_when_requested():
    with pytest.raises(ExactRangeError):
        _ = G464(G464.max_finite) * G464(2)
    with promote_exact():
        result = G464(G464.max_finite) * G464(2)
    assert result.tier is G4X


def test_subnormal_floor_is_part_of_each_public_range_contract():
    G416(G416.min_positive)
    with pytest.raises(ExactRangeError):
        G416(G416.min_positive / 2)
    G416(0)


def test_float_inputs_are_decoded_exactly_not_decimalized():
    value = scalar(np.float32(0.1), G432)
    numerator, denominator = float(np.float32(0.1)).as_integer_ratio()
    assert value.value == Fraction(numerator, denominator)


def test_exact_fraction_arithmetic_is_preserved():
    left = scalar("1/3", G416)
    right = scalar("1/6", G416)
    assert (left + right).value == Fraction(1, 2)


def test_arrays_are_strict_and_exact():
    left = tensor([[1, 2], [3, 4]], G416)
    right = tensor([[5, 6], [7, 8]], G416)
    result = left @ right
    assert result.tier is G416
    assert result.fractions().tolist() == [[Fraction(19), Fraction(22)], [Fraction(43), Fraction(50)]]


def test_tier_table_names_all_public_contracts():
    rows = tier_table()
    assert [row["name"] for row in rows] == ["G416", "G432", "G464", "G4X"]
    assert all(row["strict_default"] for row in rows)
