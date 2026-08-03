import pytest

import rns_engine as rns


def test_single_int8_plane_has_expected_int32_bound():
    plan = rns.plan_grouped_coefficient_capacity(
        inner_dimension=5120,
        left_digit_abs_bounds=[127],
        right_digit_abs_bounds=[127],
    )

    assert plan.coefficient_count == 1
    assert plan.plane_pair_count == 1
    assert plan.coefficient_pair_counts == (1,)
    assert plan.coefficient_abs_bounds == (82_580_480,)
    assert plan.max_abs_bound == 82_580_480
    assert plan.safe is True
    assert plan.minimum_signed_accumulator_bits == 28
    assert plan.require_safe() is plan


def test_eight_plane_grouped_coefficients_fit_int32_while_global_result_does_not():
    local = rns.plan_grouped_coefficient_capacity(
        inner_dimension=5120,
        left_digit_abs_bounds=[127] * 8,
        right_digit_abs_bounds=[127] * 8,
        accumulator_bits=32,
    )
    global_plan = rns.plan_digit_plane_gemm_capacity(
        inner_dimension=5120,
        radix=128,
        left_digit_abs_bounds=[127] * 8,
        right_digit_abs_bounds=[127] * 8,
    )

    expected_pair_counts = (1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1)
    one_pair_bound = 82_580_480

    assert local.coefficient_pair_counts == expected_pair_counts
    assert local.coefficient_abs_bounds == tuple(
        count * one_pair_bound for count in expected_pair_counts
    )
    assert local.max_abs_bound == 660_643_840
    assert local.minimum_signed_accumulator_bits == 31
    assert local.headroom == 1_486_839_807
    assert local.safe is True
    assert global_plan.current_unique is False
    assert global_plan.capacity.required_modulus_bits == 126


def test_grouped_coefficient_plan_detects_int32_overflow():
    plan = rns.plan_grouped_coefficient_capacity(
        inner_dimension=20_000,
        left_digit_abs_bounds=[127] * 8,
        right_digit_abs_bounds=[127] * 8,
        accumulator_bits=32,
    )

    assert plan.max_abs_bound == 20_000 * 8 * 127 * 127
    assert plan.safe is False
    assert plan.headroom < 0
    assert plan.minimum_signed_accumulator_bits == 33
    with pytest.raises(OverflowError, match="exceeds"):
        plan.require_safe()


def test_grouped_bounds_match_manual_nonuniform_convolution():
    plan = rns.plan_grouped_coefficient_capacity(
        inner_dimension=11,
        left_digit_abs_bounds=[7, 3, 1],
        right_digit_abs_bounds=[5, 2],
    )

    assert plan.coefficient_pair_counts == (1, 2, 2, 1)
    assert plan.coefficient_abs_bounds == (
        11 * (7 * 5),
        11 * (7 * 2 + 3 * 5),
        11 * (3 * 2 + 1 * 5),
        11 * (1 * 2),
    )


def test_empty_plane_body_has_zero_local_bound():
    plan = rns.plan_grouped_coefficient_capacity(
        inner_dimension=5120,
        left_digit_abs_bounds=[],
        right_digit_abs_bounds=[127],
    )

    assert plan.coefficient_count == 0
    assert plan.coefficient_pair_counts == ()
    assert plan.coefficient_abs_bounds == ()
    assert plan.max_abs_bound == 0
    assert plan.minimum_signed_accumulator_bits == 1
    assert plan.safe is True


def test_grouped_coefficient_inputs_are_strict_integers():
    with pytest.raises(TypeError, match="inner_dimension"):
        rns.plan_grouped_coefficient_capacity(1.5, [1], [1])

    with pytest.raises(ValueError, match="inner_dimension"):
        rns.plan_grouped_coefficient_capacity(-1, [1], [1])

    with pytest.raises(ValueError, match=r"left_digit_abs_bounds\[0\]"):
        rns.plan_grouped_coefficient_capacity(1, [-1], [1])

    with pytest.raises(ValueError, match="accumulator_bits"):
        rns.plan_grouped_coefficient_capacity(1, [1], [1], accumulator_bits=0)
