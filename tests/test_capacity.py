import math

import pytest

import rns_engine as rns


def test_single_plane_int8_gemm_fits_current_four_rail_signed_range():
    plan = rns.plan_digit_plane_gemm_capacity(
        inner_dimension=5120,
        radix=128,
        left_digit_abs_bounds=[127],
        right_digit_abs_bounds=[127],
    )

    assert plan.plane_pair_count == 1
    assert plan.grouped_coefficient_count == 1
    assert plan.left_value_abs_bound == 127
    assert plan.right_value_abs_bound == 127
    assert plan.max_abs_bound == 82_580_480
    assert plan.current_unique is True
    assert plan.unique is True
    assert plan.capacity.additional_bits_required == 0
    assert plan.require_unique() is plan


def test_radix128_plane_count_crosses_current_capacity_at_three_planes():
    two_planes = rns.plan_digit_plane_gemm_capacity(
        5120,
        128,
        [127, 127],
        [127, 127],
    )
    three_planes = rns.plan_digit_plane_gemm_capacity(
        5120,
        128,
        [127, 127, 127],
        [127, 127, 127],
    )

    assert two_planes.left_value_abs_bound == 128**2 - 1
    assert two_planes.current_unique is True
    assert three_planes.left_value_abs_bound == 128**3 - 1
    assert three_planes.current_unique is False
    with pytest.raises(OverflowError, match="modular-only"):
        three_planes.require_unique()


def test_eight_full_radix128_planes_need_126_bit_modulus():
    plan = rns.plan_digit_plane_gemm_capacity(
        inner_dimension=5120,
        radix=128,
        left_digit_abs_bounds=[127] * 8,
        right_digit_abs_bounds=[127] * 8,
    )

    full_value_bound = 128**8 - 1
    expected_output_bound = 5120 * full_value_bound**2

    assert plan.left_value_abs_bound == full_value_bound
    assert plan.right_value_abs_bound == full_value_bound
    assert plan.plane_pair_count == 64
    assert plan.grouped_coefficient_count == 15
    assert plan.max_abs_bound == expected_output_bound
    assert plan.capacity.minimum_required_modulus == 2 * expected_output_bound + 1
    assert plan.capacity.required_modulus_bits == 126
    assert plan.capacity.additional_bits_required == 71
    assert plan.current_unique is False


def test_plane_pair_ledger_matches_factored_value_bound_exactly():
    plan = rns.plan_digit_plane_gemm_capacity(
        inner_dimension=37,
        radix=16,
        left_digit_abs_bounds=[7, 3, 1],
        right_digit_abs_bounds=[5, 2],
        addend_abs_bound=11,
    )

    expanded = rns.plan_weighted_sum_capacity(
        plan.plane_pair_weights,
        plan.plane_pair_abs_bounds,
        addend_abs_bound=11,
    )

    assert expanded.max_abs_bound == plan.max_abs_bound
    assert plan.plane_pair_weights == (1, 16, 16, 256, 256, 4096)
    assert plan.plane_pair_abs_bounds == (
        37 * 7 * 5,
        37 * 7 * 2,
        37 * 3 * 5,
        37 * 3 * 2,
        37 * 1 * 5,
        37 * 1 * 2,
    )


def test_candidate_extra_rail_is_checked_by_exact_coprime_product():
    baseline = rns.plan_digit_plane_gemm_capacity(
        5120,
        128,
        [127] * 8,
        [127] * 8,
    )
    extra_modulus = baseline.capacity.minimum_single_coprime_modulus

    expanded = rns.plan_digit_plane_gemm_capacity(
        5120,
        128,
        [127] * 8,
        [127] * 8,
        additional_moduli=[extra_modulus],
    )

    assert math.gcd(extra_modulus, rns.M) == 1
    assert expanded.capacity.available_modulus >= expanded.capacity.minimum_required_modulus
    assert expanded.unique is True
    assert expanded.require_unique() is expanded


def test_capacity_planner_rejects_non_coprime_or_invalid_candidate_rails():
    with pytest.raises(ValueError, match="coprime"):
        rns.plan_signed_capacity(1, additional_moduli=[127])

    with pytest.raises(ValueError, match="coprime"):
        rns.plan_signed_capacity(1, additional_moduli=[3, 3])

    with pytest.raises(ValueError, match="> 1"):
        rns.plan_signed_capacity(1, additional_moduli=[1])


def test_capacity_planner_uses_exact_integer_inputs_only():
    with pytest.raises(TypeError, match="max_abs_bound"):
        rns.plan_signed_capacity(1.5)

    with pytest.raises(ValueError, match="max_abs_bound"):
        rns.plan_signed_capacity(-1)

    with pytest.raises(ValueError, match="radix"):
        rns.plan_digit_plane_gemm_capacity(1, 1, [1], [1])

    with pytest.raises(ValueError, match=r"left_digit_abs_bounds\[0\]"):
        rns.plan_digit_plane_gemm_capacity(1, 2, [-1], [1])

    with pytest.raises(ValueError, match="same length"):
        rns.plan_weighted_sum_capacity([1, 2], [3])


def test_empty_digit_plane_body_reduces_to_addend_capacity():
    plan = rns.plan_digit_plane_gemm_capacity(
        inner_dimension=5120,
        radix=128,
        left_digit_abs_bounds=[],
        right_digit_abs_bounds=[127],
        addend_abs_bound=9,
    )

    assert plan.plane_pair_count == 0
    assert plan.grouped_coefficient_count == 0
    assert plan.max_abs_bound == 9
    assert plan.current_unique is True
