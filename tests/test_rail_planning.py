from math import gcd

import pytest

import rns_engine as rns


def _eight_plane_capacity():
    return rns.plan_digit_plane_gemm_capacity(
        inner_dimension=5120,
        radix=128,
        left_digit_abs_bounds=[127] * 8,
        right_digit_abs_bounds=[127] * 8,
    ).capacity


def test_31_bit_ceiling_requires_three_additional_mersenne_rails():
    capacity = _eight_plane_capacity()
    search = rns.search_mersenne_rails_for_capacity(
        capacity,
        max_exponent=31,
        max_rails=4,
        minimal_rail_count_only=True,
    )

    assert capacity.additional_bits_required == 71
    assert search.minimum_rail_count == 3
    assert search.solutions
    assert all(plan.rail_count == 3 for plan in search.solutions)
    assert all(plan.sufficient for plan in search.solutions)

    maximum_two_rail_product = (2**31 - 1) * (2**30 - 1)
    assert maximum_two_rail_product < capacity.minimum_additional_product_factor


def test_smallest_product_solution_is_exact_and_pairwise_coprime():
    capacity = _eight_plane_capacity()
    search = rns.search_mersenne_rails_for_capacity(
        capacity,
        max_exponent=31,
        max_rails=3,
    )
    plan = search.smallest_product_plan

    assert plan is not None
    assert plan.exponents == (11, 29, 31)
    assert plan.moduli == (2**11 - 1, 2**29 - 1, 2**31 - 1)
    assert plan.additional_product_bits == 71
    assert plan.additional_product >= capacity.minimum_additional_product_factor
    assert plan.excess_product == (
        plan.additional_product - capacity.minimum_additional_product_factor
    )
    assert plan.require_sufficient() is plan

    running = rns.M
    for modulus in plan.moduli:
        assert gcd(running, modulus) == 1
        running *= modulus


def test_most_balanced_minimum_count_solution_is_23_24_25():
    capacity = _eight_plane_capacity()
    search = rns.search_mersenne_rails_for_capacity(
        capacity,
        max_exponent=31,
        max_rails=3,
    )
    plan = search.most_balanced_plan

    assert plan is not None
    assert plan.exponents == (23, 24, 25)
    assert plan.exponent_span == 2
    assert plan.total_storage_bits == 72
    assert plan.additional_product_bits == 72
    assert plan.sufficient is True


def test_search_can_return_more_than_the_minimum_rail_count():
    search = rns.search_mersenne_rail_sets(
        required_additional_product=10**8,
        min_exponent=2,
        max_exponent=17,
        max_rails=4,
        minimal_rail_count_only=False,
    )

    counts = {plan.rail_count for plan in search.solutions}
    assert counts
    assert min(counts) == search.minimum_rail_count
    assert max(counts) > min(counts)


def test_limit_is_applied_after_exact_ordering():
    capacity = _eight_plane_capacity()
    full = rns.search_mersenne_rails_for_capacity(
        capacity,
        max_exponent=31,
        max_rails=3,
    )
    limited = rns.search_mersenne_rails_for_capacity(
        capacity,
        max_exponent=31,
        max_rails=3,
        limit=3,
    )

    assert limited.solutions == full.solutions[:3]


def test_composite_mersenne_moduli_are_allowed_when_coprime():
    candidate = rns.MersenneRailCandidate(11)

    assert candidate.modulus == 2047
    assert candidate.coprime_with_current_engine is True
    # 2047 = 23 * 89. CRT capacity requires coprimality, not primality.
    assert candidate.modulus == 23 * 89


def test_insufficient_manual_set_is_reported_not_promoted():
    capacity = _eight_plane_capacity()
    plan = rns.MersenneRailSetPlan(
        capacity.minimum_additional_product_factor,
        (
            rns.MersenneRailCandidate(11),
            rns.MersenneRailCandidate(17),
        ),
    )

    assert plan.sufficient is False
    with pytest.raises(OverflowError, match="insufficient"):
        plan.require_sufficient()


def test_non_coprime_mersenne_candidates_are_rejected():
    # Existing rail 127 is 2**7 - 1.
    with pytest.raises(ValueError, match="not coprime"):
        rns.MersenneRailSetPlan(
            2,
            (rns.MersenneRailCandidate(14),),
        )

    # gcd(2**6 - 1, 2**9 - 1) = 2**3 - 1 = 7.
    with pytest.raises(ValueError, match="not coprime"):
        rns.MersenneRailSetPlan(
            2,
            (
                rns.MersenneRailCandidate(6),
                rns.MersenneRailCandidate(9),
            ),
        )


def test_search_inputs_are_strict_and_bounded():
    with pytest.raises(TypeError, match="required_additional_product"):
        rns.search_mersenne_rail_sets(1.5)

    with pytest.raises(ValueError, match="required_additional_product"):
        rns.search_mersenne_rail_sets(0)

    with pytest.raises(ValueError, match="max_exponent"):
        rns.search_mersenne_rail_sets(2, min_exponent=10, max_exponent=9)

    with pytest.raises(ValueError, match="max_rails"):
        rns.search_mersenne_rail_sets(2, max_rails=0)

    with pytest.raises(ValueError, match="limit"):
        rns.search_mersenne_rail_sets(2, limit=0)

    with pytest.raises(TypeError, match="capacity"):
        rns.search_mersenne_rails_for_capacity(object())
