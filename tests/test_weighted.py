import numpy as np
import pytest

import rns_engine as rns


def test_weighted_int32_matrix_matches_direct_integer_sum():
    partials = np.array(
        [
            [[1, -2], [3, -4]],
            [[5, 6], [-7, -8]],
            [[-9, 10], [11, -12]],
        ],
        dtype=np.int32,
    )
    weights = [1, -3, 8]

    receipt = rns.accumulate_weighted_int32(
        partials,
        weights,
        require_unique=True,
    )
    expected = sum(
        int(weight) * partial.astype(np.int64)
        for weight, partial in zip(weights, partials)
    )

    np.testing.assert_array_equal(receipt.decode_signed(), expected)
    assert receipt.output_shape == (2, 2)
    assert receipt.term_count == 3
    assert receipt.output_size == 4
    assert receipt.weights == (1, -3, 8)
    assert receipt.term_abs_bounds == (4, 8, 12)
    assert receipt.certificate.max_abs_bound == 124
    assert receipt.certificate.unique is True


def test_weighted_scalar_output_preserves_shape():
    partials = np.array([7, -3, 5], dtype=np.int32)

    receipt = rns.accumulate_weighted_int32(partials, [2, 4, -1])

    assert receipt.output_shape == ()
    assert receipt.decode_signed().shape == ()
    assert int(receipt.decode_signed()) == -3


def test_int32_minimum_has_exact_unsigned_magnitude_bound():
    partials = np.array([[-2_147_483_648]], dtype=np.int32)

    receipt = rns.accumulate_weighted_int32(partials, [1])

    assert receipt.term_abs_bounds == (2_147_483_648,)
    assert receipt.certificate.max_abs_bound == 2_147_483_648
    np.testing.assert_array_equal(
        receipt.decode_signed(),
        np.array([-2_147_483_648], dtype=np.int64),
    )


def test_arbitrary_precision_weights_are_reduced_only_for_native_rails():
    weight = 2**100 + 123_456_789
    partials = np.array([[1, -1]], dtype=np.int32)

    receipt = rns.accumulate_weighted_int32(partials, [weight])

    assert receipt.weights == (weight,)
    assert receipt.certificate.max_abs_bound == weight
    assert receipt.certificate.unique is False

    expected_modular = np.array(
        [weight % rns.M, (-weight) % rns.M],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(receipt.decode_modular(), expected_modular)


def test_require_unique_rejects_modular_only_weighted_sum_before_claiming_exact_integer():
    partials = np.array([[1]], dtype=np.int32)

    with pytest.raises(OverflowError, match="modular-only"):
        rns.accumulate_weighted_int32(
            partials,
            [rns.HALF_M],
            require_unique=True,
        )

    receipt = rns.accumulate_weighted_int32(partials, [rns.HALF_M])
    assert receipt.certificate.unique is False
    with pytest.raises(OverflowError, match="modular-only"):
        receipt.decode_signed()

    np.testing.assert_array_equal(
        receipt.decode_signed(require_unique=False),
        np.array([rns.SIGNED_MIN], dtype=np.int64),
    )


def test_weighted_sum_bound_uses_full_weights_and_declared_bounds():
    certificate = rns.certify_weighted_sum_bound(
        weights=[1, -16, 256],
        term_abs_bounds=[127, 1_000, 82_580_480],
        addend_abs_bound=9,
    )

    assert certificate.max_abs_bound == 127 + 16_000 + 256 * 82_580_480 + 9
    assert certificate.minimum_required_modulus == 2 * certificate.max_abs_bound + 1


def test_empty_term_axis_produces_exact_zero_receipt():
    partials = np.empty((0, 2, 3), dtype=np.int32)

    receipt = rns.accumulate_weighted_int32(partials, [])

    np.testing.assert_array_equal(
        receipt.decode_signed(),
        np.zeros((2, 3), dtype=np.int64),
    )
    assert receipt.term_abs_bounds == ()
    assert receipt.certificate.max_abs_bound == 0
    assert receipt.certificate.unique is True


def test_zero_weight_skips_term_without_changing_bound_law():
    partials = np.array([[2_000_000_000, -2_000_000_000], [3, 4]], dtype=np.int32)

    receipt = rns.accumulate_weighted_int32(partials, [0, 7])

    assert receipt.term_abs_bounds == (2_000_000_000, 4)
    assert receipt.certificate.max_abs_bound == 28
    np.testing.assert_array_equal(
        receipt.decode_signed(),
        np.array([21, 28], dtype=np.int64),
    )


def test_weighted_accumulator_validates_shape_dtype_and_weight_count():
    with pytest.raises(ValueError, match="shape"):
        rns.accumulate_weighted_int32(np.array(7, dtype=np.int32), [1])

    with pytest.raises(TypeError, match="dtype int32"):
        rns.accumulate_weighted_int32(np.array([[1]], dtype=np.int64), [1])

    with pytest.raises(ValueError, match="weights length"):
        rns.accumulate_weighted_int32(np.array([[1], [2]], dtype=np.int32), [1])

    with pytest.raises(TypeError, match=r"weights\[0\]"):
        rns.accumulate_weighted_int32(np.array([[1]], dtype=np.int32), [1.5])


def test_weighted_bound_validates_lengths_and_nonnegative_term_bounds():
    with pytest.raises(ValueError, match="same length"):
        rns.certify_weighted_sum_bound([1, 2], [3])

    with pytest.raises(ValueError, match=r"term_abs_bounds\[0\]"):
        rns.certify_weighted_sum_bound([1], [-1])

    with pytest.raises(TypeError, match="addend_abs_bound"):
        rns.certify_weighted_sum_bound([1], [1], addend_abs_bound=1.5)


def test_fused_weighted_kernel_matches_staged_reference_randomized():
    from rns_engine.weighted import (
        _accumulate_weighted_int32_staged,
        _validate_weighted_inputs,
    )

    rng = np.random.default_rng(20260802)
    for terms, outputs in [(0, 7), (1, 1), (2, 17), (5, 257), (9, 1024)]:
        partials = rng.integers(
            np.iinfo(np.int32).min,
            np.iinfo(np.int32).max,
            size=(terms, outputs),
            dtype=np.int32,
        )
        weights = tuple(
            ((-1) ** position) * (2 ** (13 * position) + 17)
            for position in range(terms)
        )

        fused = rns.accumulate_weighted_int32(partials, weights)
        flat, exact_weights, output_shape, output_size, _ = _validate_weighted_inputs(
            partials,
            weights,
        )
        staged_rails, staged_bounds = _accumulate_weighted_int32_staged(
            flat,
            exact_weights,
            output_size,
        )

        assert fused.output_shape == output_shape
        assert fused.term_abs_bounds == staged_bounds
        for fused_rail, staged_rail in zip(fused.encoded.rails(), staged_rails):
            np.testing.assert_array_equal(fused_rail, staged_rail)


def test_fused_kernel_handles_noncontiguous_input_without_changing_result():
    base = np.arange(4 * 20, dtype=np.int32).reshape(4, 20)
    partials = base[:, ::2]
    assert not partials.flags.c_contiguous

    receipt = rns.accumulate_weighted_int32(partials, [1, -2, 3, -4])
    expected = sum(
        weight * term.astype(np.int64)
        for weight, term in zip([1, -2, 3, -4], partials)
    )

    np.testing.assert_array_equal(receipt.decode_signed(), expected)
