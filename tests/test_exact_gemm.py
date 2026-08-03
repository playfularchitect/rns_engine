import numpy as np
import pytest

import rns_engine as rns


def test_signed_radix_planes_round_trip_exactly():
    values = np.asarray(
        [[0, 1, -1], [128**5 - 1, -(128**5) + 1, 2**50 + 3]],
        dtype=object,
    )
    planes = rns.decompose_signed_radix(values, radix=128, plane_count=8)

    assert planes.dtype == np.int8
    assert planes.shape == (8, 2, 3)
    assert np.array_equal(rns.reconstruct_signed_radix(planes, radix=128), values)
    assert int(np.max(np.abs(planes.astype(np.int16)))) <= 127


def test_grouped_plane_gemm_reconstructs_direct_big_integer_gemm():
    left = np.asarray(
        [[2**40 + 5, -777, 91], [123, -(2**35) + 9, 17]],
        dtype=object,
    )
    right = np.asarray(
        [[7, 2**31 + 1], [-19, 23], [29, -31]],
        dtype=object,
    )
    left_planes = rns.decompose_signed_radix(left, plane_count=8)
    right_planes = rns.decompose_signed_radix(right, plane_count=8)
    grouped = rns.grouped_plane_gemm(left_planes, right_planes)

    assert grouped.dtype == np.int32
    assert grouped.shape == (15, 2, 2)
    assert np.array_equal(
        rns.reconstruct_grouped_partials(grouped),
        rns.exact_integer_matmul(left, right),
    )


def test_shared_scale_exact_gemm_uses_wide_rns_and_preserves_scale():
    left = rns.SharedScaleMatrix(
        np.asarray([[2**42 + 17, -(2**35) + 5], [77, 999999]], dtype=object),
        scale=6,
    )
    right = rns.SharedScaleMatrix(
        np.asarray([[17, -(2**31) + 1], [2**33 + 9, 41]], dtype=object),
        scale=35,
    )
    config = rns.WideRNSConfig.balanced_seven_rail()

    receipt = rns.exact_shared_scale_gemm(
        left,
        right,
        config=config,
        left_plane_count=8,
        right_plane_count=8,
    )

    assert receipt.exact_match
    assert receipt.unique
    assert receipt.output_scale == 210
    assert receipt.local_capacity.safe
    assert receipt.global_capacity.unique
    result = receipt.as_matrix()
    assert result.scale == 210
    assert np.array_equal(result.numerators, rns.exact_integer_matmul(left.numerators, right.numerators))


def test_shared_scale_reduction_only_removes_global_common_factor():
    matrix = rns.SharedScaleMatrix(np.asarray([[6, 12], [-18, 24]], dtype=object), 30)
    reduced = matrix.reduced()

    assert reduced.scale == 5
    assert np.array_equal(reduced.numerators, np.asarray([[1, 2], [-3, 4]], dtype=object))


def test_grouped_gemm_refuses_actual_int32_overflow():
    left = np.asarray([[[127] * 200000]], dtype=np.int8)
    right = np.asarray([[[127]] for _ in range(200000)], dtype=np.int8).reshape(1, 200000, 1)

    with pytest.raises(OverflowError, match="INT32"):
        rns.grouped_plane_gemm(left, right)
