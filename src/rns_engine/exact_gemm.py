from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from operator import index
from typing import Any

import numpy as np

from .capacity import DigitPlaneGemmCapacityPlan, plan_digit_plane_gemm_capacity
from .coefficients import (
    GroupedCoefficientCapacityPlan,
    plan_grouped_coefficient_capacity,
)
from .wide import (
    WideRNSConfig,
    WideWeightedResult,
    accumulate_weighted_int32_wide,
)


def _require_int(value: Any, *, name: str) -> int:
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _integer_array(values: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    for position, value in enumerate(array.flat):
        try:
            index(value)
        except TypeError as exc:
            raise TypeError(f"{name}.flat[{position}] must be an integer") from exc
    return array


def exact_integer_matmul(left: Any, right: Any) -> np.ndarray:
    """Matrix multiplication using Python integers only."""

    lhs = _integer_array(left, name="left", ndim=2)
    rhs = _integer_array(right, name="right", ndim=2)
    if lhs.shape[1] != rhs.shape[0]:
        raise ValueError(
            "inner dimensions must match "
            f"({lhs.shape[1]} != {rhs.shape[0]})"
        )

    output = np.empty((lhs.shape[0], rhs.shape[1]), dtype=object)
    for row in range(lhs.shape[0]):
        for column in range(rhs.shape[1]):
            total = 0
            for inner in range(lhs.shape[1]):
                total += int(lhs[row, inner]) * int(rhs[inner, column])
            output[row, column] = total
    return output


def decompose_signed_radix(
    values: Any,
    *,
    radix: int = 128,
    plane_count: int | None = None,
) -> np.ndarray:
    """Decompose signed integers into sign-magnitude radix planes.

    Each value's magnitude is expanded in base ``radix`` and the original sign
    is applied to every nonzero digit. With ``radix <= 128`` every digit fits
    exactly in signed INT8, matching the intended Tensor Core input contract.
    """

    radix = _require_int(radix, name="radix")
    if not 2 <= radix <= 128:
        raise ValueError("radix must be in [2, 128] for INT8 digit planes")
    integers = _integer_array(values, name="values")

    max_abs = max((abs(int(value)) for value in integers.flat), default=0)
    required_planes = 1
    remaining = max_abs
    while remaining >= radix:
        remaining //= radix
        required_planes += 1

    if plane_count is None:
        planes = required_planes
    else:
        planes = _require_int(plane_count, name="plane_count")
        if planes <= 0:
            raise ValueError("plane_count must be > 0")
        if planes < required_planes:
            raise OverflowError(
                f"plane_count={planes} cannot represent magnitude {max_abs} "
                f"in radix {radix}; need {required_planes}"
            )

    output = np.zeros((planes, *integers.shape), dtype=np.int8)
    for offset, raw in enumerate(integers.flat):
        value = int(raw)
        sign = -1 if value < 0 else 1
        magnitude = abs(value)
        coordinate = np.unravel_index(offset, integers.shape)
        for plane in range(planes):
            digit = magnitude % radix
            magnitude //= radix
            output[(plane, *coordinate)] = sign * digit
    return output


def reconstruct_signed_radix(planes: np.ndarray, *, radix: int = 128) -> np.ndarray:
    digits = np.asarray(planes)
    if digits.ndim < 1:
        raise ValueError("planes must have a leading plane dimension")
    if not np.issubdtype(digits.dtype, np.signedinteger):
        raise TypeError("planes must use a signed integer dtype")
    radix = _require_int(radix, name="radix")
    if radix < 2:
        raise ValueError("radix must be >= 2")

    output = np.zeros(digits.shape[1:], dtype=object)
    weight = 1
    for plane in digits:
        output = output + plane.astype(object, copy=False) * weight
        weight *= radix
    return output


def grouped_plane_gemm(
    left_planes: np.ndarray,
    right_planes: np.ndarray,
    *,
    require_int32: bool = True,
) -> np.ndarray:
    """Compute grouped digit-plane GEMM coefficients exactly.

    Output coefficient ``k`` equals ``sum_{i+j=k} A_i @ B_j``. This is the
    precise signed INT32 body expected from the first CUDA integration stage.
    """

    left = np.asarray(left_planes)
    right = np.asarray(right_planes)
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError("left_planes and right_planes must have shape (planes, rows, cols)")
    if left.shape[2] != right.shape[1]:
        raise ValueError("plane GEMM inner dimensions must match")
    if not np.issubdtype(left.dtype, np.signedinteger):
        raise TypeError("left_planes must use a signed integer dtype")
    if not np.issubdtype(right.dtype, np.signedinteger):
        raise TypeError("right_planes must use a signed integer dtype")

    grouped_count = left.shape[0] + right.shape[0] - 1
    grouped = [
        np.zeros((left.shape[1], right.shape[2]), dtype=object)
        for _ in range(grouped_count)
    ]
    for left_index, left_plane in enumerate(left):
        for right_index, right_plane in enumerate(right):
            grouped[left_index + right_index] += exact_integer_matmul(
                left_plane,
                right_plane,
            )

    object_output = np.stack(grouped, axis=0)
    if not require_int32:
        return object_output

    int32_min = np.iinfo(np.int32).min
    int32_max = np.iinfo(np.int32).max
    for value in object_output.flat:
        integer = int(value)
        if integer < int32_min or integer > int32_max:
            raise OverflowError(
                f"grouped coefficient {integer} does not fit signed INT32"
            )
    return object_output.astype(np.int32)


def reconstruct_grouped_partials(
    grouped_partials: np.ndarray,
    *,
    radix: int = 128,
) -> np.ndarray:
    partials = np.asarray(grouped_partials)
    if partials.ndim < 1:
        raise ValueError("grouped_partials must have a leading coefficient dimension")
    radix = _require_int(radix, name="radix")
    if radix < 2:
        raise ValueError("radix must be >= 2")

    output = np.zeros(partials.shape[1:], dtype=object)
    weight = 1
    for partial in partials:
        output += partial.astype(object, copy=False) * weight
        weight *= radix
    return output


@dataclass(frozen=True, slots=True)
class SharedScaleMatrix:
    """Exact matrix represented as one integer numerator matrix / one scale."""

    numerators: np.ndarray
    scale: int = 1

    def __post_init__(self) -> None:
        numerators = _integer_array(self.numerators, name="numerators", ndim=2)
        scale = _require_int(self.scale, name="scale")
        if scale <= 0:
            raise ValueError("scale must be > 0")
        object.__setattr__(self, "numerators", numerators)
        object.__setattr__(self, "scale", scale)

    @property
    def shape(self) -> tuple[int, int]:
        return self.numerators.shape

    def reduced(self) -> SharedScaleMatrix:
        common = self.scale
        for value in self.numerators.flat:
            common = gcd(common, abs(int(value)))
            if common == 1:
                return self
        if common <= 1:
            return self
        return SharedScaleMatrix(self.numerators // common, self.scale // common)


@dataclass(frozen=True, slots=True)
class ExactSharedScaleGemmReceipt:
    left: SharedScaleMatrix
    right: SharedScaleMatrix
    radix: int
    config: WideRNSConfig
    left_planes: np.ndarray
    right_planes: np.ndarray
    grouped_partials: np.ndarray
    weighted_result: WideWeightedResult
    direct_numerator: np.ndarray
    reconstructed_numerator: np.ndarray
    output_scale: int
    local_capacity: GroupedCoefficientCapacityPlan
    global_capacity: DigitPlaneGemmCapacityPlan

    @property
    def exact_match(self) -> bool:
        return np.array_equal(self.direct_numerator, self.reconstructed_numerator)

    @property
    def unique(self) -> bool:
        return self.weighted_result.unique and self.global_capacity.unique

    def require_exact(self) -> ExactSharedScaleGemmReceipt:
        if not self.exact_match:
            raise AssertionError("RNS reconstruction does not match direct integer GEMM")
        if not self.unique:
            raise OverflowError("the result is exact modulo the rail product but not uniquely signed")
        return self

    def as_matrix(self, *, reduce: bool = False) -> SharedScaleMatrix:
        self.require_exact()
        matrix = SharedScaleMatrix(self.reconstructed_numerator, self.output_scale)
        return matrix.reduced() if reduce else matrix


def exact_shared_scale_gemm(
    left: SharedScaleMatrix,
    right: SharedScaleMatrix,
    *,
    config: WideRNSConfig,
    radix: int = 128,
    left_plane_count: int | None = None,
    right_plane_count: int | None = None,
    require_unique: bool = True,
) -> ExactSharedScaleGemmReceipt:
    if left.shape[1] != right.shape[0]:
        raise ValueError("matrix inner dimensions must match")

    left_planes = decompose_signed_radix(
        left.numerators,
        radix=radix,
        plane_count=left_plane_count,
    )
    right_planes = decompose_signed_radix(
        right.numerators,
        radix=radix,
        plane_count=right_plane_count,
    )
    grouped = grouped_plane_gemm(left_planes, right_planes, require_int32=True)
    weights = tuple(radix**position for position in range(grouped.shape[0]))
    weighted = accumulate_weighted_int32_wide(
        grouped,
        weights,
        config,
        require_unique=require_unique,
    )
    reconstructed = weighted.decode_signed(require_unique=require_unique)
    direct = exact_integer_matmul(left.numerators, right.numerators)

    left_bounds = tuple(
        max((abs(int(value)) for value in plane.flat), default=0)
        for plane in left_planes
    )
    right_bounds = tuple(
        max((abs(int(value)) for value in plane.flat), default=0)
        for plane in right_planes
    )
    local = plan_grouped_coefficient_capacity(
        inner_dimension=left.shape[1],
        left_digit_abs_bounds=left_bounds,
        right_digit_abs_bounds=right_bounds,
        accumulator_bits=32,
    )
    global_plan = plan_digit_plane_gemm_capacity(
        inner_dimension=left.shape[1],
        radix=radix,
        left_digit_abs_bounds=left_bounds,
        right_digit_abs_bounds=right_bounds,
        additional_moduli=config.extra_moduli,
    )

    receipt = ExactSharedScaleGemmReceipt(
        left=left,
        right=right,
        radix=radix,
        config=config,
        left_planes=left_planes,
        right_planes=right_planes,
        grouped_partials=grouped,
        weighted_result=weighted,
        direct_numerator=direct,
        reconstructed_numerator=reconstructed,
        output_scale=left.scale * right.scale,
        local_capacity=local,
        global_capacity=global_plan,
    )
    if require_unique:
        receipt.require_exact()
    return receipt
