from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Any, Iterable, Sequence


def _require_integer(value: Any, *, name: str) -> int:
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _require_nonnegative_integer(value: Any, *, name: str) -> int:
    integer = _require_integer(value, name=name)
    if integer < 0:
        raise ValueError(f"{name} must be >= 0")
    return integer


def _nonnegative_tuple(values: Iterable[Any], *, name: str) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of nonnegative integers") from exc

    return tuple(
        _require_nonnegative_integer(value, name=f"{name}[{position}]")
        for position, value in enumerate(raw)
    )


def _ceil_log2(value: int) -> int:
    if value <= 1:
        return 0
    return (value - 1).bit_length()


@dataclass(frozen=True, slots=True)
class GroupedCoefficientCapacityPlan:
    """Exact local-accumulator receipt for grouped digit-plane products.

    For raw positional coefficients

    ``R[k] = sum_{i+j=k} GEMM(A_i, B_j)``,

    each contributing plane-pair GEMM has magnitude bound

    ``inner_dimension * left_bound[i] * right_bound[j]``.

    This receipt answers whether those grouped coefficients fit a signed native
    accumulator such as INT32. It deliberately does not answer whether the
    later radix-weighted wide reconstruction fits the global RNS modulus; that
    is handled by :class:`DigitPlaneGemmCapacityPlan`.
    """

    inner_dimension: int
    left_digit_abs_bounds: tuple[int, ...]
    right_digit_abs_bounds: tuple[int, ...]
    accumulator_bits: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "inner_dimension",
            _require_nonnegative_integer(
                self.inner_dimension,
                name="inner_dimension",
            ),
        )
        object.__setattr__(
            self,
            "left_digit_abs_bounds",
            _nonnegative_tuple(
                self.left_digit_abs_bounds,
                name="left_digit_abs_bounds",
            ),
        )
        object.__setattr__(
            self,
            "right_digit_abs_bounds",
            _nonnegative_tuple(
                self.right_digit_abs_bounds,
                name="right_digit_abs_bounds",
            ),
        )
        bits = _require_integer(self.accumulator_bits, name="accumulator_bits")
        if bits < 1:
            raise ValueError("accumulator_bits must be >= 1")
        object.__setattr__(self, "accumulator_bits", bits)

    @property
    def coefficient_count(self) -> int:
        if not self.left_digit_abs_bounds or not self.right_digit_abs_bounds:
            return 0
        return len(self.left_digit_abs_bounds) + len(self.right_digit_abs_bounds) - 1

    @property
    def plane_pair_count(self) -> int:
        return len(self.left_digit_abs_bounds) * len(self.right_digit_abs_bounds)

    @property
    def coefficient_pair_counts(self) -> tuple[int, ...]:
        return tuple(
            sum(
                1
                for left_position in range(len(self.left_digit_abs_bounds))
                for right_position in range(len(self.right_digit_abs_bounds))
                if left_position + right_position == coefficient_position
            )
            for coefficient_position in range(self.coefficient_count)
        )

    @property
    def coefficient_abs_bounds(self) -> tuple[int, ...]:
        return tuple(
            sum(
                self.inner_dimension
                * self.left_digit_abs_bounds[left_position]
                * self.right_digit_abs_bounds[right_position]
                for left_position in range(len(self.left_digit_abs_bounds))
                for right_position in range(len(self.right_digit_abs_bounds))
                if left_position + right_position == coefficient_position
            )
            for coefficient_position in range(self.coefficient_count)
        )

    @property
    def max_abs_bound(self) -> int:
        return max(self.coefficient_abs_bounds, default=0)

    @property
    def signed_accumulator_limit(self) -> int:
        return (1 << (self.accumulator_bits - 1)) - 1

    @property
    def safe(self) -> bool:
        return self.max_abs_bound <= self.signed_accumulator_limit

    @property
    def headroom(self) -> int:
        return self.signed_accumulator_limit - self.max_abs_bound

    @property
    def minimum_signed_accumulator_bits(self) -> int:
        return 1 + _ceil_log2(self.max_abs_bound + 1)

    def require_safe(self) -> GroupedCoefficientCapacityPlan:
        if not self.safe:
            raise OverflowError(
                "grouped coefficient exceeds the signed accumulator: bound "
                f"{self.max_abs_bound} is greater than the "
                f"{self.accumulator_bits}-bit limit "
                f"{self.signed_accumulator_limit}"
            )
        return self


def plan_grouped_coefficient_capacity(
    inner_dimension: int,
    left_digit_abs_bounds: Sequence[int] | Iterable[int],
    right_digit_abs_bounds: Sequence[int] | Iterable[int],
    accumulator_bits: int = 32,
) -> GroupedCoefficientCapacityPlan:
    """Build an exact local grouped-coefficient accumulator receipt."""

    return GroupedCoefficientCapacityPlan(
        inner_dimension=inner_dimension,
        left_digit_abs_bounds=tuple(left_digit_abs_bounds),
        right_digit_abs_bounds=tuple(right_digit_abs_bounds),
        accumulator_bits=accumulator_bits,
    )


__all__ = [
    "GroupedCoefficientCapacityPlan",
    "plan_grouped_coefficient_capacity",
]
