from __future__ import annotations

from dataclasses import dataclass
from math import gcd, prod
from operator import index
from typing import Any, Iterable, Sequence

from ._core import M


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


def _require_radix(value: Any) -> int:
    radix = _require_integer(value, name="radix")
    if radix < 2:
        raise ValueError("radix must be >= 2")
    return radix


def _nonnegative_tuple(values: Iterable[Any], *, name: str) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of nonnegative integers") from exc

    return tuple(
        _require_nonnegative_integer(value, name=f"{name}[{position}]")
        for position, value in enumerate(raw)
    )


def _integer_tuple(values: Iterable[Any], *, name: str) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of integers") from exc

    return tuple(
        _require_integer(value, name=f"{name}[{position}]")
        for position, value in enumerate(raw)
    )


def _validate_additional_moduli(values: Iterable[Any]) -> tuple[int, ...]:
    moduli = _integer_tuple(values, name="additional_moduli")
    running_product = int(M)

    for position, modulus in enumerate(moduli):
        if modulus <= 1:
            raise ValueError(f"additional_moduli[{position}] must be > 1")

        common_factor = gcd(running_product, modulus)
        if common_factor != 1:
            raise ValueError(
                f"additional_moduli[{position}]={modulus} must be coprime with "
                f"the existing modulus product; gcd={common_factor}"
            )
        running_product *= modulus

    return moduli


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True, slots=True)
class SignedCapacityPlan:
    """Exact modulus-capacity receipt for a symmetric signed result bound.

    The current engine uniquely represents every integer in ``[-bound, bound]``
    only when its modulus is at least ``2 * bound + 1``. Optional additional
    moduli are treated as proposed CRT rails and must be pairwise coprime with
    the current four-rail modulus product and with one another.
    """

    max_abs_bound: int
    additional_moduli: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        bound = _require_nonnegative_integer(
            self.max_abs_bound,
            name="max_abs_bound",
        )
        moduli = _validate_additional_moduli(self.additional_moduli)
        object.__setattr__(self, "max_abs_bound", bound)
        object.__setattr__(self, "additional_moduli", moduli)

    @property
    def current_modulus(self) -> int:
        return int(M)

    @property
    def additional_modulus_product(self) -> int:
        return prod(self.additional_moduli, start=1)

    @property
    def available_modulus(self) -> int:
        return self.current_modulus * self.additional_modulus_product

    @property
    def minimum_required_modulus(self) -> int:
        return 2 * self.max_abs_bound + 1

    @property
    def current_unique(self) -> bool:
        return self.current_modulus >= self.minimum_required_modulus

    @property
    def unique(self) -> bool:
        return self.available_modulus >= self.minimum_required_modulus

    @property
    def current_headroom(self) -> int:
        return (self.current_modulus - 1) // 2 - self.max_abs_bound

    @property
    def available_headroom(self) -> int:
        return (self.available_modulus - 1) // 2 - self.max_abs_bound

    @property
    def modulus_shortfall(self) -> int:
        return max(0, self.minimum_required_modulus - self.available_modulus)

    @property
    def minimum_additional_product_factor(self) -> int:
        """Smallest exact product factor needed beyond the current modulus."""

        return max(
            1,
            _ceil_div(self.minimum_required_modulus, self.current_modulus),
        )

    @property
    def minimum_single_coprime_modulus(self) -> int:
        """Smallest one-rail modulus that closes the current capacity deficit.

        This is a mathematical CRT-capacity result, not a claim that the
        returned modulus has efficient reduction or hardware properties.
        """

        candidate = self.minimum_additional_product_factor
        if candidate <= 1:
            return 1
        while gcd(candidate, self.current_modulus) != 1:
            candidate += 1
        return candidate

    @property
    def additional_bits_required(self) -> int:
        """Power-of-two bit budget needed beyond the current modulus product."""

        factor = self.minimum_additional_product_factor
        return 0 if factor <= 1 else (factor - 1).bit_length()

    @property
    def required_modulus_bits(self) -> int:
        return self.minimum_required_modulus.bit_length()

    @property
    def available_modulus_bits(self) -> int:
        return self.available_modulus.bit_length()

    def require_unique(self) -> SignedCapacityPlan:
        if not self.unique:
            raise OverflowError(
                "signed result is modular-only: available modulus "
                f"{self.available_modulus} is smaller than the required modulus "
                f"{self.minimum_required_modulus} for absolute bound "
                f"{self.max_abs_bound}"
            )
        return self


def plan_signed_capacity(
    max_abs_bound: int,
    additional_moduli: Sequence[int] | Iterable[int] = (),
) -> SignedCapacityPlan:
    """Plan exact signed capacity for a known absolute result bound."""

    return SignedCapacityPlan(
        max_abs_bound=max_abs_bound,
        additional_moduli=tuple(additional_moduli),
    )


def plan_weighted_sum_capacity(
    weights: Sequence[int] | Iterable[int],
    term_abs_bounds: Sequence[int] | Iterable[int],
    addend_abs_bound: int = 0,
    additional_moduli: Sequence[int] | Iterable[int] = (),
) -> SignedCapacityPlan:
    """Plan capacity for ``sum(weight[t] * term[t]) + addend`` exactly."""

    exact_weights = _integer_tuple(weights, name="weights")
    exact_bounds = _nonnegative_tuple(term_abs_bounds, name="term_abs_bounds")
    if len(exact_weights) != len(exact_bounds):
        raise ValueError(
            "weights and term_abs_bounds must have the same length "
            f"({len(exact_weights)} != {len(exact_bounds)})"
        )

    addend = _require_nonnegative_integer(
        addend_abs_bound,
        name="addend_abs_bound",
    )
    bound = addend + sum(
        abs(weight) * term_bound
        for weight, term_bound in zip(exact_weights, exact_bounds)
    )
    return plan_signed_capacity(bound, additional_moduli)


@dataclass(frozen=True, slots=True)
class DigitPlaneGemmCapacityPlan:
    """Exact worst-case capacity plan for positional digit-plane GEMM.

    If ``A = sum(A_i * radix**i)`` and ``B = sum(B_j * radix**j)``, with
    per-plane entry magnitude bounds ``left_digit_abs_bounds[i]`` and
    ``right_digit_abs_bounds[j]``, then one output entry obeys::

        |C| <= K * left_value_abs_bound * right_value_abs_bound + addend

    This is algebraically identical to summing every plane-pair GEMM with its
    positional weight ``radix**(i + j)``. No floating point is used.
    """

    inner_dimension: int
    radix: int
    left_digit_abs_bounds: tuple[int, ...]
    right_digit_abs_bounds: tuple[int, ...]
    addend_abs_bound: int = 0
    additional_moduli: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "inner_dimension",
            _require_nonnegative_integer(
                self.inner_dimension,
                name="inner_dimension",
            ),
        )
        object.__setattr__(self, "radix", _require_radix(self.radix))
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
        object.__setattr__(
            self,
            "addend_abs_bound",
            _require_nonnegative_integer(
                self.addend_abs_bound,
                name="addend_abs_bound",
            ),
        )
        object.__setattr__(
            self,
            "additional_moduli",
            _validate_additional_moduli(self.additional_moduli),
        )

    @property
    def left_plane_count(self) -> int:
        return len(self.left_digit_abs_bounds)

    @property
    def right_plane_count(self) -> int:
        return len(self.right_digit_abs_bounds)

    @property
    def plane_pair_count(self) -> int:
        return self.left_plane_count * self.right_plane_count

    @property
    def grouped_coefficient_count(self) -> int:
        if self.left_plane_count == 0 or self.right_plane_count == 0:
            return 0
        return self.left_plane_count + self.right_plane_count - 1

    @property
    def left_value_abs_bound(self) -> int:
        return sum(
            digit_bound * self.radix**position
            for position, digit_bound in enumerate(self.left_digit_abs_bounds)
        )

    @property
    def right_value_abs_bound(self) -> int:
        return sum(
            digit_bound * self.radix**position
            for position, digit_bound in enumerate(self.right_digit_abs_bounds)
        )

    @property
    def plane_pair_abs_bounds(self) -> tuple[int, ...]:
        return tuple(
            self.inner_dimension * left_bound * right_bound
            for left_bound in self.left_digit_abs_bounds
            for right_bound in self.right_digit_abs_bounds
        )

    @property
    def plane_pair_weights(self) -> tuple[int, ...]:
        return tuple(
            self.radix ** (left_position + right_position)
            for left_position in range(self.left_plane_count)
            for right_position in range(self.right_plane_count)
        )

    @property
    def max_abs_bound(self) -> int:
        return (
            self.inner_dimension
            * self.left_value_abs_bound
            * self.right_value_abs_bound
            + self.addend_abs_bound
        )

    @property
    def capacity(self) -> SignedCapacityPlan:
        return plan_signed_capacity(
            self.max_abs_bound,
            self.additional_moduli,
        )

    @property
    def current_unique(self) -> bool:
        return self.capacity.current_unique

    @property
    def unique(self) -> bool:
        return self.capacity.unique

    def require_unique(self) -> DigitPlaneGemmCapacityPlan:
        self.capacity.require_unique()
        return self


def plan_digit_plane_gemm_capacity(
    inner_dimension: int,
    radix: int,
    left_digit_abs_bounds: Sequence[int] | Iterable[int],
    right_digit_abs_bounds: Sequence[int] | Iterable[int],
    addend_abs_bound: int = 0,
    additional_moduli: Sequence[int] | Iterable[int] = (),
) -> DigitPlaneGemmCapacityPlan:
    """Build an exact positional digit-plane GEMM capacity receipt."""

    return DigitPlaneGemmCapacityPlan(
        inner_dimension=inner_dimension,
        radix=radix,
        left_digit_abs_bounds=tuple(left_digit_abs_bounds),
        right_digit_abs_bounds=tuple(right_digit_abs_bounds),
        addend_abs_bound=addend_abs_bound,
        additional_moduli=tuple(additional_moduli),
    )


__all__ = [
    "DigitPlaneGemmCapacityPlan",
    "SignedCapacityPlan",
    "plan_digit_plane_gemm_capacity",
    "plan_signed_capacity",
    "plan_weighted_sum_capacity",
]
