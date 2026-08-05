"""Public G4 exact range tiers.

The tier number names the familiar IEEE binary floating-point *range envelope*.
It does not expose or constrain the hidden physical representation.

Strict range checking is the default. Automatic promotion exists only as an
explicit opt-in for advanced callers.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
import math
import operator
from typing import Any, Callable, Iterator

import numpy as np


class ExactRangeError(ArithmeticError):
    """Raised when an exact nonzero value leaves its declared G4 tier."""


class ExactValueError(ValueError):
    """Raised when a value has no lawful exact G4 interpretation."""


class PromotionPolicy(str, Enum):
    STRICT = "strict"
    PROMOTE = "promote"


_DEFAULT_POLICY: ContextVar[PromotionPolicy] = ContextVar(
    "rns_engine_g4_promotion_policy",
    default=PromotionPolicy.STRICT,
)


def _power_of_two(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(1 << exponent, 1)
    return Fraction(1, 1 << (-exponent))


def as_fraction(value: Any) -> Fraction:
    """Return the exact rational value represented by ``value``.

    Finite Python and NumPy floating values are decoded exactly from their
    binary value. Decimal and string inputs retain their exact decimal value.
    NaN and infinities are rejected because G4 tiers represent exact numbers.
    """

    if isinstance(value, G4Scalar):
        return value.value
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ExactValueError("G4 exact tiers reject NaN and infinity")
        return Fraction(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return Fraction(int(value), 1)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ExactValueError("G4 exact tiers reject NaN and infinity")
        numerator, denominator = numeric.as_integer_ratio()
        return Fraction(numerator, denominator)
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise ExactValueError(f"cannot parse exact value {value!r}") from exc
    if isinstance(value, bool):
        return Fraction(int(value), 1)
    try:
        return Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ExactValueError(f"cannot convert {type(value).__name__} to an exact value") from exc


@dataclass(frozen=True, slots=True)
class G4Tier:
    """Stable public exact range contract."""

    name: str
    bits: int | None
    exponent_bits: int | None
    fraction_bits: int | None
    exponent_bias: int | None

    @property
    def bounded(self) -> bool:
        return self.bits is not None

    @property
    def max_finite(self) -> Fraction | None:
        if not self.bounded:
            return None
        assert self.exponent_bits is not None
        assert self.fraction_bits is not None
        assert self.exponent_bias is not None
        maximum_exponent = ((1 << self.exponent_bits) - 2) - self.exponent_bias
        significand = Fraction((1 << (self.fraction_bits + 1)) - 1, 1 << self.fraction_bits)
        return significand * _power_of_two(maximum_exponent)

    @property
    def min_positive(self) -> Fraction | None:
        if not self.bounded:
            return None
        assert self.fraction_bits is not None
        assert self.exponent_bias is not None
        return _power_of_two(1 - self.exponent_bias - self.fraction_bits)

    @property
    def numpy_float_dtype(self) -> np.dtype[Any] | None:
        if self.bits == 16:
            return np.dtype(np.float16)
        if self.bits == 32:
            return np.dtype(np.float32)
        if self.bits == 64:
            return np.dtype(np.float64)
        return None

    def contains(self, value: Any) -> bool:
        exact = as_fraction(value)
        if not self.bounded or exact == 0:
            return True
        magnitude = abs(exact)
        minimum = self.min_positive
        maximum = self.max_finite
        assert minimum is not None and maximum is not None
        return minimum <= magnitude <= maximum

    def require_contains(self, value: Any, *, label: str = "value") -> Fraction:
        exact = as_fraction(value)
        if not self.contains(exact):
            minimum = self.min_positive
            maximum = self.max_finite
            raise ExactRangeError(
                f"{label}={exact} is outside {self.name}: "
                f"nonzero magnitude must be in [{minimum}, {maximum}]"
            )
        return exact

    def __call__(self, value: Any) -> "G4Scalar":
        return G4Scalar(value=as_fraction(value), tier=self)

    def __repr__(self) -> str:
        return self.name


G416 = G4Tier("G416", 16, 5, 10, 15)
G432 = G4Tier("G432", 32, 8, 23, 127)
G464 = G4Tier("G464", 64, 11, 52, 1023)
G4X = G4Tier("G4X", None, None, None, None)

exact16 = G416
exact32 = G432
exact64 = G464
exact = G4X

_BOUNDED_TIERS: tuple[G4Tier, ...] = (G416, G432, G464)
_TIER_ALIASES: dict[str, G4Tier] = {
    "g416": G416,
    "exact16": G416,
    "float16_exact": G416,
    "g432": G432,
    "exact32": G432,
    "float32_exact": G432,
    "g464": G464,
    "exact64": G464,
    "float64_exact": G464,
    "g4x": G4X,
    "exact": G4X,
}


def resolve_tier(tier: G4Tier | str) -> G4Tier:
    if isinstance(tier, G4Tier):
        return tier
    try:
        return _TIER_ALIASES[str(tier).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unknown G4 tier {tier!r}") from exc


def current_promotion_policy() -> PromotionPolicy:
    return _DEFAULT_POLICY.get()


@contextmanager
def strict_exact() -> Iterator[None]:
    """Temporarily force strict range behavior."""

    token = _DEFAULT_POLICY.set(PromotionPolicy.STRICT)
    try:
        yield
    finally:
        _DEFAULT_POLICY.reset(token)


@contextmanager
def promote_exact() -> Iterator[None]:
    """Advanced opt-in: promote results to the smallest sufficient G4 tier."""

    token = _DEFAULT_POLICY.set(PromotionPolicy.PROMOTE)
    try:
        yield
    finally:
        _DEFAULT_POLICY.reset(token)


def _tier_rank(tier: G4Tier) -> int:
    if tier is G416:
        return 0
    if tier is G432:
        return 1
    if tier is G464:
        return 2
    return 3


def _widest_tier(left: G4Tier, right: G4Tier) -> G4Tier:
    return left if _tier_rank(left) >= _tier_rank(right) else right


def _result_tier(value: Fraction, declared: G4Tier, policy: PromotionPolicy) -> G4Tier:
    if declared.contains(value):
        return declared
    if policy is PromotionPolicy.STRICT:
        declared.require_contains(value, label="result")
        raise AssertionError("unreachable")
    start = _tier_rank(declared)
    for tier in _BOUNDED_TIERS[start + 1 :]:
        if tier.contains(value):
            return tier
    return G4X


@dataclass(frozen=True, slots=True)
class G4Scalar:
    value: Fraction
    tier: G4Tier = G432

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", as_fraction(self.value))
        object.__setattr__(self, "tier", resolve_tier(self.tier))
        self.tier.require_contains(self.value)

    def _coerce(self, other: Any) -> "G4Scalar":
        if isinstance(other, G4Scalar):
            return other
        return G4Scalar(as_fraction(other), self.tier)

    def _binary(self, other: Any, operation: Callable[[Fraction, Fraction], Fraction]) -> "G4Scalar":
        right = self._coerce(other)
        declared = _widest_tier(self.tier, right.tier)
        value = operation(self.value, right.value)
        tier = _result_tier(value, declared, current_promotion_policy())
        return G4Scalar(value, tier)

    def to(self, tier: G4Tier | str) -> "G4Scalar":
        return G4Scalar(self.value, resolve_tier(tier))

    def __add__(self, other: Any) -> "G4Scalar":
        return self._binary(other, operator.add)

    def __radd__(self, other: Any) -> "G4Scalar":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "G4Scalar":
        return self._binary(other, operator.sub)

    def __rsub__(self, other: Any) -> "G4Scalar":
        return self._coerce(other)._binary(self, operator.sub)

    def __mul__(self, other: Any) -> "G4Scalar":
        return self._binary(other, operator.mul)

    def __rmul__(self, other: Any) -> "G4Scalar":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "G4Scalar":
        right = self._coerce(other)
        if right.value == 0:
            raise ZeroDivisionError("exact division by zero")
        return self._binary(right, operator.truediv)

    def __rtruediv__(self, other: Any) -> "G4Scalar":
        if self.value == 0:
            raise ZeroDivisionError("exact division by zero")
        return self._coerce(other)._binary(self, operator.truediv)

    def __neg__(self) -> "G4Scalar":
        return G4Scalar(-self.value, self.tier)

    def __abs__(self) -> "G4Scalar":
        return G4Scalar(abs(self.value), self.tier)

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        if self.value.denominator != 1:
            raise TypeError("cannot convert a non-integer exact value to int")
        return self.value.numerator

    def __repr__(self) -> str:
        return f"{self.tier.name}({self.value})"


class G4Array:
    """Correctness-first exact array carrying one stable public G4 tier."""

    __array_priority__ = 1000

    def __init__(self, values: Any, tier: G4Tier | str = G432):
        self.tier = resolve_tier(tier)
        raw = np.asarray(values, dtype=object)
        converter = np.frompyfunc(as_fraction, 1, 1)
        exact_values = converter(raw)
        self._values = np.asarray(exact_values, dtype=object)
        self._require_range(self._values)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    @property
    def ndim(self) -> int:
        return self._values.ndim

    @property
    def size(self) -> int:
        return self._values.size

    def _require_range(self, values: np.ndarray[Any, np.dtype[object]]) -> None:
        for index, value in np.ndenumerate(values):
            self.tier.require_contains(value, label=f"value{index}")

    def _coerce(self, other: Any) -> "G4Array":
        if isinstance(other, G4Array):
            return other
        return G4Array(other, self.tier)

    @staticmethod
    def _select_array_tier(
        values: np.ndarray[Any, np.dtype[object]],
        declared: G4Tier,
    ) -> G4Tier:
        policy = current_promotion_policy()
        selected = declared
        for value in values.flat:
            selected = _result_tier(as_fraction(value), selected, policy)
        return selected

    def _binary(self, other: Any, operation: Callable[[Any, Any], Any]) -> "G4Array":
        right = self._coerce(other)
        declared = _widest_tier(self.tier, right.tier)
        values = operation(self._values, right._values)
        values = np.asarray(values, dtype=object)
        tier = self._select_array_tier(values, declared)
        return G4Array(values, tier)

    def __add__(self, other: Any) -> "G4Array":
        return self._binary(other, operator.add)

    def __radd__(self, other: Any) -> "G4Array":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "G4Array":
        return self._binary(other, operator.sub)

    def __rsub__(self, other: Any) -> "G4Array":
        return self._coerce(other)._binary(self, operator.sub)

    def __mul__(self, other: Any) -> "G4Array":
        return self._binary(other, operator.mul)

    def __rmul__(self, other: Any) -> "G4Array":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "G4Array":
        right = self._coerce(other)
        if any(value == 0 for value in right._values.flat):
            raise ZeroDivisionError("exact division by zero")
        return self._binary(right, operator.truediv)

    def __matmul__(self, other: Any) -> "G4Array":
        right = self._coerce(other)
        declared = _widest_tier(self.tier, right.tier)
        values = np.matmul(self._values, right._values)
        values = np.asarray(values, dtype=object)
        tier = self._select_array_tier(values, declared)
        return G4Array(values, tier)

    def sum(self, axis: int | tuple[int, ...] | None = None) -> "G4Array":
        values = np.asarray(self._values.sum(axis=axis), dtype=object)
        tier = self._select_array_tier(values, self.tier)
        return G4Array(values, tier)

    def to(self, tier: G4Tier | str) -> "G4Array":
        return G4Array(self._values.copy(), resolve_tier(tier))

    def fractions(self, *, copy: bool = True) -> np.ndarray[Any, np.dtype[object]]:
        return self._values.copy() if copy else self._values

    def to_numpy(self, dtype: Any | None = None) -> np.ndarray[Any, Any]:
        target = dtype if dtype is not None else self.tier.numpy_float_dtype
        if target is None:
            return np.asarray(self._values, dtype=object)
        converter = np.frompyfunc(float, 1, 1)
        return np.asarray(converter(self._values), dtype=target)

    def __getitem__(self, key: Any) -> Any:
        value = self._values[key]
        if isinstance(value, np.ndarray):
            return G4Array(value, self.tier)
        return G4Scalar(as_fraction(value), self.tier)

    def __repr__(self) -> str:
        return f"G4Array(tier={self.tier.name}, shape={self.shape}, values={self._values!r})"


def scalar(value: Any, dtype: G4Tier | str = G432) -> G4Scalar:
    return G4Scalar(as_fraction(value), resolve_tier(dtype))


def tensor(values: Any, dtype: G4Tier | str = G432) -> G4Array:
    return G4Array(values, resolve_tier(dtype))


def tier_table() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for tier in (*_BOUNDED_TIERS, G4X):
        rows.append(
            {
                "name": tier.name,
                "bits": tier.bits,
                "strict_default": True,
                "min_positive": None if tier.min_positive is None else str(tier.min_positive),
                "max_finite": None if tier.max_finite is None else str(tier.max_finite),
                "float_opponent": None if tier.bits is None else f"FP{tier.bits}",
            }
        )
    return tuple(rows)


__all__ = [
    "ExactRangeError",
    "ExactValueError",
    "PromotionPolicy",
    "G4Tier",
    "G4Scalar",
    "G4Array",
    "G416",
    "G432",
    "G464",
    "G4X",
    "exact16",
    "exact32",
    "exact64",
    "exact",
    "as_fraction",
    "resolve_tier",
    "current_promotion_policy",
    "strict_exact",
    "promote_exact",
    "scalar",
    "tensor",
    "tier_table",
]
