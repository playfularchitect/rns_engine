from __future__ import annotations

from dataclasses import dataclass
from math import gcd, prod
from operator import index
from typing import Any, Iterable, Sequence

import numpy as np

from ._core import M, M0, M1, M2, M3


BASE_MODULI: tuple[int, ...] = (int(M0), int(M1), int(M2), int(M3))
SMALLEST_PRODUCT_EXTRA_EXPONENTS: tuple[int, ...] = (11, 29, 31)
BALANCED_EXTRA_EXPONENTS: tuple[int, ...] = (23, 24, 25)


def _require_int(value: Any, *, name: str) -> int:
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _normalize_moduli(moduli: Iterable[Any]) -> tuple[int, ...]:
    try:
        raw = tuple(moduli)
    except TypeError as exc:
        raise TypeError("moduli must be an iterable of integers") from exc
    if not raw:
        raise ValueError("moduli must contain at least one modulus")

    normalized: list[int] = []
    running = 1
    for position, value in enumerate(raw):
        modulus = _require_int(value, name=f"moduli[{position}]")
        if modulus <= 1:
            raise ValueError(f"moduli[{position}] must be > 1")
        common = gcd(running, modulus)
        if common != 1:
            raise ValueError(
                f"moduli[{position}]={modulus} is not coprime with the "
                f"running product; gcd={common}"
            )
        normalized.append(modulus)
        running *= modulus
    return tuple(normalized)


def mersenne_modulus(exponent: int) -> int:
    exponent = _require_int(exponent, name="exponent")
    if exponent < 2:
        raise ValueError("exponent must be >= 2")
    return (1 << exponent) - 1


def moduli_from_mersenne_exponents(
    exponents: Iterable[int],
    *,
    include_base: bool = True,
) -> tuple[int, ...]:
    extras = tuple(mersenne_modulus(exponent) for exponent in exponents)
    return _normalize_moduli((*BASE_MODULI, *extras) if include_base else extras)


def _rail_dtype(modulus: int) -> np.dtype[Any]:
    if modulus <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if modulus <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    if modulus <= np.iinfo(np.uint64).max:
        return np.dtype(np.uint64)
    return np.dtype(object)


def _object_array(values: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=object)
    except Exception as exc:  # pragma: no cover - NumPy supplies the details.
        raise TypeError(f"{name} must be array-like") from exc

    for position, value in enumerate(array.flat):
        try:
            index(value)
        except TypeError as exc:
            raise TypeError(f"{name}.flat[{position}] must be an integer") from exc
    return array


def _residue_array(values: np.ndarray, modulus: int) -> np.ndarray:
    flat = [int(value) % modulus for value in values.flat]
    return np.asarray(flat, dtype=_rail_dtype(modulus)).reshape(values.shape)


@dataclass(frozen=True, slots=True)
class WideRNSConfig:
    """Reference CRT configuration with an arbitrary pairwise-coprime rail set.

    This body is intentionally correctness-first. It is the CPU oracle for
    future CUDA rails; it does not claim that Python/NumPy reconstruction is a
    production-speed implementation.
    """

    moduli: tuple[int, ...]
    name: str = "custom"

    def __post_init__(self) -> None:
        object.__setattr__(self, "moduli", _normalize_moduli(self.moduli))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

    @classmethod
    def current(cls) -> WideRNSConfig:
        return cls(BASE_MODULI, "current-four-rail")

    @classmethod
    def smallest_product_seven_rail(cls) -> WideRNSConfig:
        return cls(
            moduli_from_mersenne_exponents(SMALLEST_PRODUCT_EXTRA_EXPONENTS),
            "seven-rail-smallest-product",
        )

    @classmethod
    def balanced_seven_rail(cls) -> WideRNSConfig:
        return cls(
            moduli_from_mersenne_exponents(BALANCED_EXTRA_EXPONENTS),
            "seven-rail-balanced",
        )

    @property
    def product(self) -> int:
        return prod(self.moduli, start=1)

    @property
    def product_bits(self) -> int:
        return self.product.bit_length()

    @property
    def rail_count(self) -> int:
        return len(self.moduli)

    @property
    def extra_moduli(self) -> tuple[int, ...]:
        if self.moduli[: len(BASE_MODULI)] == BASE_MODULI:
            return self.moduli[len(BASE_MODULI) :]
        return self.moduli

    @property
    def signed_min(self) -> int:
        return -(self.product // 2)

    @property
    def signed_max(self) -> int:
        return (self.product - 1) // 2

    def uniquely_represents_bound(self, max_abs_bound: int) -> bool:
        bound = _require_int(max_abs_bound, name="max_abs_bound")
        if bound < 0:
            raise ValueError("max_abs_bound must be >= 0")
        return 2 * bound < self.product

    def require_unique_bound(self, max_abs_bound: int) -> WideRNSConfig:
        if not self.uniquely_represents_bound(max_abs_bound):
            raise OverflowError(
                f"{self.name} is modular-only for absolute bound "
                f"{max_abs_bound}: product={self.product}"
            )
        return self

    def encode(self, values: Any) -> tuple[np.ndarray, ...]:
        integers = _object_array(values, name="values")
        return tuple(_residue_array(integers, modulus) for modulus in self.moduli)

    def decode(self, rails: Sequence[np.ndarray], *, signed: bool = False) -> np.ndarray:
        if len(rails) != self.rail_count:
            raise ValueError(
                f"expected {self.rail_count} rails, received {len(rails)}"
            )
        arrays = tuple(np.asarray(rail) for rail in rails)
        shape = arrays[0].shape
        if any(array.shape != shape for array in arrays):
            raise ValueError("all rails must have identical shapes")

        flat_rails = [array.reshape(-1) for array in arrays]
        output: list[int] = []
        for offset in range(flat_rails[0].size):
            x = int(flat_rails[0][offset]) % self.moduli[0]
            running = self.moduli[0]
            for rail, modulus in zip(flat_rails[1:], self.moduli[1:]):
                residue = int(rail[offset]) % modulus
                correction = ((residue - x) * pow(running, -1, modulus)) % modulus
                x += running * correction
                running *= modulus
            if signed and x > self.signed_max:
                x -= self.product
            output.append(x)
        return np.asarray(output, dtype=object).reshape(shape)

    def decode_scalar(self, residues: Sequence[int], *, signed: bool = False) -> int:
        arrays = tuple(np.asarray(value) for value in residues)
        return int(self.decode(arrays, signed=signed).item())

    def add(
        self,
        left: Sequence[np.ndarray],
        right: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, ...]:
        return _binary_rail_op(self, left, right, operation="add")

    def mul(
        self,
        left: Sequence[np.ndarray],
        right: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, ...]:
        return _binary_rail_op(self, left, right, operation="mul")


def _binary_rail_op(
    config: WideRNSConfig,
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
    *,
    operation: str,
) -> tuple[np.ndarray, ...]:
    if len(left) != config.rail_count or len(right) != config.rail_count:
        raise ValueError("left and right must match the configuration rail count")

    result: list[np.ndarray] = []
    for position, (lhs, rhs, modulus) in enumerate(
        zip(left, right, config.moduli)
    ):
        lhs_array = np.asarray(lhs)
        rhs_array = np.asarray(rhs)
        if lhs_array.shape != rhs_array.shape:
            raise ValueError(f"rail {position} shapes do not match")
        lhs_object = lhs_array.astype(object, copy=False)
        rhs_object = rhs_array.astype(object, copy=False)
        if operation == "add":
            raw = (lhs_object + rhs_object) % modulus
        elif operation == "mul":
            raw = (lhs_object * rhs_object) % modulus
        else:  # pragma: no cover - internal-only dispatch.
            raise AssertionError(operation)
        result.append(np.asarray(raw, dtype=_rail_dtype(modulus)))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class WideWeightedResult:
    config: WideRNSConfig
    rails: tuple[np.ndarray, ...]
    weights: tuple[int, ...]
    term_abs_bounds: tuple[int, ...]
    output_shape: tuple[int, ...]

    @property
    def max_abs_bound(self) -> int:
        return sum(
            abs(weight) * bound
            for weight, bound in zip(self.weights, self.term_abs_bounds)
        )

    @property
    def unique(self) -> bool:
        return self.config.uniquely_represents_bound(self.max_abs_bound)

    def require_unique(self) -> WideWeightedResult:
        self.config.require_unique_bound(self.max_abs_bound)
        return self

    def decode_modular(self) -> np.ndarray:
        return self.config.decode(self.rails, signed=False)

    def decode_signed(self, *, require_unique: bool = True) -> np.ndarray:
        if require_unique:
            self.require_unique()
        return self.config.decode(self.rails, signed=True)


def accumulate_weighted_int32_wide(
    partials: np.ndarray,
    weights: Iterable[int],
    config: WideRNSConfig,
    *,
    require_unique: bool = False,
) -> WideWeightedResult:
    """Reference weighted accumulation for CUDA-style signed INT32 partials."""

    values = np.asarray(partials)
    if values.dtype != np.int32:
        raise TypeError("partials dtype must be exactly int32")
    if values.ndim < 1:
        raise ValueError("partials must have a leading term dimension")

    exact_weights = tuple(
        _require_int(value, name=f"weights[{position}]")
        for position, value in enumerate(tuple(weights))
    )
    if len(exact_weights) != values.shape[0]:
        raise ValueError(
            "weights length must equal partial term count "
            f"({len(exact_weights)} != {values.shape[0]})"
        )

    output_shape = tuple(values.shape[1:])
    term_abs_bounds: list[int] = []
    for term in values:
        if term.size == 0:
            term_abs_bounds.append(0)
        else:
            minimum = int(term.min())
            maximum = int(term.max())
            term_abs_bounds.append(max(abs(minimum), abs(maximum)))

    rails: list[np.ndarray] = []
    for modulus in config.moduli:
        accumulator = np.zeros(output_shape, dtype=object)
        for term, weight in zip(values, exact_weights):
            if weight == 0:
                continue
            accumulator = (
                accumulator
                + term.astype(object, copy=False) * (weight % modulus)
            ) % modulus
        rails.append(np.asarray(accumulator, dtype=_rail_dtype(modulus)))

    result = WideWeightedResult(
        config=config,
        rails=tuple(rails),
        weights=exact_weights,
        term_abs_bounds=tuple(term_abs_bounds),
        output_shape=output_shape,
    )
    if require_unique:
        result.require_unique()
    return result


assert prod(BASE_MODULI, start=1) == int(M)
