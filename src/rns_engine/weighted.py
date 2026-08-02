from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Any, Iterable, Sequence

import numpy as np

from ._core import M
from ._core import add as _add
from ._core import decode as _decode
from ._core import mul_u64 as _mul_u64
from .engine import EncodedArray
from .signed import SignedRangeCertificate, certify_signed_bound, decode_signed, encode_signed


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


def _integer_tuple(values: Iterable[Any], *, name: str) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of integers") from exc

    return tuple(
        _require_integer(value, name=f"{name}[{position}]")
        for position, value in enumerate(raw)
    )


def _bound_tuple(values: Iterable[Any], *, name: str) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of nonnegative integers") from exc

    return tuple(
        _require_nonnegative_integer(value, name=f"{name}[{position}]")
        for position, value in enumerate(raw)
    )


def certify_weighted_sum_bound(
    weights: Sequence[int] | Iterable[int],
    term_abs_bounds: Sequence[int] | Iterable[int],
    addend_abs_bound: int = 0,
) -> SignedRangeCertificate:
    """Certify ``sum(weight[t] * term[t]) + addend`` from magnitude bounds.

    The conservative exact bound is::

        sum(abs(weight[t]) * term_abs_bound[t]) + addend_abs_bound

    Weights are unrestricted Python integers. The certificate is unique only
    when the resulting bound is strictly less than ``M/2``.
    """

    exact_weights = _integer_tuple(weights, name="weights")
    exact_bounds = _bound_tuple(term_abs_bounds, name="term_abs_bounds")
    if len(exact_weights) != len(exact_bounds):
        raise ValueError(
            "weights and term_abs_bounds must have the same length "
            f"({len(exact_weights)} != {len(exact_bounds)})"
        )

    addend = _require_nonnegative_integer(
        addend_abs_bound,
        name="addend_abs_bound",
    )
    total = addend + sum(
        abs(weight) * bound
        for weight, bound in zip(exact_weights, exact_bounds)
    )
    return certify_signed_bound(total)


@dataclass(frozen=True, slots=True)
class WeightedInt32Result:
    """RNS receipt for a weighted sum of signed INT32 partial outputs."""

    encoded: EncodedArray
    certificate: SignedRangeCertificate
    output_shape: tuple[int, ...]
    weights: tuple[int, ...]
    term_abs_bounds: tuple[int, ...]

    @property
    def term_count(self) -> int:
        return len(self.weights)

    @property
    def output_size(self) -> int:
        return int(np.prod(self.output_shape, dtype=np.int64)) if self.output_shape else 1

    def decode_modular(self) -> np.ndarray:
        """Return the canonical unsigned residue in ``[0, M)``."""

        return _decode(*self.encoded.rails()).reshape(self.output_shape)

    def decode_signed(self, *, require_unique: bool = True) -> np.ndarray:
        """Return the centered result, optionally requiring uniqueness proof."""

        if require_unique:
            self.certificate.require_unique()
        return decode_signed(*self.encoded.rails()).reshape(self.output_shape)


def accumulate_weighted_int32(
    partials: Any,
    weights: Sequence[int] | Iterable[int],
    *,
    require_unique: bool = False,
) -> WeightedInt32Result:
    """Combine signed INT32 partial outputs through native RNS kernels.

    ``partials`` has shape ``(terms, *output_shape)``. Each term is encoded by
    the native four-rail encoder, multiplied by its exact positional weight
    modulo ``M`` through the native scalar-broadcast kernel, and accumulated by
    the native rail-add kernel. Python orchestrates the terms; the arithmetic on
    every output element remains in the compiled RNS body.

    The returned range receipt uses the actual maximum absolute INT32 value in
    every term and the full Python integer weights, not their reduced residues.
    Therefore ``certificate.require_unique()`` is a valid no-wrap gate for the
    represented weighted sum.
    """

    raw = np.asarray(partials)
    if raw.ndim < 1:
        raise ValueError("partials must have shape (terms, *output_shape)")
    if raw.dtype != np.dtype(np.int32):
        raise TypeError("partials must have dtype int32")

    exact_weights = _integer_tuple(weights, name="weights")
    term_count = int(raw.shape[0])
    if len(exact_weights) != term_count:
        raise ValueError(
            "weights length must match partial term count "
            f"({len(exact_weights)} != {term_count})"
        )

    contiguous = np.ascontiguousarray(raw)
    output_shape = tuple(int(dimension) for dimension in contiguous.shape[1:])
    output_size = int(np.prod(output_shape, dtype=np.int64)) if output_shape else 1
    flat = contiguous.reshape(term_count, output_size)

    term_abs_bounds: list[int] = []
    for term in flat:
        if term.size == 0:
            term_abs_bounds.append(0)
            continue
        widened = term.astype(np.int64, copy=False)
        term_abs_bounds.append(int(np.max(np.abs(widened))))

    bounds = tuple(term_abs_bounds)
    certificate = certify_weighted_sum_bound(exact_weights, bounds)
    if require_unique:
        certificate.require_unique()

    accumulator = encode_signed(np.zeros(output_size, dtype=np.int64))
    modulus = int(M)

    for term, weight in zip(flat, exact_weights):
        if weight == 0 or output_size == 0:
            continue
        encoded = encode_signed(term.astype(np.int64, copy=False))
        scaled = _mul_u64(*encoded, weight % modulus)
        accumulator = _add(*accumulator, *scaled)

    return WeightedInt32Result(
        encoded=EncodedArray(*accumulator),
        certificate=certificate,
        output_shape=output_shape,
        weights=exact_weights,
        term_abs_bounds=bounds,
    )


__all__ = [
    "WeightedInt32Result",
    "accumulate_weighted_int32",
    "certify_weighted_sum_bound",
]
