from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Any

import numpy as np

from ._core import M
from ._core import decode as _decode
from ._core import encode as _encode

HALF_M = int(M) // 2
SIGNED_MIN = -HALF_M
SIGNED_MAX = HALF_M - 1
UNIQUE_SIGNED_MIN = -HALF_M + 1
UNIQUE_SIGNED_MAX = HALF_M - 1


def _as_signed_int64_array(x: Any) -> np.ndarray:
    raw = np.asarray(x)
    if raw.ndim != 1:
        raise ValueError("expected a 1D array-like of signed integer values")

    if np.issubdtype(raw.dtype, np.integer):
        if np.issubdtype(raw.dtype, np.unsignedinteger):
            if raw.size and np.any(raw > np.uint64(SIGNED_MAX)):
                raise OverflowError(
                    f"signed RNS input must be in [{SIGNED_MIN}, {SIGNED_MAX}]"
                )
        else:
            if raw.size and (
                np.any(raw < np.int64(SIGNED_MIN))
                or np.any(raw > np.int64(SIGNED_MAX))
            ):
                raise OverflowError(
                    f"signed RNS input must be in [{SIGNED_MIN}, {SIGNED_MAX}]"
                )
        return np.ascontiguousarray(raw, dtype=np.int64)

    if raw.dtype == object:
        values: list[int] = []
        for position, value in enumerate(raw):
            try:
                integer = index(value)
            except TypeError as exc:
                raise TypeError(
                    f"signed RNS input at index {position} is not an integer"
                ) from exc
            if integer < SIGNED_MIN or integer > SIGNED_MAX:
                raise OverflowError(
                    f"signed RNS input at index {position} must be in "
                    f"[{SIGNED_MIN}, {SIGNED_MAX}]"
                )
            values.append(integer)
        return np.ascontiguousarray(values, dtype=np.int64)

    raise TypeError("signed RNS input must contain integers")


def encode_signed(x: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode canonical centered integers without silent modular wrapping.

    Accepted values are in ``[-M/2, M/2 - 1]``. Negative values are mapped to
    their canonical residues before entering the native four-rail encoder.
    """

    signed = _as_signed_int64_array(x)
    residues = signed.copy()
    negative = residues < 0
    residues[negative] += int(M)
    return _encode(np.ascontiguousarray(residues, dtype=np.uint64))


def decode_signed(
    r0: Any,
    r1: Any,
    r2: Any,
    r3: Any,
) -> np.ndarray:
    """Decode residues into the canonical centered interval.

    The residue ``M/2`` is represented as ``-M/2``. This is a canonical
    interpretation, not proof that an unknown mathematical result did not wrap;
    use :func:`certify_signed_bound` for that guarantee.
    """

    residues = _decode(r0, r1, r2, r3)
    centered = np.ascontiguousarray(residues, dtype=np.int64)
    upper_half = residues >= np.uint64(HALF_M)
    centered[upper_half] -= int(M)
    return centered


@dataclass(frozen=True, slots=True)
class SignedRangeCertificate:
    """Certificate that a known absolute bound has one signed RNS decoding."""

    max_abs_bound: int

    def __post_init__(self) -> None:
        try:
            bound = index(self.max_abs_bound)
        except TypeError as exc:
            raise TypeError("max_abs_bound must be an integer") from exc
        if bound < 0:
            raise ValueError("max_abs_bound must be >= 0")
        object.__setattr__(self, "max_abs_bound", bound)

    @property
    def modulus(self) -> int:
        return int(M)

    @property
    def half_modulus(self) -> int:
        return HALF_M

    @property
    def unique(self) -> bool:
        return self.max_abs_bound < HALF_M

    @property
    def headroom(self) -> int:
        """Remaining strict signed headroom; negative means the bound exceeds it."""

        return UNIQUE_SIGNED_MAX - self.max_abs_bound

    @property
    def minimum_required_modulus(self) -> int:
        """Smallest modulus that uniquely supports ``[-bound, +bound]``."""

        return 2 * self.max_abs_bound + 1

    def require_unique(self) -> "SignedRangeCertificate":
        if not self.unique:
            raise OverflowError(
                "signed result is modular-only: the certified absolute bound "
                f"{self.max_abs_bound} must be less than M/2 ({HALF_M})"
            )
        return self


def certify_signed_bound(max_abs_bound: int) -> SignedRangeCertificate:
    """Return a strict uniqueness certificate for a known absolute bound."""

    return SignedRangeCertificate(max_abs_bound=max_abs_bound)


__all__ = [
    "HALF_M",
    "SIGNED_MIN",
    "SIGNED_MAX",
    "UNIQUE_SIGNED_MIN",
    "UNIQUE_SIGNED_MAX",
    "SignedRangeCertificate",
    "certify_signed_bound",
    "encode_signed",
    "decode_signed",
]
