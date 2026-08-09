"""Public exact G4 Series 1 matrix multiplication on certified Tesla T4 shapes."""
from __future__ import annotations

import ctypes
from operator import index
from typing import Any

import numpy as np

from .exact_gemm import SharedScaleMatrix
from .g4_runtime import ensure_t4, matmul_library, shape_map

_INT8_MIN = -128
_INT8_MAX = 127
_NATIVE_CACHE: dict[str, ctypes.CDLL] = {}


def _integer_matrix(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix")

    if np.issubdtype(array.dtype, np.integer):
        if array.size:
            lo = int(np.min(array)); hi = int(np.max(array))
            if lo < _INT8_MIN or hi > _INT8_MAX:
                raise OverflowError(
                    f"{name} is outside the certified G4 Series 1 fast-input range: "
                    f"signed INT8 [{_INT8_MIN}, {_INT8_MAX}] (observed [{lo}, {hi}])"
                )
        return np.ascontiguousarray(array, dtype=np.int8)

    if array.dtype != object:
        raise TypeError(f"{name} must contain integers or use SharedScaleMatrix for exact rationals")

    checked = np.empty(array.shape, dtype=np.int8)
    for position, value in np.ndenumerate(array):
        try:
            integer = index(value)
        except TypeError as exc:
            raise TypeError(f"{name}{position} is not an integer") from exc
        if integer < _INT8_MIN or integer > _INT8_MAX:
            raise OverflowError(
                f"{name}{position}={integer} is outside the certified G4 Series 1 "
                f"fast-input range [{_INT8_MIN}, {_INT8_MAX}]"
            )
        checked[position] = integer
    return np.ascontiguousarray(checked)


def _operand(value: Any, *, name: str) -> tuple[np.ndarray, int, bool]:
    if isinstance(value, SharedScaleMatrix):
        return _integer_matrix(value.numerators, name=f"{name}.numerators"), int(value.scale), True
    return _integer_matrix(value, name=name), 1, False


def _native_library(family: str) -> ctypes.CDLL:
    cached = _NATIVE_CACHE.get(family)
    if cached is not None:
        return cached
    ensure_t4()
    path = matmul_library(family)
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:
        raise RuntimeError(
            "G4 Series 1 user math currently requires Linux with the NVIDIA Tesla T4 CUDA runtime available"
        ) from exc
    fn = lib.rns_g4s1_matmul
    fn.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    fn.restype = ctypes.c_int
    _NATIVE_CACHE[family] = lib
    return lib


def _run_native(left: np.ndarray, right: np.ndarray, family: str) -> np.ndarray:
    m, k = left.shape
    k2, n = right.shape
    if k != k2:
        raise ValueError(f"matrix inner dimensions must match ({k} != {k2})")
    output = np.empty((m, n), dtype=np.int32, order="C")
    lib = _native_library(family)
    err = ctypes.create_string_buffer(2048)
    rc = lib.rns_g4s1_matmul(
        m,
        n,
        k,
        left.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        right.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        err,
        len(err),
    )
    if rc:
        detail = err.value.decode("utf-8", errors="replace") or f"native error {rc}"
        raise RuntimeError(f"G4 Series 1 matmul failed: {detail}")
    return output


def g4_matmul(left: Any, right: Any, *, reduce: bool = False):
    """Multiply one certified G4 Series 1 matrix pair exactly on a Tesla T4.

    Integer inputs must fit signed INT8 and return an exact ``numpy.int32``
    matrix. Exact rational inputs use :class:`SharedScaleMatrix`; numerator
    matrices must fit signed INT8 and the result is another exact
    ``SharedScaleMatrix``.

    The fast runtime covers the frozen 1,024 Series 1 ``(M,N,K)`` shapes.
    Unsupported shapes and unsupported numeric representations fail closed;
    G4 never silently converts the operation to floating point.

    FP32-class G4 Series 2 support is in development and is intended to extend
    this same API once physically certified.
    """
    lhs, left_scale, left_rational = _operand(left, name="left")
    rhs, right_scale, right_rational = _operand(right, name="right")
    if lhs.shape[1] != rhs.shape[0]:
        raise ValueError(f"matrix inner dimensions must match ({lhs.shape[1]} != {rhs.shape[0]})")

    key = (int(lhs.shape[0]), int(rhs.shape[1]), int(lhs.shape[1]))
    shape = shape_map().get(key)
    if shape is None:
        raise NotImplementedError(
            f"G4 Series 1 does not have a certified fast implementation for "
            f"M={key[0]}, N={key[1]}, K={key[2]}. The current catalog contains 1,024 specific shapes."
        )

    numerator = _run_native(lhs, rhs, str(shape["family"]))
    if not (left_rational or right_rational):
        return numerator

    result = SharedScaleMatrix(numerator, left_scale * right_scale)
    return result.reduced() if reduce else result


__all__ = ["g4_matmul"]
