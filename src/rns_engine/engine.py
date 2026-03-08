from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from ._core import encode as _encode
from ._core import decode as _decode
from ._core import add as _add
from ._core import sub as _sub
from ._core import mul as _mul
from ._core import div_ as _div
from ._core import fma as _fma
from ._core import M


@dataclass(frozen=True)
class EncodedArray:
    """Encoded residue payload.

    This is a lightweight wrapper around the three residue rails returned by
    the C++ core. It makes chained exact arithmetic easier to read and gives
    Session a stable object to cache.
    """

    r0: np.ndarray
    r1: np.ndarray
    r2: np.ndarray

    def rails(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.r0, self.r1, self.r2

    @property
    def size(self) -> int:
        return int(len(self.r0))


class SessionCache:
    """Small LRU cache for encoded arrays.

    Keys are byte-stable fingerprints of uint64 input arrays. The cache is
    intentionally simple and Python-only: the high-value part is skipping
    repeated encode work for reused inputs/constants while keeping the C++ core
    untouched.
    """

    def __init__(self, capacity: int = 32):
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = int(capacity)
        self._store: OrderedDict[tuple[Any, ...], EncodedArray] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def clear(self) -> None:
        self._store.clear()
        self.hits = self.misses = self.evictions = 0

    def info(self) -> dict[str, int]:
        return {
            "capacity": self.capacity,
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }

    def _touch(self, key: tuple[Any, ...]) -> EncodedArray:
        value = self._store.pop(key)
        self._store[key] = value
        self.hits += 1
        return value

    def get(self, key: tuple[Any, ...]) -> EncodedArray | None:
        if key in self._store:
            return self._touch(key)
        self.misses += 1
        return None

    def put(self, key: tuple[Any, ...], value: EncodedArray) -> EncodedArray:
        if self.capacity == 0:
            return value
        if key in self._store:
            self._store.pop(key)
        elif len(self._store) >= self.capacity:
            self._store.popitem(last=False)
            self.evictions += 1
        self._store[key] = value
        return value


def _as_uint64_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.uint64)
    if arr.ndim != 1:
        raise ValueError("expected a 1D array-like of uint64 values")
    return np.ascontiguousarray(arr)


def _cache_key(arr: np.ndarray, tag: str | None = None) -> tuple[Any, ...]:
    arr = _as_uint64_array(arr)
    return (
        tag or "raw",
        arr.dtype.str,
        arr.shape,
        memoryview(arr).tobytes(),
    )


class Session:
    """High-level exact arithmetic session.

    Adds three practical things on top of the v0.2 core:
    - cached encoding for repeated arrays/constants
    - object-style chaining via EncodedArray
    - workload-oriented helpers: service / one_shot / hot_loop
    """

    def __init__(self, cache_capacity: int = 32):
        self.cache = SessionCache(cache_capacity)

    def clear_cache(self) -> None:
        self.cache.clear()

    def cache_info(self) -> Mapping[str, int]:
        return self.cache.info()

    def encode(self, x: Any, *, use_cache: bool = True, tag: str | None = None) -> EncodedArray:
        arr = _as_uint64_array(x)
        key = _cache_key(arr, tag) if use_cache else None
        if key is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        payload = EncodedArray(*_encode(arr))
        if key is not None:
            self.cache.put(key, payload)
        return payload

    def decode(self, x: EncodedArray) -> np.ndarray:
        return _decode(*x.rails())

    def add(self, a: EncodedArray, b: EncodedArray) -> EncodedArray:
        return EncodedArray(*_add(*a.rails(), *b.rails()))

    def sub(self, a: EncodedArray, b: EncodedArray) -> EncodedArray:
        return EncodedArray(*_sub(*a.rails(), *b.rails()))

    def mul(self, a: EncodedArray, b: EncodedArray) -> EncodedArray:
        return EncodedArray(*_mul(*a.rails(), *b.rails()))

    def div(self, a: EncodedArray, b: EncodedArray) -> EncodedArray:
        return EncodedArray(*_div(*a.rails(), *b.rails()))

    def fma(self, a: EncodedArray, b: EncodedArray, c: EncodedArray) -> EncodedArray:
        return EncodedArray(*_fma(*a.rails(), *b.rails(), *c.rails()))

    def service(self, x: Any, *, selected: str = "identity") -> np.ndarray:
        """Conservative default mode.

        Today this simply means: encode once (with cache), return decoded result.
        The point is API stability: future internal policy changes can happen
        behind this name without breaking callers.
        """
        enc = self.encode(x, use_cache=True, tag=f"service:{selected}")
        return self.decode(enc)

    def one_shot_affine(self, x: Any, *, multiplier: int, addend: int) -> np.ndarray:
        """Best for single raw calls: encode inputs, apply exact fma, decode."""
        x_arr = _as_uint64_array(x)
        m_arr = np.full(len(x_arr), np.uint64(multiplier % M), dtype=np.uint64)
        k_arr = np.full(len(x_arr), np.uint64(addend % M), dtype=np.uint64)
        ex = self.encode(x_arr, use_cache=False)
        em = self.encode(m_arr, use_cache=True, tag=("const", int(multiplier % M), len(x_arr)))
        ek = self.encode(k_arr, use_cache=True, tag=("const", int(addend % M), len(x_arr)))
        return self.decode(self.fma(ex, em, ek))

    def hot_loop_affine(self, x: Any, *, multiplier: int, addend: int, iterations: int) -> np.ndarray:
        """Best for repeated exact loops: stay in residue space, decode once."""
        if iterations < 0:
            raise ValueError("iterations must be >= 0")
        x_arr = _as_uint64_array(x)
        m_arr = np.full(len(x_arr), np.uint64(multiplier % M), dtype=np.uint64)
        k_arr = np.full(len(x_arr), np.uint64(addend % M), dtype=np.uint64)
        r = self.encode(x_arr, use_cache=True, tag=("loop-x", len(x_arr), int(x_arr[0]) if len(x_arr) else 0))
        em = self.encode(m_arr, use_cache=True, tag=("const", int(multiplier % M), len(x_arr)))
        ek = self.encode(k_arr, use_cache=True, tag=("const", int(addend % M), len(x_arr)))
        for _ in range(iterations):
            r = self.fma(r, em, ek)
        return self.decode(r)


__all__ = ["EncodedArray", "SessionCache", "Session"]
