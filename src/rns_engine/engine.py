from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._core import M
from ._core import add as _add
from ._core import affine_repeat_u64 as _affine_repeat_u64
from ._core import decode as _decode
from ._core import div_ as _div
from ._core import encode as _encode
from ._core import fma as _fma
from ._core import fma_u64 as _fma_u64
from ._core import mul as _mul
from ._core import mul_u64 as _mul_u64
from ._core import sub as _sub


def _as_uint64_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.uint64)
    if arr.ndim != 1:
        raise ValueError("expected a 1D array-like of uint64 values")
    return np.ascontiguousarray(arr)


def _as_readonly_u16_1d(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.uint16)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D uint16 array")
    arr = np.ascontiguousarray(arr)
    arr.setflags(write=False)
    return arr


def _as_readonly_u32_1d(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.uint32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D uint32 array")
    arr = np.ascontiguousarray(arr)
    arr.setflags(write=False)
    return arr


def _cache_key(arr: Any, tag: object | None = None) -> tuple[Any, ...]:
    arr = _as_uint64_array(arr)
    return (
        tag if tag is not None else "raw",
        arr.dtype.str,
        arr.shape,
        memoryview(arr).tobytes(),
    )


def _wrap_encoded(
    rails: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> "EncodedArray":
    return EncodedArray(*rails)


@dataclass(frozen=True, slots=True)
class EncodedArray:
    """Read-only encoded 4-rail residue payload."""

    r0: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    r3: np.ndarray

    def __post_init__(self) -> None:
        r0 = _as_readonly_u16_1d(self.r0, name="r0")
        r1 = _as_readonly_u16_1d(self.r1, name="r1")
        r2 = _as_readonly_u16_1d(self.r2, name="r2")
        r3 = _as_readonly_u32_1d(self.r3, name="r3")

        n = len(r0)
        if len(r1) != n or len(r2) != n or len(r3) != n:
            raise ValueError("rail length mismatch")

        object.__setattr__(self, "r0", r0)
        object.__setattr__(self, "r1", r1)
        object.__setattr__(self, "r2", r2)
        object.__setattr__(self, "r3", r3)

    def rails(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.r0, self.r1, self.r2, self.r3

    @property
    def size(self) -> int:
        return int(self.r0.shape[0])


class SessionCache:
    """Small LRU cache for encoded arrays."""

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
        self.hits = 0
        self.misses = 0
        self.evictions = 0

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


class Session:
    """High-level exact arithmetic session."""

    def __init__(self, cache_capacity: int = 32):
        self.cache = SessionCache(cache_capacity)

    def clear_cache(self) -> None:
        self.cache.clear()

    def cache_info(self) -> Mapping[str, int]:
        return self.cache.info()

    def encode(
        self,
        x: Any,
        *,
        use_cache: bool = True,
        tag: object | None = None,
    ) -> EncodedArray:
        arr = _as_uint64_array(x)
        key = _cache_key(arr, tag) if use_cache else None

        if key is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        payload = _wrap_encoded(_encode(arr))
        if key is not None:
            return self.cache.put(key, payload)
        return payload

    def decode(self, x: EncodedArray) -> np.ndarray:
        return _decode(*x.rails())

    def add(self, a: EncodedArray, b: EncodedArray) -> EncodedArray:
        return _wrap_encoded(_add(*a.rails(), *b.rails()))

    def sub(self, a: EncodedArray, b: EncodedArray) -> EncodedArray:
        return _wrap_encoded(_sub(*a.rails(), *b.rails()))

    def mul(self, a: EncodedArray, b: EncodedArray) -> EncodedArray:
        return _wrap_encoded(_mul(*a.rails(), *b.rails()))

    def div(self, a: EncodedArray, b: EncodedArray) -> EncodedArray:
        return _wrap_encoded(_div(*a.rails(), *b.rails()))

    def fma(self, a: EncodedArray, b: EncodedArray, c: EncodedArray) -> EncodedArray:
        return _wrap_encoded(_fma(*a.rails(), *b.rails(), *c.rails()))

    def mul_u64(self, a: EncodedArray, multiplier: int) -> EncodedArray:
        mm = int(multiplier % M)
        return _wrap_encoded(_mul_u64(*a.rails(), mm))

    def fma_u64(self, a: EncodedArray, multiplier: int, addend: int) -> EncodedArray:
        mm = int(multiplier % M)
        kk = int(addend % M)
        return _wrap_encoded(_fma_u64(*a.rails(), mm, kk))

    def affine_repeat_u64(
        self,
        a: EncodedArray,
        *,
        multiplier: int,
        addend: int,
        iterations: int,
    ) -> EncodedArray:
        if iterations < 0:
            raise ValueError("iterations must be >= 0")
        mm = int(multiplier % M)
        kk = int(addend % M)
        return _wrap_encoded(_affine_repeat_u64(*a.rails(), mm, kk, int(iterations)))

    def service(self, x: Any, *, selected: str = "identity") -> np.ndarray:
        enc = self.encode(x, use_cache=True, tag=f"service:{selected}")
        return self.decode(enc)

    def one_shot_affine(self, x: Any, *, multiplier: int, addend: int) -> np.ndarray:
        x_arr = _as_uint64_array(x)
        ex = self.encode(x_arr, use_cache=False)
        return self.decode(self.fma_u64(ex, multiplier, addend))

    def hot_loop_affine(
        self,
        x: Any,
        *,
        multiplier: int,
        addend: int,
        iterations: int,
    ) -> np.ndarray:
        if iterations < 0:
            raise ValueError("iterations must be >= 0")

        x_arr = _as_uint64_array(x)
        x_tag = ("loop-x", len(x_arr), int(x_arr[0]) if len(x_arr) else 0)
        r = self.encode(x_arr, use_cache=True, tag=x_tag)
        r = self.affine_repeat_u64(
            r,
            multiplier=multiplier,
            addend=addend,
            iterations=iterations,
        )
        return self.decode(r)


__all__ = ["EncodedArray", "SessionCache", "Session"]
