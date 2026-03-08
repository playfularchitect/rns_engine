"""
rns_engine — Exact integer arithmetic via Residue Number System (RNS).

v0.2.0 — All rail arrays now uint16 (was uint32 for rail 1 in v0.1.x).
         New fma(a, b, c) operation: a*b+c in one kernel call.
         Faster: fma is ~4x faster than add(*mul(a,b), c) for tight loops.

Dynamic range: [0, 68,174,282,752)  =  127 × 8191 × 65536
AVX2-accelerated on x86; scalar fallback on all other platforms.

Quick start
-----------
>>> import rns_engine as rns
>>> import numpy as np
>>>
>>> a = np.array([123456789, 999999999], dtype=np.uint64)
>>> b = np.array([987654321, 111111111], dtype=np.uint64)
>>> c = np.array([7, 7], dtype=np.uint64)
>>>
>>> ea = rns.encode(a)
>>> eb = rns.encode(b)
>>> ec = rns.encode(c)
>>>
>>> result = rns.decode(*rns.mul(*ea, *eb))    # exact multiplication
>>> result = rns.decode(*rns.fma(*ea, *eb, *ec))  # a*b+c, one kernel call
>>>
>>> # Tight loop — stays in residue space, decode once at the end:
>>> r = rns.encode(a)
>>> m = rns.encode(np.full(len(a), 1_000_003, dtype=np.uint64))
>>> k = rns.encode(np.full(len(a), 7,         dtype=np.uint64))
>>> for _ in range(1000):
...     r = rns.fma(*r, *m, *k)   # r = r*m + k, exact, every iteration
>>> out = rns.decode(*r)

Notes
-----
- All values in [0, M) where M = rns.M = 68,174,282,752
- Values outside this range are reduced mod M on encode
- v0.2.0 API change: all rail arrays are now dtype=uint16
  (v0.1.x returned uint32 for rail 1 — update any code that checks dtypes)
- Division requires b coprime to all moduli:
    b % 127  != 0
    b % 8191 != 0
    b is odd  (coprime to 2^16)
"""

from ._core import (
    encode, decode, op,
    add, sub, mul, div_, fma,
    M, M0, M1, M2, HAS_AVX2,
)

__version__ = "0.3.0"
__all__ = ["encode", "decode", "op", "add", "sub", "mul", "div_", "fma",
           "M", "M0", "M1", "M2", "HAS_AVX2",
           "EncodedArray", "SessionCache", "Session"]


def info():
    """Print a summary of the engine configuration."""
    print(f"rns_engine v{__version__}")
    print(f"  Dynamic range : [0, {M:,})")
    print(f"  Moduli        : {M0} x {M1} x {M2}")
    print(f"  AVX2          : {'yes' if HAS_AVX2 else 'no (scalar fallback)'}")
    print(f"  Operations    : add  sub  mul  div_  fma")
    print(f"  Rail dtype    : uint16 (all three rails)")

from .engine import EncodedArray, SessionCache, Session
