"""
rns_engine — Exact integer arithmetic via Residue Number System (RNS).

v0.3.0
------
- All rail arrays are uint16.
- Includes fused multiply-add: fma(a, b, c) = a*b + c in one core call.
- Adds high-level session/cache utilities via EncodedArray, SessionCache, Session.

Dynamic range: [0, 68,174,282,752) = 127 × 8191 × 65536

Notes
-----
- Values outside [0, M) are reduced mod M during encode.
- Division requires the divisor to be invertible on every rail:
    b % 127  != 0
    b % 8191 != 0
    b is odd
- HAS_AVX2 indicates whether this extension was compiled with AVX2 enabled.
  It is a build-time property, not runtime CPU detection.

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
>>> out1 = rns.decode(*rns.mul(*ea, *eb))
>>> out2 = rns.decode(*rns.fma(*ea, *eb, *ec))
>>>
>>> s = rns.Session()
>>> out3 = s.one_shot_affine(a, multiplier=1_000_003, addend=7)
"""

from ._core import (
    HAS_AVX2,
    M,
    M0,
    M1,
    M2,
    add,
    decode,
    div_,
    encode,
    fma,
    mul,
    op,
    sub,
)
from .engine import EncodedArray, Session, SessionCache

__version__ = "0.3.0"

__all__ = [
    "encode",
    "decode",
    "op",
    "add",
    "sub",
    "mul",
    "div_",
    "fma",
    "M",
    "M0",
    "M1",
    "M2",
    "HAS_AVX2",
    "EncodedArray",
    "SessionCache",
    "Session",
    "info",
]


def info() -> None:
    """Print a summary of the engine configuration."""
    print(f"rns_engine v{__version__}")
    print(f"  Dynamic range : [0, {M:,})")
    print(f"  Moduli        : {M0} x {M1} x {M2}")
    print(f"  AVX2          : {'yes' if HAS_AVX2 else 'no'}")
    print("  Operations    : add  sub  mul  div_  fma")
    print("  Rail dtype    : uint16 (all three rails)")
    print("  High-level    : EncodedArray  SessionCache  Session")
