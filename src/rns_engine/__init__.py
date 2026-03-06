"""
rns_engine — Exact integer arithmetic via Residue Number System (RNS).

Dynamic range: [0, 68,174,282,752)  =  127 × 8191 × 65536
AVX2-accelerated on x86; scalar fallback on all other platforms.

Quick start
-----------
>>> import rns_engine as rns
>>> import numpy as np
>>>
>>> a = np.array([123456789, 999999999], dtype=np.uint64)
>>> b = np.array([987654321, 111111111], dtype=np.uint64)
>>>
>>> ea = rns.encode(a)          # -> (r0, r1, r2) residue arrays
>>> eb = rns.encode(b)
>>>
>>> result = rns.decode(*rns.mul(*ea, *eb))   # exact multiplication
>>> # stays in residue space for multi-step expressions:
>>> s1 = rns.add(*ea, *eb)      # a + b
>>> s2 = rns.mul(*s1, *eb)      # (a + b) * b
>>> out = rns.decode(*s2)       # decode once at the end

Notes
-----
- All values must be in [0, M) where M = rns.M = 68,174,282,752
- Values outside this range are reduced mod M on encode
- Division requires b to be coprime to all moduli:
    b % 127  != 0
    b % 8191 != 0
    b % 65536 is odd  (coprime to 2^16)
"""

from ._core import (
    encode,
    decode,
    op,
    add,
    sub,
    mul,
    div_,
    M,
    M0,
    M1,
    M2,
    HAS_AVX2,
)

__version__ = "0.1.0"
__all__ = ["encode", "decode", "op", "add", "sub", "mul", "div_",
           "M", "M0", "M1", "M2", "HAS_AVX2"]


def info():
    """Print a summary of the engine configuration."""
    print(f"rns_engine v{__version__}")
    print(f"  Dynamic range : [0, {M:,})")
    print(f"  Moduli        : {M0} × {M1} × {M2}")
    print(f"  AVX2          : {'yes' if HAS_AVX2 else 'no (scalar fallback)'}")
    print(f"  Operations    : add  sub  mul  div_")
