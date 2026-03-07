# rns_engine

Alpha prototype for **batched exact modular integer arithmetic** using a fixed 3-rail Residue Number System (RNS).

Current implementation:
- fixed moduli: `127`, `8191`, `65536`
- dynamic range: `[0, 68,174,282,752)`
- operations on batches of values encoded into residue rails
- AVX2 fast path on supported x86-64 CPUs
- scalar fallback on unsupported hardware

This package is a **prototype**, not a finished arbitrary-precision integer engine.

## What it currently does

`rns_engine` encodes `uint64` values into three residue arrays and performs arithmetic independently on each rail. Results can then be reconstructed with CRT / Garner decoding.

Supported operations:
- `encode`
- `decode`
- `add`
- `sub`
- `mul`
- `div_`
- `fma`

These operations are exact **modulo**:

```text
M = 127 × 8191 × 65536 = 68,174,282,752

Values outside [0, M) are reduced modulo M on encode.

What it does NOT currently do

This version does not provide:

arbitrary precision integers

configurable rail sets

automatic precision growth

fault-detecting CRT reconstruction

a full benchmark / proof suite inside the repo

hardened production-grade API validation

Important correctness scope

Arithmetic is exact within the represented modulus range of the current 3-rail system.

This means:

add/sub/mul/fma are exact in residue space

decoded results are exact modulo M

if you want ordinary integer results rather than modular wraparound, your true result must still lie within the represented range

Division

div_ is only valid when the divisor is invertible on every rail.

For the current moduli, that means:

b % 127 != 0

b % 8191 != 0

b must be odd (so it is invertible modulo 65536)

If those conditions are not met, division is not mathematically valid in this representation.

Quick example
import numpy as np
import rns_engine as rns

a = np.array([123456789, 999999999], dtype=np.uint64)
b = np.array([987654321, 111111111], dtype=np.uint64)

ea = rns.encode(a)
eb = rns.encode(b)

out = rns.decode(*rns.mul(*ea, *eb))
print(out)
Current status

This repository currently contains a small fixed-precision prototype intended as a stepping stone toward a future N-rail configurable engine.

The long-term direction is:

configurable rail sets

larger precision contexts

stronger tests and benchmarks

native baselines

clearer guarantees

Build
pip install -e .
pytest tests/ -v
License

MIT
