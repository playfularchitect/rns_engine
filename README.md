# rns_engine

**Exact integer arithmetic via a 3-rail Residue Number System (RNS), with AVX2 acceleration.**

No floating point. No approximation. Exact results within the engine's dynamic range.

---

## What it does

Standard Python integers are exact but slow. NumPy is fast, but fixed-width integer arrays can overflow and float-based pipelines drift. `rns_engine` gives you **exact arithmetic modulo a fixed dynamic range** by decomposing integers into residues across three coprime moduli:

- `127`
- `8191`
- `65536`

Arithmetic is performed independently on each rail, then reconstructed with the Chinese Remainder Theorem (Garner form).

**Dynamic range:** `[0, 68,174,282,752)`

```
127 × 8191 × 65536 = 68,174,282,752
```

---

## Install

```bash
pip install rns_engine
```

### AVX2 note

`rns_engine` can be built with AVX2 acceleration on supported x86_64 systems. `HAS_AVX2` is a build-time property of the compiled extension, not runtime CPU detection. For most users, `pip install rns_engine` is the right starting point.

---

## Quick start

```python
import numpy as np
import rns_engine as rns

a = np.array([123456789, 999999999], dtype=np.uint64)
b = np.array([987654321, 111111111], dtype=np.uint64)

# Encode once
ea = rns.encode(a)
eb = rns.encode(b)

# Exact multiplication in residue space
out = rns.decode(*rns.mul(*ea, *eb))

# Chain multiple operations — decode once at the end
s1 = rns.add(*ea, *eb)   # a + b
s2 = rns.mul(*s1, *eb)   # (a + b) * b
s3 = rns.sub(*s2, *ea)   # (a + b) * b - a
out = rns.decode(*s3)
```

---

## Session quick start

```python
import numpy as np
import rns_engine as rns

s = rns.Session(cache_capacity=32)
x = np.array([1, 2, 3, 4], dtype=np.uint64)

# Cache-aware encode
ex = s.encode(x)

# Chain exact ops without decoding between steps
res = s.mul(s.add(ex, ex), ex)
out = s.decode(res)

# One-shot affine exact arithmetic
one = s.one_shot_affine(x, multiplier=1_000_003, addend=7)

# Hot-loop affine exact arithmetic (stay in residue space, decode once)
hot = s.hot_loop_affine(x, multiplier=1_000_003, addend=7, iterations=1000)
```

---

## Operations

| Function | Description |
|---|---|
| `rns.encode(x)` | `uint64[] -> (r0, r1, r2)` residue arrays |
| `rns.decode(r0, r1, r2)` | residues -> `uint64[]` via Garner reconstruction |
| `rns.add(*ea, *eb)` | exact addition mod each rail |
| `rns.sub(*ea, *eb)` | exact subtraction mod each rail |
| `rns.mul(*ea, *eb)` | exact multiplication mod each rail |
| `rns.div_(*ea, *eb)` | exact division, divisor must be invertible on every rail |
| `rns.fma(*ea, *eb, *ec)` | fused multiply-add: `(a*b)+c` |
| `rns.op(*ea, *eb, code)` | generic op: `0=add 1=mul 2=sub 3=div` |

---

## Division constraint

Division requires the divisor to be invertible on all three rails:

- `b % 127 != 0`
- `b % 8191 != 0`
- `b` must be odd

A quick way to sanitize values:

```python
b = np.where(b % 2 == 0, b + 1, b)
b = np.where(b % 127 == 0, b + 2, b)
b = np.where(b % 8191 == 0, b + 4, b)
b = b % rns.M
```

---

## Data model

- input arrays to `encode(...)` are treated as `uint64`
- values outside `[0, M)` are reduced mod `M` during encode
- all residue rails are returned as `uint16`
- high-level `EncodedArray` objects store three read-only `uint16` rails

---

## Performance

### Benchmark (Google Colab, x86_64, AVX2 enabled)

Workload: `x = (x * 1_000_003 + 7) mod M`, N=1,000,000 nodes, 1,000 iterations, 10-run average.

| Library | M ops/s | Exact? |
|---|---|---|
| RNS `fma` (arith + decode) | **785.4** | yes |
| RNS `mul+add` | 441.0 | yes |
| RNS arith only | 795.2 | yes |
| Float64 (NumPy) | 10.4 | **no** — drifts at iter ~4 |
| Python BigInt | 3.96 | yes |
| GMP (gmpy2) | 2.72 | yes |
| FLINT (python-flint) | 3.77 | yes |

- **~76x faster than float64** — and float64 is wrong. RNS is right.
- **~217x faster than Python BigInt**
- **~307x faster than GMP**
- **~225x faster than FLINT**
- float64 accumulated an average error of **22 billion units** by iteration 1,000
- encode+decode overhead amortizes to **~1.15%** of runtime over a 1,000-iteration pipeline

### Round-trip breakdown (1 fma op)

| Step | Share |
|---|---|
| Encode | 34.0% |
| Arithmetic | 6.2% |
| Decode | 60.1% |

The arithmetic kernel is cheap relative to data movement. The performance win comes from staying in residue space for many iterations and decoding once at the end.

### Practical takeaway

- use `one_shot_affine(...)` for a clean convenience call
- use `hot_loop_affine(...)` for throughput
- use raw `fma(...)` for the thinnest possible path into the kernel
- stay in residue space as long as possible — decode once

---

## Why RNS?

In a Residue Number System, addition and multiplication have **no carry propagation between rails**. Each rail is independent. That makes RNS attractive for:

- exact modular arithmetic within a fixed dynamic range
- SIMD-friendly kernels
- workloads that benefit from repeated operations before decode
- parallel computation

---

## How it works

### Encode

```
x -> (x mod 127, x mod 8191, x mod 65536)
```

### Operate

Each rail is processed independently. Addition:

```
(a0 + b0) mod 127
(a1 + b1) mod 8191
(a2 + b2) mod 65536
```

### Decode

Garner-style CRT reconstruction:

```
t0 = r0
t1 = (r1 - t0) * inv(127 mod 8191) mod 8191
t2 = (r2 - (t0 + t1*127)) * inv(127*8191 mod 65536) mod 65536
x  = t0 + t1*127 + t2*127*8191
```

For mod `127` and mod `8191`, reduction uses the Mersenne fold trick:

```
x mod (2^k - 1) = (x & mask) + (x >> k)
```

This avoids general division in those reductions.

---

## Building from source

```bash
git clone https://github.com/playfularchitect/rns_engine.git
cd rns_engine
pip install -e .
pytest tests/ -v
```

### Enable AVX2 for a local build

```bash
RNS_ENGINE_ENABLE_AVX2=1 pip install .
```

Requirements: Python 3.10+, C++17 compiler, NumPy, pybind11.

---

## Introspection

```python
import rns_engine as rns
rns.info()

rns.M        # dynamic range
rns.M0       # 127
rns.M1       # 8191
rns.M2       # 65536
rns.HAS_AVX2 # True if built with AVX2
```

---

## Changelog

### v0.3.0
- adds `Session()` high-level API with LRU encode cache
- adds `EncodedArray` for chained exact arithmetic
- adds `one_shot_affine(...)` and `hot_loop_affine(...)`

### v0.2.0
- all rail arrays now `uint16` (was `uint32` for rail 1 in v0.1.x)
- new `fma(a, b, c)` op: `a*b+c` in one kernel call (~1.8x faster than `add(*mul(a,b), c)`)
- direct AVX2 loads — eliminates temp buffer copies
- **breaking change**: rail-1 array dtype changed from `uint32` to `uint16`

### v0.1.0
- initial release

---

## License

Apache 2.0 — Copyright 2026 Evan Wesley


