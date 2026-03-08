

````
# rns_engine

**Exact integer arithmetic via a 3-rail Residue Number System (RNS), with optional AVX2 acceleration.**

No floating point. No approximation. Exact results within the engine’s dynamic range.

---

## What it does

Standard Python integers are exact but slow. NumPy is fast but can overflow fixed-width integer dtypes and is not designed as an exact modular arithmetic engine. `rns_engine` gives you **exact arithmetic modulo a fixed dynamic range** by decomposing integers into residues across three coprime moduli:

- `127`
- `8191`
- `65536`

Arithmetic is performed independently on each rail, then reconstructed with the Chinese Remainder Theorem (Garner form).

**Dynamic range:** `[0, 68,174,282,752)`

That is:

`127 × 8191 × 65536 = 68,174,282,752`

---

## New in v0.3.0

- keeps the exact `v0.2` core and adds a **high-level session/cache API**
- `Session()` lets you cache encoded inputs/constants instead of paying repeated encode cost
- `EncodedArray` makes chained exact arithmetic easier to read
- adds `one_shot_affine(...)` and `hot_loop_affine(...)`
- all rail arrays are `uint16`
- includes fused multiply-add: `fma(a, b, c) = a*b + c`

---

## Install

```bash
pip install rns_engine
````

### AVX2 note

`rns_engine` can be built with AVX2 acceleration on supported x86_64 systems.

`HAS_AVX2` is a **build-time property** of the compiled extension, not runtime CPU detection. A default portable build does not assume AVX2 unless explicitly enabled during build.

For most users, `pip install rns_engine` is the right starting point.

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

# cache-aware encode
ex = s.encode(x)

# chain exact ops without decoding between steps
res = s.mul(s.add(ex, ex), ex)
out = s.decode(res)

# one-shot affine exact arithmetic
one = s.one_shot_affine(x, multiplier=1_000_003, addend=7)

# hot-loop affine exact arithmetic (stay in residue space, decode once)
hot = s.hot_loop_affine(x, multiplier=1_000_003, addend=7, iterations=1000)
```

---

## Operations

| Function                 | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| `rns.encode(x)`          | `uint64[] -> (r0, r1, r2)` residue arrays                |
| `rns.decode(r0, r1, r2)` | residues -> `uint64[]` via Garner reconstruction         |
| `rns.add(*ea, *eb)`      | exact addition mod each rail                             |
| `rns.sub(*ea, *eb)`      | exact subtraction mod each rail                          |
| `rns.mul(*ea, *eb)`      | exact multiplication mod each rail                       |
| `rns.div_(*ea, *eb)`     | exact division, divisor must be invertible on every rail |
| `rns.fma(*ea, *eb, *ec)` | fused multiply-add: `(a*b)+c`                            |
| `rns.op(*ea, *eb, code)` | generic op: `0=add 1=mul 2=sub 3=div`                    |

---

## Division constraint

Division requires the divisor to be invertible on all three rails:

* `b % 127 != 0`
* `b % 8191 != 0`
* `b` must be odd

If the divisor is not invertible, `rns_engine` raises an error.

```python
import numpy as np
import rns_engine as rns

b = np.array([3, 5, 7, 9], dtype=np.uint64) % rns.M
eb = rns.encode(b)
```

A quick way to sanitize values for experiments:

```python
b = np.where(b % 2 == 0, b + 1, b)
b = np.where(b % 127 == 0, b + 2, b)
b = np.where(b % 8191 == 0, b + 4, b)
b = b % rns.M
```

---

## Data model

* input arrays to `encode(...)` are treated as `uint64`
* values outside `[0, M)` are reduced mod `M` during encode
* all residue rails are returned as `uint16`
* high-level `EncodedArray` objects store three read-only `uint16` rails

---

## Performance

Performance depends on:

* whether the extension was built with AVX2
* input size
* operation mix
* Python overhead vs hot-loop reuse

In general:

* `add`, `sub`, `mul`, and `fma` are the fast paths
* `div_` is slower because it requires modular inverse work per element
* repeated workloads benefit most from staying in residue space and decoding once

You should benchmark on your own hardware and workload before making hard throughput claims.

---

## Why RNS?

In a Residue Number System, addition and multiplication have **no carry propagation between rails**. Each rail is independent.

That makes RNS attractive for:

* exact modular arithmetic within a fixed dynamic range
* parallel computation
* SIMD-friendly kernels
* workloads that benefit from repeated operations before decode

---

## How it works

Three coprime moduli:

* `m0 = 127`
* `m1 = 8191`
* `m2 = 65536`

Dynamic range:

`M = 127 × 8191 × 65536 = 68,174,282,752`

### Encode

```text
x -> (x mod 127, x mod 8191, x mod 65536)
```

### Operate

Each rail is processed independently.

For example, addition is:

```text
(a0 + b0) mod 127
(a1 + b1) mod 8191
(a2 + b2) mod 65536
```

### Decode

Garner-style reconstruction:

```text
t0 = r0
t1 = (r1 - t0) * inv(127 mod 8191) mod 8191
t2 = (r2 - (t0 + t1*127)) * inv(127*8191 mod 65536) mod 65536
x  = t0 + t1*127 + t2*127*8191
```

For mod `127` and mod `8191`, reduction uses the Mersenne-style fold trick:

```text
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

### Enable AVX2 for a local source build

```bash
RNS_ENGINE_ENABLE_AVX2=1 pip install .
```

You need:

* Python 3.10+
* a C++17-capable compiler
* NumPy
* pybind11

---

## Introspection

```python
import rns_engine as rns
rns.info()
```

You can also inspect:

* `rns.M`
* `rns.M0`, `rns.M1`, `rns.M2`
* `rns.HAS_AVX2`

---

## License

Apache 2.0

```



