# rns_engine

Exact integer arithmetic for Python using a 3-rail Residue Number System (RNS).

Most integer arithmetic in Python and NumPy looks exact but isn't — or is exact but slow. Here's the tradeoff you're usually forced to make:

| Option | Fast | Exact |
|---|---|---|
| Python `int` | ❌ | ✅ arbitrary precision |
| NumPy `int64` | ✅ | ❌ silently overflows |
| `decimal` / `gmpy2` | ❌ | ✅ |
| **rns_engine (AVX2)** | **✅** | **✅** |

`rns_engine` breaks that tradeoff. On AVX2 hardware (Intel/AMD), it processes 16 values simultaneously in SIMD registers while guaranteeing bit-perfect results — **errors are structurally impossible**, not just unlikely. There is no rounding mode to configure, no overflow to guard against, no precision to tune. The architecture makes incorrect results unrepresentable.

```python
pip install rns_engine
```

---

## What it does

Numbers are encoded as residues across three coprime moduli (`127 × 8191 × 65536`), arithmetic is performed independently on each rail in parallel, and results are reconstructed exactly via Garner's CRT algorithm. The Chinese Remainder Theorem guarantees a unique exact answer for every operation within the dynamic range — the same way a lock with three independent tumblers has exactly one key.

**Dynamic range:** `[0, 68,174,282,752)` — about 68 billion unique integers representable exactly.

---

## Quick start

```python
import rns_engine as rns
import numpy as np

rns.info()
# rns_engine v0.1.0
#   Dynamic range : [0, 68,174,282,752)
#   Moduli        : 127 × 8191 × 65536
#   AVX2          : yes   ← Intel/AMD Linux/Windows
#   Operations    : add  sub  mul  div_

a = np.array([1000, 2000, 3000], dtype=np.uint64)
b = np.array([500,  800,  1200], dtype=np.uint64)

ra = rns.encode(a)
rb = rns.encode(b)

result = rns.decode(*rns.add(*ra, *rb))
print(result)  # [1500 2800 4200] — exact, always
```

---

## Operations

```python
rns.add(*ra, *rb)       # addition
rns.sub(*ra, *rb)       # subtraction
rns.mul(*ra, *rb)       # multiplication
rns.div_(*ra, *rb)      # division (b must be coprime to all moduli)
rns.op(*ra, *rb, opcode)  # 0=add 1=mul 2=sub 3=div
```

All operations take and return `(r0, r1, r2)` tuples of numpy arrays.

---

## Performance

| Platform | AVX2 | add/sub/mul | div |
|---|---|---|---|
| Linux x86-64 (Intel/AMD) | ✅ yes | ~200–420 M ops/s | ~1.6 M ops/s |
| Windows x86-64 (Intel/AMD) | ✅ yes | ~200–420 M ops/s | ~1.6 M ops/s |
| Apple Silicon (M1/M2/M3) | ❌ no | scalar fallback | scalar fallback |
| Linux ARM64 | ❌ no | scalar fallback | scalar fallback |

### Apple Silicon / Mac users

AVX2 is an Intel/AMD instruction set — Apple Silicon Macs don't support it, so the library falls back to scalar arithmetic. **The results are identical and fully exact**, just slower at large scale.

For small arrays (thousands of elements) you won't notice any difference. For heavy workloads (millions of operations), **use Google Colab** — Colab runs on Linux x86 CPUs with AVX2 enabled:

```python
# In a Colab notebook:
!pip install rns_engine
import rns_engine as rns
rns.info()  # AVX2: yes
```

Speed only matters if you're processing large batches. For verification, testing, and moderate workloads your Mac is fine.

---

## How it works

RNS represents each integer as a tuple of residues:

```
x  →  (x mod 127,  x mod 8191,  x mod 65536)
```

Arithmetic on residues is independent per rail and never carries between them — no overflow, no interaction. Reconstruction uses Garner's algorithm, which recovers the original integer exactly from its residues via the Chinese Remainder Theorem.

Division requires the divisor to be coprime to all three moduli (odd, nonzero mod 127 and 8191).

---

## Use cases

- Exact arithmetic pipelines where floating-point error is unacceptable
- Weight encoding and integrity verification for neural networks
- Cryptographic and number-theoretic computations
- High-throughput batch integer arithmetic on CPU

---

## License

MIT
