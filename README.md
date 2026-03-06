# rns_engine

**Exact integer arithmetic via AVX2-accelerated Residue Number System (RNS).**

No floating point. No approximation. Errors are structurally impossible.

---

## What it does

Standard Python integers are exact but slow. NumPy is fast but uses floating point or silently overflows. `rns_engine` gives you **exact integer arithmetic at hundreds of millions of operations per second** — the best of both worlds.

It works by decomposing integers into residues across three coprime moduli (127, 8191, 65536), performing all operations in residue space using AVX2 SIMD instructions, and reconstructing exact results via the Chinese Remainder Theorem.

**Dynamic range:** `[0, 68,174,282,752)` — about 68 billion.

## Install

```bash
pip install rns_engine
```

Requires a CPU with AVX2 (any Intel/AMD since ~2013). Falls back to scalar arithmetic on ARM and older hardware.

## Quick start

```python
import rns_engine as rns
import numpy as np

# Works on arrays of uint64
a = np.array([123456789, 999999999], dtype=np.uint64)
b = np.array([987654321, 111111111], dtype=np.uint64)

# Encode once
ea = rns.encode(a)   # returns (r0, r1, r2) residue arrays
eb = rns.encode(b)

# Operate in residue space — no intermediate decode needed
result = rns.decode(*rns.mul(*ea, *eb))   # exact multiplication

# Chain multiple operations — decode once at the end
s1 = rns.add(*ea, *eb)      # a + b
s2 = rns.mul(*s1, *eb)      # (a + b) * b
s3 = rns.sub(*s2, *ea)      # (a + b) * b - a
out = rns.decode(*s3)        # one decode, three operations
```

## Operations

| Function | Description |
|----------|-------------|
| `rns.encode(x)` | `uint64[]` → `(r0, r1, r2)` residue arrays |
| `rns.decode(r0, r1, r2)` | Residues → `uint64[]` via Garner's algorithm |
| `rns.add(*ea, *eb)` | Exact addition |
| `rns.sub(*ea, *eb)` | Exact subtraction |
| `rns.mul(*ea, *eb)` | Exact multiplication |
| `rns.div_(*ea, *eb)` | Exact division (b must be coprime to all moduli) |
| `rns.op(*ea, *eb, code)` | Generic: `0`=add `1`=mul `2`=sub `3`=div |

### Division constraint

Division requires `b` to be invertible on all three rails:
- `b % 127  != 0`
- `b % 8191 != 0`
- `b % 65536` is **odd** (coprime to 2^16)

```python
# Safe way to ensure b is valid for division:
b = np.where(b % 2 == 0, b + 1, b)   # make odd
b = np.where(b % 127  == 0, b + 2, b)
b = np.where(b % 8191 == 0, b + 4, b)
b = b % rns.M
```

## Performance

On a machine with AVX2 (tested on Google Colab T4):

| Operation | Throughput |
|-----------|-----------|
| add | ~200–400 M ops/sec |
| sub | ~200–400 M ops/sec |
| mul | ~200–400 M ops/sec |
| div | ~1.6 M ops/sec (scalar modinv per element) |

## Why RNS?

In a Residue Number System, **addition and multiplication have no carry propagation between digits**. Each residue rail is independent. This makes RNS ideal for:

- **Exact arithmetic** — results are always correct within the dynamic range
- **Parallel computation** — rails can run simultaneously
- **Error detection** — CRT reconstruction fails loudly if any rail is corrupted
- **Cryptography** — modular arithmetic is the native language of RSA, ECC, etc.

## How it works

Three coprime moduli: `m0 = 127`, `m1 = 8191`, `m2 = 65536`

Dynamic range: `M = 127 × 8191 × 65536 = 68,174,282,752`

**Encode:** `x → (x mod 127, x mod 8191, x mod 65536)`

**Operate:** each rail independently, e.g. add: `(a+b) mod mᵢ` per rail

**Decode (Garner's algorithm):**
```
t0 = r0
t1 = (r1 - t0) × inv(127, 8191)  mod 8191
t2 = (r2 - t0 - t1×127) × inv(127×8191, 65536)  mod 65536
x  = t0 + t1×127 + t2×127×8191
```

Mod 127 and mod 8191 reductions use the Mersenne-prime trick:
`x mod (2^k - 1) = (x & mask) + (x >> k)` — no division needed.

## Building from source

```bash
git clone https://github.com/playfularchitect/rns_engine
cd rns_engine
pip install pybind11 numpy
pip install -e .
pytest tests/ -v
```

Requires `g++` with C++17 support.

## License

MIT
