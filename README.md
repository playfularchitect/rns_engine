# rns_engine

**Exact arithmetic without the usual speed-or-accuracy tradeoff.**

G4 Series 1 contains two separate exact GEMM systems tested **directly against NVIDIA's optimized cuBLASLt FP16-input GEMM implementation** on the same class of NVIDIA Tesla T4 GPU across **1,024 matrix-multiplication shapes**.

| Exact arithmetic tested | Faster than NVIDIA | Win rate |
|---|---:|---:|
| **G4 exact integer GEMM** | **938 / 1,024 shapes** | **91.60%** |
| **G4 exact rational GEMM** | **870 / 1,024 shapes** | **84.96%** |

These are **two separate benchmark results**. Integer and rational systems are run, certified, and scored independently.

> **Exact** describes correctness. **Win** describes speed. A calculation remains mathematically exact even on a shape where NVIDIA wins the timing comparison.

The tests use standard Tesla T4 hardware available through Google Colab. G4 Series 1 is frozen to the Tesla T4 / compute capability 7.5 contract.

---

## Install

```bash
pip install -U rns_engine
```

Supported Python versions: **3.10–3.14**.

The G4 Series 1 GPU path currently requires Linux, an NVIDIA Tesla T4, and `nvcc` for first-use compilation of the verified execution sources. Standard Google Colab T4 runtimes provide the required CUDA toolchain.

---

## Use G4

```python
import numpy as np
import rns_engine as rns

A = np.arange(16 * 16, dtype=np.int16).reshape(16, 16) % 17 - 8
B = np.arange(16 * 16, dtype=np.int16).reshape(16, 16) % 13 - 6

C = rns.g4_matmul(A, B)

assert C.dtype == np.int32
```

`g4_matmul()` accepts signed-INT8 integer matrices on the certified Series 1 shapes and returns exact signed-INT32 output.

Exact shared-scale rationals use the same API:

```python
left = rns.SharedScaleMatrix(A, scale=3)
right = rns.SharedScaleMatrix(B, scale=5)

C = rns.g4_matmul(left, right)

# Exactly (A @ B) / 15. No floating conversion.
print(C.numerators)
print(C.scale)
```

Current Series 1 fast-path contract:

- hardware: **NVIDIA Tesla T4 / compute capability 7.5**;
- shapes: the frozen **1,024 Series 1 `(M,N,K)` shapes**;
- integer inputs: signed INT8 matrices;
- integer output: exact signed INT32 matrix;
- rational inputs: `SharedScaleMatrix` with signed-INT8 numerators and positive integer shared scales;
- rational output: exact `SharedScaleMatrix`;
- unsupported shapes or representations **fail closed** — no silent floating-point fallback.

The reusable caller-supplied execution path was physically certified on **1,024 / 1,024 supported shapes** with full-range signed-INT8 data. Additional extreme-value and sparse-extreme tests also passed.

The package SHA-pins the stripped public execution-source bodies, shape manifest, certification receipt, compile flags, and provenance hashes.

**FP32-class G4 Series 2 support is in development and is intended to extend this same `g4_matmul()` API after physical certification.**

---

## Reproduce the benchmarks

Show the frozen Series 1 evidence without a GPU:

```python
import rns_engine as rns

rns.g4_results()
rns.g4_results("integer")
rns.g4_results("rational")
```

On a Tesla T4 there are three benchmark entry points:

```python
import rns_engine as rns

# Integers first, then rationals. Scores stay separate.
rns.g4_benchmark("full")

# Run either benchmark independently.
rns.g4_integer_benchmark("full")
rns.g4_rational_benchmark("full")
```

Modes:

```text
quick      24 shapes
standard  128 shapes
full     1,024 shapes
```

`g4_benchmark()` is only a convenience runner. It runs **G4 INTEGERS first and G4 RATIONALS second**, but it does **not** combine them into one win percentage.

The public runners report exactness, speed wins/losses/ties, XOPS/G4OPS, reproducibility, runtime/source provenance hashes, all-result-row hashes, and cryptographic run receipts.

The benchmark paths are replay-only: they rerun frozen Series 1 implementations rather than performing a new G4 search.

See [`G4_SERIES1_EVIDENCE.md`](G4_SERIES1_EVIDENCE.md) for the benchmark protocol, claim boundaries, and replay provenance.

---

## XOPS: exact throughput

**XOP** — one mathematically exact arithmetic operation.  
**XOPS** — exact arithmetic operations per second.  
**G4OPS** — XOPS delivered by a G4 implementation.

For GEMM, XOPS intentionally uses the same conventional operation count as FLOPS:

```text
XOP count = 2 × M × N × K
```

That keeps exact and floating-point throughput directly comparable.

The eligibility rule is strict:

```text
exactness PASS -> operations may count toward XOPS
exactness FAIL -> 0 XOPS credit
```

Every public benchmark prints this key before reporting XOPS/G4OPS so saved output remains self-explaining.

For rational GEMM, headline G4OPS uses the **end-to-end exact-result timing boundary**, including the rational metadata/bookkeeping required to produce the exact rational result. Only the conventional `2*M*N*K` GEMM operations receive XOP credit.

---

## What is G4?

G4 is the privately developed autonomous optimization/research system that discovered the benchmarked implementations.

`rns_engine` exposes the frozen evidence, reproducible replay paths, the certified Series 1 execution library, and lower-level exact arithmetic tools used by the project.

---

## Lower-level exact arithmetic

The package also exposes lower-level exact arithmetic primitives independently of G4.

Basic residue arithmetic:

```python
import numpy as np
import rns_engine as rns

x = np.array([123456789, 999999999], dtype=np.uint64)
y = np.array([987654321, 111111111], dtype=np.uint64)

ex = rns.encode(x)
ey = rns.encode(y)

exact_add = rns.decode(*rns.add(*ex, *ey))
exact_mul = rns.decode(*rns.mul(*ex, *ey))
```

Signed values:

```python
values = np.array([-5, 0, 7], dtype=np.int64)
rails = rns.encode_signed(values)
round_trip = rns.decode_signed(*rails)

assert np.array_equal(round_trip, values)
```

A centered decode gives the canonical signed representative. When proving that an unknown mathematical result did not wrap, use the signed range-certificate APIs as well.

Useful lower-level entry points include:

```text
encode / decode
encode_signed / decode_signed
add / sub / mul / div_ / fma
Session / SignedSession
certify_signed_bound
certify_signed_dot_bound
SharedScaleMatrix
exact_integer_matmul
exact_shared_scale_gemm
```

These tools are supporting exact-arithmetic machinery. The main public G4 surface is:

```text
g4_results
g4_benchmark
g4_integer_benchmark
g4_rational_benchmark
g4_matmul
```

---

## Scope and limitations


- **G4 Series 1 is frozen to Tesla T4 / compute capability 7.5.** Results on another GPU are a different experiment.
- The benchmark catalog contains the declared **1,024 GEMM shapes**, not every possible matrix size.
- The current G4 Series 1 user-math fast path accepts signed-INT8 integer matrices or shared-scale rational matrices with signed-INT8 numerators.
- Integer outputs are exact signed INT32 matrices under the certified Series 1 contract.
- Shared-scale rational outputs are exact; Series 1 does not claim arbitrary per-element denominator support.
- Unsupported G4 shapes or representations fail closed rather than silently falling back to approximate floating point.
- Integer and rational benchmark scores are always reported separately.
- FP32-class G4 Series 2 is not part of Series 1 and is not claimed as shipped.

---

## Evidence and integrity

The public Series 1 runtime and replay paths are integrity checked with SHA-256. Public benchmark runs include exactness gates and cryptographic receipts so fresh timing evidence can be distinguished from the frozen historical headline results.

Frozen headline results remain:

```text
G4 exact integers  vs NVIDIA FP16: 938 / 1024 faster (91.60%)
G4 exact rationals vs NVIDIA FP16: 870 / 1024 faster (84.96%)
```

Timing classifications can move between fresh runs because clocks, thermals, driver state, and measurement noise move. Mathematical exactness must not.

See [`G4_SERIES1_EVIDENCE.md`](G4_SERIES1_EVIDENCE.md) for the detailed evidence contract.

---

## License

`rns_engine` is licensed under **AGPL-3.0-only**. See [`LICENSE`](LICENSE).
