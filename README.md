# rns_engine

**Exact arithmetic, built to compete with floating-point speed.**

G4 is a **general autonomous search and optimization system**. Give it a problem, a legal space of possible solutions, and a way to measure or certify success. G4 explores that space, learns from the results, decides what to try next, can promote or create new strategies when the declared problem space permits it, and preserves the strongest verified solutions.

**G4 Series 1 applies that system to one problem: exact GPU matrix multiplication on NVIDIA Tesla T4.** Series 1 is a frozen application of G4, not a definition of what G4 can work on.

For speed, Series 1 compares its exact GEMM implementations against NVIDIA's optimized cuBLASLt FP16-input GEMM. **FP16 is the fast floating-point speed baseline; G4 is exactness-gated separately.** The benchmark asks whether exact arithmetic can compete with floating-point-class speed, not whether FP16 provides the same correctness guarantee.

## Series 1 at a glance

| | Certified Series 1 contract |
|---|---|
| **Integer math** | signed INT8 matrices -> exact signed INT32 result |
| **Shared-scale rational math** | signed INT8 numerator matrix + one positive integer scale per matrix -> exact shared-scale result |
| **GPU** | NVIDIA Tesla T4 / compute capability 7.5 |
| **Shapes** | frozen set of 1,024 `(M,N,K)` GEMM shapes |
| **Unsupported input** | fails closed; no silent floating-point fallback |

A shared-scale rational matrix has **one common scale/denominator for the whole matrix**. Scales are positive Python integers, the output scale is their exact product, and optional `reduce=True` normalization is explicit rather than automatic. Series 1 does not claim arbitrary per-element rational denominators.

## Frozen integer scorecard

| Metric | Result |
|---|---:|
| **Record** | **938 wins / 86 losses / 0 ties / 0 errors** |
| **All-shape geomean** | **1.235x** |
| **All-shape median** | **1.257x** |
| **Win-only geomean** | **1.311x** |
| **G4 throughput on 86 losing shapes** | **64.60% of NVIDIA** |

## Frozen shared-scale rational scorecard

| Metric | Result |
|---|---:|
| **Record** | **870 wins / 110 losses / 41 ties / 3 errors** |
| **Win-only geomean** | **1.417x** |
| **Win-only median** | **1.406x** |
| **Best certified win** | **2.978x** |
| **All-shape aggregate** | not claimed by the frozen public summary |

Speedup = NVIDIA time / G4 time.  
**Exact** describes correctness. **Win** describes speed.

The three final rational errors were **post-measurement bookkeeping exceptions** (`KeyError: 'forecast_debt_bits'`), not silent arithmetic mismatches. Their rows had already produced paired timings and recorded real non-integer inputs with the range and FP16-value-set proof flags passing. They remain classified as errors and are not counted as wins or ties.

### Why FP16?

FP16 cuBLASLt is the Series 1 **speed baseline** because the Series 1 question is whether exact integer and shared-scale rational arithmetic can reach floating-point-class throughput on Tesla T4. It is not the exactness baseline. Series 1 does **not** claim that G4 beats every possible NVIDIA integer GEMM configuration.

---

## Install

```bash
pip install -U rns_engine
```

Supported Python versions: **3.10–3.14**.

The G4 Series 1 GPU path currently requires Linux, an NVIDIA Tesla T4, and `nvcc` for first-use compilation of the verified execution sources. Standard Google Colab T4 runtimes provide the required CUDA toolchain.

---

## Use G4 Series 1

```python
import numpy as np
import rns_engine as rns

A = (np.arange(16 * 16).reshape(16, 16) % 17 - 8).astype(np.int8)
B = (np.arange(16 * 16).reshape(16, 16) % 13 - 6).astype(np.int8)

C = rns.g4_matmul(A, B)

assert C.dtype == np.int32
```

`g4_matmul()` accepts signed-INT8 integer matrices on the certified Series 1 shapes and returns the exact signed-INT32 result.

### Benchmark timing vs `g4_matmul()` wall time

The frozen **integer** Series 1 headline is a **resident-data GPU GEMM benchmark**. It measures the computation with GPU data already resident and does not include Python-call overhead or CPU↔GPU transfer time.

The frozen **rational** benchmark has its own exact-result timing boundary and includes the rational metadata/bookkeeping required to produce the exact rational result. It still is not a one-shot CPU-NumPy `g4_matmul()` wall-time claim.

`g4_matmul()` is currently a **CPU-NumPy convenience API**: it accepts CPU arrays and returns a CPU NumPy-backed result. Its end-to-end wall time therefore includes data movement and runtime/wrapper work that the resident-data integer benchmark excludes.

Do **not** compare a pre-resident PyTorch GPU-kernel timing directly with CPU→CPU `g4_matmul()` wall time. A fair comparison must use the same timing boundary on both sides. Series 1 does not claim that one-shot CPU-NumPy→CPU-NumPy `g4_matmul()` calls beat pre-resident PyTorch GPU GEMM calls.

A `SharedScaleMatrix` represents one integer numerator matrix divided by one positive integer scale:

```python
left = rns.SharedScaleMatrix(A, scale=3)
right = rns.SharedScaleMatrix(B, scale=5)

C = rns.g4_matmul(left, right)

# Exactly (A @ B) / 15. No floating conversion.
print(C.numerators)
print(C.scale)
```

For every certified Series 1 shape, full-range signed-INT8 inputs fit safely in the signed-INT32 output contract. The frozen manifest has `K <= 4096`; the worst possible magnitude bound is therefore `4096 × 128 × 128 = 67,108,864`, well below signed INT32 range.

The reusable caller-supplied execution path was certified on **1,024 / 1,024 supported shapes on actual Tesla T4 hardware** with full-range signed-INT8 data. Additional extreme-value and sparse-extreme tests also passed.

The package SHA-pins the **public execution-source bodies**, shape manifest, certification receipt, compile flags, and provenance hashes. The public execution sources contain the frozen Series 1 runtime; the private G4 search/learning machinery is not required to replay the result.

---

## Reproduce the benchmarks

Show the frozen Series 1 evidence without a GPU:

```python
import rns_engine as rns

rns.g4_results()
rns.g4_results("integer")
rns.g4_results("rational")
```

`g4_results()` prints the frozen scorecards, including aggregate integer performance, integer loss-side magnitude, and the rational win/loss/tie/error breakdown.

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

The benchmark paths are **replay-only**: they rerun the frozen Series 1 implementations. They do not run a new G4 search.

The public runners report exactness, speed classifications, XOPS/G4OPS, runtime/source provenance hashes, result-row hashes, and cryptographic run receipts.

### How speed certification works

The public replay first checks correctness, then uses repeated paired timings on the same T4 to classify speed.

The integer headline timing boundary is **resident-data GEMM execution**, not one-shot CPU-array API wall time. The rational headline includes its documented exact-result bookkeeping boundary. Neither scorecard is a claim about one-shot CPU-array `g4_matmul()` wall time.

For the current integer public replay, winner classification uses **31 paired timing blocks per shape**, **20,000 bootstrap resamples**, a **95% confidence interval**, a **1.002 promotion threshold**, and at least **20 / 31 winning blocks**. Runtime or exactness failures fail closed.

The historical frozen integer archive predates that uniform public replay rule and preserves its actual measurement depth: **827 shapes used 21 paired blocks, 190 used 31, and 7 used 127**. Exact replay passed before and after timing on all 1,024 frozen integer shapes.

Every certified rational winner used **31 paired timing blocks** and passed the non-integer-input, signed-range, and FP16-value-set proof gates.

See [`G4_SERIES1_EVIDENCE.md`](G4_SERIES1_EVIDENCE.md) for the detailed benchmark protocol, claim boundaries, and replay provenance.

---

## XOPS: exact throughput

**XOP** — one mathematically exact arithmetic operation.  
**XOPS** — exact arithmetic operations per second.  
**G4OPS** — XOPS delivered by a G4 implementation.

For GEMM, XOPS uses the same conventional arithmetic count as FLOPS:

```text
XOP count = 2 × M × N × K
```

That puts exact and floating-point work on the same **operation-count scale**. The correctness and representation semantics are still different.

The eligibility rule is strict:

```text
exactness PASS -> operations may count toward XOPS
exactness FAIL -> 0 XOPS credit
```

For rational GEMM, headline G4OPS uses the **end-to-end exact-result timing boundary**, including the rational metadata/bookkeeping required to produce the exact rational result. Only the conventional `2*M*N*K` GEMM operations receive XOP credit.

---

## What is G4?

G4 is **not a GPU-specific system**. GPU arithmetic is simply the problem Series 1 applies it to.

More generally, G4 works over a declared solution space and a declared success test. It can search candidate constructions, learn from measured outcomes, choose what to explore next, preserve useful experience, reject failures, and promote or create new strategies when the declared problem space permits it.

In Series 1, the feedback happens to be unusually objective: an implementation must return the exact mathematical result, and then the hardware determines how fast it is.

`rns_engine` exposes the frozen Series 1 evidence, reproducible replay paths, the certified execution library, and lower-level exact-arithmetic tools used by this application. The private G4 search machinery is not required to replay the published result.

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

These tools are supporting exact-arithmetic machinery. The main public G4 Series 1 surface is:

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
- The Series 1 user-math fast path accepts signed-INT8 integer matrices or shared-scale rational matrices with signed-INT8 numerators.
- Integer outputs are exact signed INT32 matrices under the certified Series 1 contract.
- Shared-scale rational outputs are exact and use one positive Python-integer scale per matrix; output scale is the exact product of the inputs, with optional explicit reduction.
- Series 1 does not claim arbitrary per-element rational denominators or general rational normalization.
- Unsupported G4 shapes or representations fail closed rather than silently falling back to approximate floating point.
- The frozen integer benchmark reports **resident-data GPU GEMM speed**; one-shot `g4_matmul()` CPU-array wall time is a different timing boundary.
- Integer and rational benchmark scores are always reported separately.
- Future G4 generations are treated as separate Series rather than rewriting frozen Series 1 evidence.

---

## Evidence and integrity

The public Series 1 runtime and replay paths are integrity checked with SHA-256. Public benchmark runs include exactness gates and cryptographic receipts so fresh timing evidence can be distinguished from the frozen historical results.

Frozen headline scorecard:

```text
G4 exact integers vs NVIDIA FP16 speed baseline:
  938 G4 wins / 86 NVIDIA wins / 0 ties / 0 errors
  all 1024: 1.235x geomean | 1.257x median
  G4 wins: 1.311x geomean | 1.289x median | 2.647x best
  NVIDIA wins: G4 retains 64.60% throughput on geometric average

G4 exact shared-scale rationals vs NVIDIA FP16 speed baseline:
  870 G4 wins / 110 NVIDIA wins / 41 ties / 3 errors
  certified G4 wins: 1.417x geomean | 1.406x median | 2.978x best
  no all-1024 rational speedup aggregate is claimed by the frozen public summary
```

Timing classifications can move between fresh runs because clocks, thermals, driver state, and measurement noise move. Mathematical exactness must not.

See [`G4_SERIES1_EVIDENCE.md`](G4_SERIES1_EVIDENCE.md) for the detailed evidence contract.

---

## License

`rns_engine` is licensed under **AGPL-3.0-only**. See [`LICENSE`](LICENSE).
