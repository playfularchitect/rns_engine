# rns_engine

**Exact arithmetic without the usual speed-or-accuracy tradeoff.**

G4 Series 1 contains two separate exact-arithmetic systems tested **directly against NVIDIA's optimized cuBLASLt FP16-input GEMM implementation** on the same class of NVIDIA Tesla T4 GPU across **1,024 matrix-multiplication shapes**.

The tests were run on **standard Tesla T4 hardware provided through Google Colab** — publicly available commodity cloud hardware, not a custom GPU or private lab rig.

| Exact arithmetic tested | Faster than NVIDIA | Win rate |
|---|---:|---:|
| **Exact integer GEMM** | **938 / 1,024 shapes** | **91.60%** |
| **Exact rational GEMM** | **870 / 1,024 shapes** | **84.96%** |

These are **two separate benchmark results**. The integer and rational systems are run, certified, and scored independently.

> **Exact** describes correctness. **Win** describes speed. A calculation remains mathematically exact even on a shape where NVIDIA wins the timing comparison.

---

## XOPS: exact throughput

G4 uses a throughput metric for exact arithmetic:

**XOP** — one mathematically exact arithmetic operation.  
**XOPS** — exact arithmetic operations per second.  
**G4OPS** — XOPS delivered by a G4 implementation.

For GEMM, XOPS intentionally uses the **same conventional operation count as FLOPS**:

```text
XOP count = 2 × M × N × K
```

That keeps exact and floating-point throughput directly comparable instead of changing the counting rule.

The eligibility rule is strict:

```text
exactness PASS -> operations may count toward XOPS
exactness FAIL -> 0 XOPS credit
```

Every public benchmark prints this key before reporting XOPS/G4OPS, so a saved notebook is self-explaining even if the reader never opened this README.

For rational GEMM, the headline G4OPS timing boundary is deliberately **end-to-end**: exact GEMM execution plus the rational metadata/bookkeeping required to produce the exact rational result. The benchmark still credits only the conventional `2*M*N*K` GEMM operations, making that rate conservative rather than hiding bookkeeping outside the clock.

---

## Use G4 directly

Series 1 is not only a benchmark replay. The package carries the SHA-pinned, public-safe execution sources for the physically certified G4 runtime covering the current 1,024-shape catalog. On first Tesla T4 use, `rns_engine` verifies that source bundle, compiles the required execution family with the frozen V7 CUDA flags, and caches the resulting kernel locally:

```python
import numpy as np
import rns_engine as rns

A = np.arange(16 * 16, dtype=np.int16).reshape(16, 16) % 17 - 8
B = np.arange(16 * 16, dtype=np.int16).reshape(16, 16) % 13 - 6

# Inputs are converted only after the exact signed-INT8 range check.
C = rns.g4_matmul(A, B)

assert C.dtype == np.int32
```

`g4_matmul()` accepts exact signed-INT8 integer matrices on the certified Series 1 shapes and returns exact signed-INT32 output.

Exact shared-scale rationals use the same API:

```python
left = rns.SharedScaleMatrix(A, scale=3)
right = rns.SharedScaleMatrix(B, scale=5)

C = rns.g4_matmul(left, right)

# Exactly (A @ B) / 15. No floating conversion.
print(C.numerators)
print(C.scale)
```

Current fast-path contract:

- hardware: **NVIDIA Tesla T4 / compute capability 7.5**;
- shapes: the frozen **1,024 Series 1 `(M,N,K)` shapes**;
- integer inputs: signed INT8 matrices;
- integer output: exact signed INT32 matrix;
- rational inputs: `SharedScaleMatrix` with signed-INT8 numerators and positive integer shared scales;
- rational output: exact `SharedScaleMatrix`;
- unsupported shapes or representations **fail closed** — no silent floating-point fallback.

The reusable execution path was physically certified on all **1,024 / 1,024** supported shapes with caller-supplied full-range signed-INT8 data. Additional extreme-value and sparse-extreme subsets also passed. The public package SHA-pins the exact stripped execution-source bodies, shape manifest, certification receipt, compile flags, and source archive provenance; private G4 search/learning code is not included.

**FP32-class G4 Series 2 support is in development and is intended to extend this same `g4_matmul()` API once physically certified.**

---

## Reproduce the evidence yourself

Install the package:

```bash
pip install -U rns_engine
```

Show the frozen Series 1 evidence without requiring a GPU:

```python
import rns_engine as rns

rns.g4_results()
rns.g4_results("integer")
rns.g4_results("rational")
```

On a **Tesla T4 / compute capability 7.5**, there are three benchmark entry points:

```python
import rns_engine as rns

# Integers first, then rationals. Scores remain separate.
rns.g4_benchmark("full")

# Run only one species if desired.
rns.g4_integer_benchmark("full")
rns.g4_rational_benchmark("full")
```

Modes are shared across all three:

```text
quick     24 shapes
standard  128 shapes
full      1,024 shapes
```

The public runners print progress/ETA, exactness, speed wins/losses/ties, XOPS/G4OPS, reproducibility, source/runtime provenance SHA-256s, all-result-row SHA-256s, and cryptographic run receipts.

`g4_benchmark()` is a convenience runner. It runs **G4 INTEGERS first and G4 RATIONALS second**, but it does **not** combine them into one win percentage.

All Series 1 public GPU APIs fail closed outside the frozen Tesla T4 hardware contract. Running them on another GPU would be a different experiment.

The benchmark paths are **replay-only**: they rerun frozen Series 1 implementations rather than performing a new G4 search.

See [`G4_SERIES1_EVIDENCE.md`](G4_SERIES1_EVIDENCE.md) for benchmark protocol, claim boundaries, and replay provenance.

---

## Public physical validation

The reusable integer + user-math runtime was built and certified on a Google Colab Tesla T4 with compute capability 7.5, driver `580.82.07`, and CUDA `12.8 / nvcc V12.8.93`.

Its full integer certification run produced:

- **1,024 / 1,024** exact integer calculations correct;
- **874 / 1,024** fresh G4 speed wins;
- **112 / 1,024** NVIDIA speed wins;
- **38 / 1,024** statistical ties;
- **953 / 1,024** faster/slower/tie classifications matching the frozen integer archive.

Those fresh timing numbers are a **reproduction run**, not a replacement for the frozen headline result of **938 / 1,024 = 91.60%**. Timing classifications can move with clocks, driver state, and normal measurement noise; mathematical exactness must not move.

The earlier fresh rational replay produced:

- **1,024 / 1,024** exact rational calculations correct;
- **834 / 1,024** fresh G4 speed wins;
- **114 / 1,024** NVIDIA speed wins;
- **76 / 1,024** statistical ties.

The frozen rational headline remains **870 / 1,024 = 84.96%**.

---

## What is G4?

G4 is the autonomous optimization/research system that discovered the benchmarked implementations.

`rns_engine` exposes the frozen evidence, reproducible replay paths, and the certified Series 1 execution library. The public package contains only stripped execution-source bodies and their provenance/certification data; the private G4 search, candidate-generation, grammar, learner, and optimization machinery is not shipped.

---

## What `rns_engine` is

`rns_engine` is a Python/C++ toolkit for exact arithmetic built around Residue Number Systems (RNS), signed range certificates, exact reconstruction, and low-precision hardware pipelines.

The package includes:

- exact unsigned and centered-signed integer arithmetic;
- no-wrap range certificates;
- fused weighted reconstruction of signed INT32 partials;
- exact shared-scale rational GEMM reference tools;
- local and global GEMM capacity planning;
- multi-rail exact reconstruction tools;
- Tensor-Core / CUDA integration contracts and fixtures;
- frozen G4 Series 1 integer and rational replay evidence;
- certified G4 Series 1 exact `g4_matmul()` on the current 1,024-shape T4 catalog.

The core RNS idea is simple: perform arithmetic across independent modular rails, then reconstruct the exact result when the declared range proves the answer is unique.

---

## Install

```bash
pip install rns_engine
```

Supported Python versions: **3.10–3.14**.

Prebuilt wheels are available for supported Linux x86-64, macOS Intel/Apple silicon, and Windows x64 Python targets. The G4 Series 1 GPU execution path itself is currently Linux + Tesla T4 specific, requires `nvcc` to compile/cache the verified execution sources on first use, and fails closed elsewhere. Standard Google Colab T4 runtimes provide the required CUDA toolchain.

---

## Quick start: exact integer arithmetic

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

You can remain in residue space across multiple operations and decode once:

```python
s1 = rns.add(*ex, *ey)
s2 = rns.mul(*s1, *ey)
s3 = rns.sub(*s2, *ex)
out = rns.decode(*s3)
```

For repeated work, use a session:

```python
session = rns.Session(cache_capacity=32)
encoded = session.encode(x)
result = session.mul(session.add(encoded, encoded), encoded)
out = session.decode(result)
```

---

## Signed exact arithmetic and no-wrap certificates

```python
import numpy as np
import rns_engine as rns

values = np.array([-5, 0, 7], dtype=np.int64)
rails = rns.encode_signed(values)
round_trip = rns.decode_signed(*rails)

assert np.array_equal(round_trip, values)
```

`encode_signed(...)` rejects values outside the supported centered range instead of silently wrapping them.

For a mathematical result to be uniquely recoverable as a signed value, you also need an independent bound proving:

```text
|result| < M / 2
```

Use a range certificate:

```python
certificate = rns.certify_signed_bound(max_abs_bound=1_000_000)
certificate.require_unique()
print(certificate.headroom)
```

For a signed dot product or one GEMM output entry:

```python
certificate = rns.certify_signed_dot_bound(
    inner_dimension=5120,
    left_abs_bound=127,
    right_abs_bound=127,
)
certificate.require_unique()
```

`decode_signed()` gives the centered representative. **It does not, by itself, prove that an unknown mathematical result did not wrap.** Use the range-certificate APIs when uniqueness matters.

---

## Exact rational GEMM reference path

The package also includes a correctness-first, hardware-independent shared-scale reference path for wider exact-rational experiments:

```python
import numpy as np
import rns_engine as rns

left = rns.SharedScaleMatrix(
    np.array([[2**40 + 3, -17], [29, 2**35 - 5]], dtype=object),
    scale=6,
)
right = rns.SharedScaleMatrix(
    np.array([[31, -(2**30) + 9], [2**33 + 1, 43]], dtype=object),
    scale=35,
)

receipt = rns.exact_shared_scale_gemm(
    left,
    right,
    config=rns.WideRNSConfig.balanced_seven_rail(),
    left_plane_count=8,
    right_plane_count=8,
)

assert receipt.exact_match
assert receipt.output_scale == 210
result = receipt.as_matrix()
```

This reference API and the G4 Series 1 T4 fast path serve different purposes: the reference path prioritizes general correctness machinery; `g4_matmul()` dispatches the frozen physically certified Series 1 implementations for its declared input/shape contract.

---

## Tensor-Core / INT32 partial reconstruction

For digit-plane, chunked-GEMM, and Tensor-Core pipelines, `rns_engine` can fuse signed INT32 partial outputs with exact positional weights:

```python
import numpy as np
import rns_engine as rns

partials = np.array(
    [
        [[1, -2], [3, -4]],
        [[5, 6], [-7, -8]],
        [[-9, 10], [11, -12]],
    ],
    dtype=np.int32,
)

receipt = rns.accumulate_weighted_int32(
    partials,
    weights=[1, -3, 8],
    require_unique=True,
)
exact = receipt.decode_signed()
```

---

## GEMM capacity planning

Two safety questions must remain separate:

1. Does each raw grouped coefficient fit the native signed accumulator?
2. Does the final weighted result fit uniquely inside the global RNS modulus product?

```python
import rns_engine as rns

local = rns.plan_grouped_coefficient_capacity(
    inner_dimension=5120,
    left_digit_abs_bounds=[127] * 8,
    right_digit_abs_bounds=[127] * 8,
    accumulator_bits=32,
)

global_plan = rns.plan_digit_plane_gemm_capacity(
    inner_dimension=5120,
    radix=128,
    left_digit_abs_bounds=[127] * 8,
    right_digit_abs_bounds=[127] * 8,
)

print(local.safe)
print(global_plan.current_unique)
```

See [`CAPACITY_PLANNING.md`](CAPACITY_PLANNING.md) for exact laws and limits.

---

## How the core four-rail engine works

The default engine decomposes values across four coprime rails:

```text
127
8191
65536
524287
```

Their product is:

```text
M = 35,742,890,181,197,824
```

Unsigned reconstruction lives in `[0, M)`. The centered signed interpretation is:

```text
[-17,871,445,090,598,912,
  17,871,445,090,598,911]
```

Arithmetic runs independently on the rails and reconstructs the canonical value with CRT/Garner-style reconstruction.

---

## API at a glance

### Core arithmetic

```text
encode / decode
encode_signed / decode_signed
add / sub / mul / div_ / fma
affine_repeat
Session / SignedSession
```

### Proof and range tools

```text
certify_signed_bound
certify_signed_dot_bound
certify_weighted_sum_bound
plan_signed_capacity
plan_weighted_sum_capacity
plan_digit_plane_gemm_capacity
plan_grouped_coefficient_capacity
```

### Exact GEMM / wide reference tools

```text
WideRNSConfig
SharedScaleMatrix
exact_integer_matmul
exact_shared_scale_gemm
decompose_signed_radix
reconstruct_signed_radix
grouped_plane_gemm
reconstruct_grouped_partials
accumulate_weighted_int32_wide
```

### G4 Series 1

```text
g4_results
g4_benchmark
g4_integer_benchmark
g4_rational_benchmark
g4_matmul
```

The four active G4 execution/benchmark APIs are:

```text
g4_benchmark           -> integers first, then rationals
g4_integer_benchmark   -> integers only
g4_rational_benchmark  -> rationals only
g4_matmul               -> use the certified exact G4 GEMM runtime
```

Use `rns.info()` for the installed engine's current range, moduli, version, and native capability summary.

---

## Scope and limitations

The public claims are intentionally narrow and reproducible:

- **G4 Series 1 is frozen to Tesla T4 / compute capability 7.5.** Results on another GPU are a different experiment.
- The reported benchmarks cover the declared **1,024 GEMM shapes**, not every possible matrix size.
- The **integer and rational results are separate benchmarks** and are never combined into one percentage.
- `g4_benchmark()` is a convenience runner that executes those two separate benchmarks in sequence.
- `g4_integer_benchmark()` and `g4_rational_benchmark()` replay frozen discoveries; they do not rerun G4's search process.
- Fresh timing classifications can move with clocks, driver state, and measurement noise. Mathematical exactness must not move.
- `g4_matmul()` currently supports signed-INT8 integer inputs or signed-INT8 shared-scale rational numerators on the certified 1,024-shape Series 1 catalog.
- Unsupported G4 shapes/representations fail closed rather than silently approximating.
- Exact signed reconstruction elsewhere in the library still requires an independently valid range bound; modular correctness alone does not prove no wrap.

### Division

Division requires the divisor to be invertible on every default rail:

- `b % 127 != 0`
- `b % 8191 != 0`
- `b` must be odd for modulus `65536`
- `b % 524287 != 0`

---

## Additional technical documentation

- [`G4_SERIES1_EVIDENCE.md`](G4_SERIES1_EVIDENCE.md) — frozen G4 evidence, replay protocol, and privacy boundary
- [`CAPACITY_PLANNING.md`](CAPACITY_PLANNING.md) — local/global exactness and accumulator capacity
- [`PRE_CUDA_READINESS.md`](PRE_CUDA_READINESS.md) — correctness oracle and CUDA integration contract
- [`CUDA_PARALLEL_LANES.md`](CUDA_PARALLEL_LANES.md) — parallel rail planning

---

## Build from source

```bash
git clone https://github.com/playfularchitect/rns_engine.git
cd rns_engine
pip install -e .
python -m pytest tests/ -v
```

Requirements:

- Python 3.10–3.14
- C++17 compiler
- build dependencies declared in `pyproject.toml`

The release workflow builds and tests supported wheels across Linux, macOS, and Windows.

---

## License

**AGPL-3.0-only**

If this software is used in a network service, the AGPL requires modified source to be made available to users. Commercial licensing for proprietary or closed-source use is available.

Inquiries: `ewesley541@gmail.com`

Copyright 2026 Evan Wesley
