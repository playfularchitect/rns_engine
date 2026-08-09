# rns_engine

**Exact arithmetic without the usual speed-or-accuracy tradeoff.**

We tested two separate exact-arithmetic systems **directly against NVIDIA's own optimized cuBLASLt FP16-input GEMM implementation** on the same class of NVIDIA Tesla T4 GPU across **1,024 different matrix-multiplication shapes**.

The tests were run on **standard Tesla T4 hardware provided through Google Colab** — publicly available commodity cloud hardware, not a custom GPU, private lab rig, or specially modified machine.

| Exact arithmetic tested | Faster than NVIDIA | Win rate |
|---|---:|---:|
| **Exact integer GEMM** | **938 / 1,024 shapes** | **91.60%** |
| **Exact rational GEMM** | **870 / 1,024 shapes** | **84.96%** |

These are **two separate results**. The integer benchmark and rational benchmark were run and scored independently.

For the rational result, we also built a separate public replay runtime and reran all 1,024 shapes from scratch on a standard Google Colab Tesla T4:

- **1,024 / 1,024** exact rational calculations reproduced the correct mathematical result.
- **834 / 1,024** were faster than NVIDIA's FP16 implementation on the fresh replay.
- **114 / 1,024** were slower than NVIDIA's FP16 implementation.
- **76 / 1,024** were statistical ties.

In both cases, we squared our exact arithmetic directly against **NVIDIA's own optimized FP16 GEMM implementation on the same publicly available Tesla T4 hardware** — and won the speed comparison across the large majority of the tested matrix sizes, **without accepting the usual tradeoff between speed and accuracy. You can have both.**

> **Exact** describes correctness. **Win** describes speed. A shape can remain mathematically exact even when NVIDIA wins the timing comparison.

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

On a **Tesla T4 / compute capability 7.5**, physically rerun the exact-rational comparison:

```python
import rns_engine as rns

rns.g4_benchmark("quick")     # 24 shapes
rns.g4_benchmark("standard")  # 128 shapes
rns.g4_benchmark("full")      # all 1,024 shapes
```

`g4_benchmark()` is deliberately fail-closed. Series 1 is frozen to the Tesla T4; running on a different GPU would be a different benchmark.

The live benchmark is **replay-only**: it reruns the frozen Series 1 implementations rather than performing a new G4 search.

See [`G4_SERIES1_EVIDENCE.md`](G4_SERIES1_EVIDENCE.md) for the benchmark protocol, claim boundaries, and replay provenance.

---

## What is G4?

G4 is the autonomous optimization/research system that discovered the benchmarked implementations.

This repository exposes the **results and a reproducible replay runtime**.

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
- frozen G4 Series 1 benchmark evidence and Tesla T4 replay.

The core idea is simple: perform arithmetic across independent modular rails, then reconstruct the exact result when the declared range proves that the answer is unique.

---

## Install

```bash
pip install rns_engine
```

Supported Python versions: **3.10–3.14**.

Prebuilt wheels are available for supported Linux x86-64, macOS Intel/Apple silicon, and Windows x64 Python targets.

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

The conservative bound is:

```text
output bound
=
inner dimension × left bound × right bound + addend bound
```

`decode_signed()` gives the centered representative. **It does not, by itself, prove that an unknown mathematical result did not wrap.** Use the range-certificate APIs when uniqueness matters.

---

## Exact rational GEMM reference path

`rns_engine` includes shared-scale exact matrix objects and a correctness-first wide-RNS reference path:

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

The public G4 T4 benchmark is a **frozen replay artifact** for the Series 1 exact-rational discoveries; it is not a general-purpose CUDA backend for arbitrary `rns_engine` operations.

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

The conservative bound is:

```text
max_abs_bound
=
sum(abs(weight[t]) × max_abs(partial[t]))
```

The proof receipt retains the original arbitrary-precision weights while execution uses lawful residues modulo the RNS product.

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

See [`CAPACITY_PLANNING.md`](CAPACITY_PLANNING.md) for the exact laws and limits.

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

Unsigned reconstruction lives in:

```text
[0, M)
```

The centered signed interpretation is:

```text
[-17,871,445,090,598,912,
  17,871,445,090,598,911]
```

Arithmetic runs independently on the rails and reconstructs the canonical value with CRT/Garner-style reconstruction.

For wider exact GEMM work, the package also contains correctness-first seven-rail reference configurations and capacity-planning tools.

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

### CUDA / backend verification

```text
CudaGemmFixture
ExactPipelineBackend
CpuExactPipelineBackend
build_cuda_gemm_fixture
verify_backend
run_pre_cuda_readiness
```

### G4 evidence and reproduction

```text
g4_results
g4_benchmark
```

Use `rns.info()` for the installed engine's current range, moduli, version, and native capability summary.

---

## Scope and limitations

The benchmark claims in this repository are intentionally narrow and reproducible:

- **G4 Series 1 is a Tesla T4 benchmark.** Results on another GPU are a different experiment.
- The reported scores cover the declared **1,024 GEMM shapes**, not every possible matrix size.
- The **integer and rational results are separate benchmarks** and should not be combined into one percentage.
- `g4_benchmark()` replays frozen discoveries; it does not rerun G4's search process.
- Fresh timing classifications can move with clocks, driver state, and normal measurement noise. Mathematical exactness must not move.
- Exact signed reconstruction requires an independently valid range bound; modular correctness alone does not prove no wrap.
- The public Series 1 replay runtime is a benchmark artifact, not a general CUDA implementation of every `rns_engine` API.

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
- [`colab/T4_EXACT_GEMM_BRINGUP.ipynb`](colab/T4_EXACT_GEMM_BRINGUP.ipynb) — earlier CUDA bring-up fixture

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
