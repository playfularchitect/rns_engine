# rns_engine

**Fast, exact integer arithmetic on a four-rail Residue Number System.**

No floating point. No approximation. Arithmetic is exact modulo the engine's fixed dynamic range.

Recent Colab benchmark: https://colab.research.google.com/drive/18RIQSLf4Vf5xUnwEmDwYNPJDLSM_Stqa?usp=sharing

---

## What it does

`rns_engine` decomposes each integer across four coprime rails:

- `127`
- `8191`
- `65536`
- `524287`

Arithmetic runs independently on the rails, then Garner-style CRT reconstructs the canonical result.

```text
M = 127 × 8191 × 65536 × 524287
  = 35,742,890,181,197,824
```

Unsigned results live in:

```text
[0, M)
```

The centered signed interpretation lives in:

```text
[-M/2, M/2 - 1]
=
[-17,871,445,090,598,912, 17,871,445,090,598,911]
```

---

## Install

```bash
pip install rns_engine
```

Supported Python versions: **3.10–3.13**.

### AVX2 note

`HAS_AVX2` is a build-time property of the compiled extension, not runtime CPU detection. Supported x86-64 wheels can use AVX2 acceleration. macOS wheels intentionally avoid an external OpenMP dependency.

---

## Unsigned quick start

```python
import numpy as np
import rns_engine as rns

x = np.array([123456789, 999999999], dtype=np.uint64)
y = np.array([987654321, 111111111], dtype=np.uint64)

ex = rns.encode(x)
ey = rns.encode(y)

out_add = rns.decode(*rns.add(*ex, *ey))
out_mul = rns.decode(*rns.mul(*ex, *ey))

# Stay in residue space across several exact operations and decode once.
s1 = rns.add(*ex, *ey)
s2 = rns.mul(*s1, *ey)
s3 = rns.sub(*s2, *ex)
out = rns.decode(*s3)
```

---

## Session API

```python
import numpy as np
import rns_engine as rns

session = rns.Session(cache_capacity=32)
x = np.array([1, 2, 3, 4], dtype=np.uint64)

encoded = session.encode(x)
result = session.mul(session.add(encoded, encoded), encoded)
out = session.decode(result)

one_step = session.one_shot_affine(
    x,
    multiplier=1_000_003,
    addend=7,
)

hot_loop = session.hot_loop_affine(
    x,
    multiplier=1_000_003,
    addend=7,
    iterations=1000,
)
```

---

## Centered signed arithmetic

```python
import numpy as np
import rns_engine as rns

values = np.array([-5, 0, 7], dtype=np.int64)
rails = rns.encode_signed(values)
round_trip = rns.decode_signed(*rails)

assert np.array_equal(round_trip, values)
```

`encode_signed(...)` accepts only integers in `[-M/2, M/2 - 1]`. It rejects out-of-range inputs instead of silently wrapping them modulo `M`.

`decode_signed(...)` returns the canonical centered representative. That representation alone does **not** prove that an unknown mathematical result did not wrap. Unique signed reconstruction requires independent range evidence:

```text
|result| < M/2
```

```python
certificate = rns.certify_signed_bound(max_abs_bound=1_000_000)
certificate.require_unique()

print(certificate.headroom)
print(certificate.minimum_required_modulus)
```

For a signed dot product—or one output entry of a GEMM—the engine can construct the conservative exact bound directly:

```python
certificate = rns.certify_signed_dot_bound(
    inner_dimension=5120,
    left_abs_bound=127,
    right_abs_bound=127,
)

assert certificate.max_abs_bound == 82_580_480
assert certificate.unique
certificate.require_unique()
```

The law is:

```text
output bound
=
inner_dimension × left bound × right bound + addend bound
```

`SignedSession` exposes the existing rail arithmetic with signed external values:

```python
session = rns.SignedSession()
a = session.encode_signed([-5, 7])
b = session.encode_signed([3, -10])
out = session.decode_signed(session.add(a, b))
```

---

## Weighted INT32 partial accumulation

Version 0.7 provides the fused direct bridge for digit-plane, chunked-GEMM, and Tensor Core pipelines.

Give the engine signed INT32 partial outputs with shape:

```text
(terms, *output_shape)
```

and one exact integer positional weight per term:

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
print(receipt.certificate.max_abs_bound)
print(receipt.certificate.headroom)
```

For partials `P[t]` and full integer weights `w[t]`, the receipt uses the exact conservative law:

```text
max_abs_bound
=
sum(abs(w[t]) × max_abs(P[t]))
```

Important details:

- The original arbitrary-precision Python weights are retained in the proof receipt.
- Only the execution weights are reduced modulo `M`, which is lawful for rail arithmetic.
- Negative weights and weights wider than `uint64` are supported.
- `INT32_MIN` is measured with its correct magnitude, `2,147,483,648`.
- `decode_signed()` requires a unique range certificate by default.
- `decode_modular()` always returns the canonical residue in `[0, M)`.
- Empty term sets produce an exact zero result.

The current implementation uses one fused native `_weighted` call to read signed INT32 partials, collect exact per-term magnitude bounds, apply positional weights modulo `M`, and accumulate all four rails. The pre-fusion staged encode → scale → add body remains available internally as an exact reference for A/B tests. Local AVX2 diagnostics measured roughly 2.1×–2.9× speedup at one million outputs and larger gains on tiny many-term cases; those figures are hardware-specific CPU evidence, not CUDA, Tensor Core, or universal performance claims.

A precomputed receipt can also be created without materializing partial arrays:

```python
certificate = rns.certify_weighted_sum_bound(
    weights=[1, 128, 128**2],
    term_abs_bounds=[100, 200, 300],
)
certificate.require_unique()
```

---

## Core API

### Rail operations

- `encode(x)` / `decode(r0, r1, r2, r3)`
- `encode_signed(x)` / `decode_signed(r0, r1, r2, r3)`
- `add(...)`, `sub(...)`, `mul(...)`, `div_(...)`, `fma(...)`
- `op(..., code)` where `0=add`, `1=mul`, `2=sub`, `3=div`

### Signed range receipts

- `certify_signed_bound(max_abs_bound)`
- `certify_signed_dot_bound(inner_dimension, left_abs_bound, right_abs_bound, addend_abs_bound=0)`
- `SignedRangeCertificate.unique`
- `SignedRangeCertificate.headroom`
- `SignedRangeCertificate.minimum_required_modulus`
- `SignedRangeCertificate.require_unique()`

### Weighted partial reconstruction

- `accumulate_weighted_int32(partials, weights, require_unique=False)`
- `certify_weighted_sum_bound(weights, term_abs_bounds, addend_abs_bound=0)`
- `WeightedInt32Result.decode_signed(require_unique=True)`
- `WeightedInt32Result.decode_modular()`

### Scalar-broadcast encoded operations

- `mul_u64(...)`
- `fma_u64(...)`
- `affine_repeat_u64(...)`

### Fused raw uint64 operations

- `add_u64_io(...)`
- `sub_u64_io(...)`
- `mul_u64_io(...)`
- `fma_u64_io(...)`
- `affine_repeat_u64_io(...)`

OpenMP forms use the `_omp` suffix. Auto-dispatch forms use the `_auto` suffix.

### High-level objects

- `Session`
- `SignedSession`
- `SessionCache`
- `EncodedArray`
- `WeightedInt32Result`

---

## Division constraint

Division requires the divisor to be invertible on every rail:

- `b % 127 != 0`
- `b % 8191 != 0`
- `b` is odd for modulus `65536`
- `b % 524287 != 0`

---

## Data model

- Unsigned `encode(...)` treats inputs as `uint64` and reduces values modulo `M`.
- `encode_signed(...)` accepts only integers in the centered interval and refuses silent wrapping.
- `decode_signed(...)` returns the canonical centered `int64` representative.
- `accumulate_weighted_int32(...)` requires an array whose dtype is exactly `int32`.
- Encoded rails use:
  - `r0`: `uint16`
  - `r1`: `uint16`
  - `r2`: `uint16`
  - `r3`: `uint32`
- `EncodedArray` stores four read-only rails.

---

## Existing verified benchmark

Verified on Google Colab Linux x86-64 with `AVX2=True`, two OpenMP processors, and two OpenMP threads. Workloads used the installed wheel.

Median throughput over five runs on 1,000,000 `uint64` values:

### Fused single-step affine

Workload: `fma_u64_io(x, 1_000_003, 7)`

- `fma_u64_io`: **47.8 million values/sec**
- `fma_u64_io_omp` with one thread: **80.7 million values/sec**
- `fma_u64_io_omp` with two threads: **84.6 million values/sec**

### Repeated affine loop

Workload: `affine_repeat_u64_io(x, 1_000_003, 7, iterations=1000)`

- scalar fused path: **61.19 billion ops/sec**
- OpenMP with one thread: **82.80 billion ops/sec**
- OpenMP with two threads: **94.86 billion ops/sec**

These measurements cover the existing affine kernels, not the new weighted INT32 orchestration path.

---

## Build from source

```bash
git clone https://github.com/playfularchitect/rns_engine.git
cd rns_engine
pip install -e .
python -m pytest tests/ -v
```

Requirements:

- Python 3.10–3.13
- C++17 compiler
- Access to the build dependencies declared in `pyproject.toml`

The release workflow compiles and tests the package on Linux, macOS, and Windows, then installs and tests every produced wheel before publication.

---

## Introspection

```python
import rns_engine as rns

rns.info()

rns.M
rns.HALF_M
rns.SIGNED_MIN
rns.SIGNED_MAX
rns.M0
rns.M1
rns.M2
rns.M3
rns.HAS_AVX2
```

---

## Current release

### v0.7.0

- fused native weighted signed INT32 partial accumulation
- one compiled pass for magnitude collection, positional weighting, and four-rail accumulation
- staged reference body retained for exact fused-versus-staged A/B benchmarking
- exact arbitrary-precision positional weights
- per-term magnitude receipts and strict no-wrap certification
- canonical modular and uniqueness-gated signed decode paths
- canonical centered signed encoding and decoding
- exact dot-product and GEMM output-bound receipts
- `SignedSession` high-level API
- four-rail native engine (`127 × 8191 × 65536 × 524287`)
- AVX2, OpenMP, and auto-dispatch kernel families
- release artifacts validated on Linux, macOS, and Windows for Python 3.10–3.13

---

## License

**AGPL-3.0-only**

If this software is used in a network service, the AGPL requires modified source to be made available to users. Commercial licensing for proprietary or closed-source use is available. Inquiries: `ewesley541@gmail.com`

Copyright 2026 Evan Wesley
