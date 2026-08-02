# rns_engine



**Extremely fast and Exact integer arithmetic on a 2 core cpu**

No floating point. No approximation. Exact results modulo the engine's dynamic range.


Link to recent benchmark on Colab: https://colab.research.google.com/drive/18RIQSLf4Vf5xUnwEmDwYNPJDLSM_Stqa?usp=sharing
---

## What it does

Standard Python integers are exact but slow. NumPy arrays are fast, but fixed-width integer arithmetic can overflow and float-based pipelines drift. `rns_engine` gives you ** extremely fast and exact modular arithmetic within a fixed dynamic range** by decomposing values across four coprime rails:

- `127`
- `8191`
- `65536`
- `524287`

Arithmetic is performed independently on each rail, then reconstructed with Garner-style CRT.

**Dynamic range:** `[0, 35,742,890,181,197,824)`
```text
127 × 8191 × 65536 × 524287 = 35,742,890,181,197,824
```

---

## Install
```bash
pip install rns_engine
```

### AVX2 note

`HAS_AVX2` is a build-time property of the compiled extension, not runtime CPU detection. On supported x86_64 builds, the core can use AVX2 acceleration.

---

## Quick start
```python
import numpy as np
import rns_engine as rns

a = np.array([123456789, 999999999], dtype=np.uint64)
b = np.array([987654321, 111111111], dtype=np.uint64)

ea = rns.encode(a)
eb = rns.encode(b)

out_add = rns.decode(*rns.add(*ea, *eb))
out_mul = rns.decode(*rns.mul(*ea, *eb))

# Chain multiple exact operations in residue space, decode once at the end
s1 = rns.add(*ea, *eb)
s2 = rns.mul(*s1, *eb)
s3 = rns.sub(*s2, *ea)
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

# Single exact affine step
one = s.one_shot_affine(x, multiplier=1_000_003, addend=7)

# Repeated exact affine loop: stay in residue space, decode once
hot = s.hot_loop_affine(x, multiplier=1_000_003, addend=7, iterations=1000)
```

---

## Centered signed arithmetic and range certificates

The native core computes exact residues modulo `M`. Version 0.5 adds a canonical signed view over:

```text
[-M/2, M/2 - 1]
=
[-17,871,445,090,598,912, 17,871,445,090,598,911]
```

```python
import numpy as np
import rns_engine as rns

values = np.array([-5, 0, 7], dtype=np.int64)
rails = rns.encode_signed(values)
round_trip = rns.decode_signed(*rails)

assert np.array_equal(round_trip, values)
```

`encode_signed(...)` rejects values outside the centered interval instead of silently wrapping them modulo `M`.

A centered decoding is a canonical interpretation of a residue. It is **not by itself proof that an unknown mathematical result did not wrap**. Unique signed reconstruction requires an independent strict magnitude bound:

```text
|result| < M/2
```

Use a range certificate before claiming that a residue is the unique signed integer result:

```python
certificate = rns.certify_signed_bound(max_abs_bound=1_000_000)
certificate.require_unique()

print(certificate.headroom)
print(certificate.minimum_required_modulus)
```

For a signed dot product or one GEMM output entry, the engine can build the worst-case receipt directly:

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

This uses the exact conservative law:

```text
output bound
=
inner_dimension × left bound × right bound + addend bound
```

`SignedSession` exposes the existing residue arithmetic with signed external values:

```python
s = rns.SignedSession()
a = s.encode_signed([-5, 7])
b = s.encode_signed([3, -10])
out = s.decode_signed(s.add(a, b))
```

---

## Core API

### Encoded rail API

* `rns.encode(x)` → `(r0, r1, r2, r3)`
* `rns.decode(r0, r1, r2, r3)` → `uint64[]`
* `rns.encode_signed(x)` → strict centered signed encoding
* `rns.decode_signed(r0, r1, r2, r3)` → canonical `int64[]`
* `rns.add(*ea, *eb)`
* `rns.sub(*ea, *eb)`
* `rns.mul(*ea, *eb)`
* `rns.div_(*ea, *eb)`
* `rns.fma(*ea, *eb, *ec)`
* `rns.op(*ea, *eb, code)` where `0=add 1=mul 2=sub 3=div`

### Range-certificate API

* `rns.certify_signed_bound(max_abs_bound)`
* `rns.certify_signed_dot_bound(inner_dimension, left_abs_bound, right_abs_bound, addend_abs_bound=0)`
* `SignedRangeCertificate.unique`
* `SignedRangeCertificate.headroom`
* `SignedRangeCertificate.minimum_required_modulus`
* `SignedRangeCertificate.require_unique()`

### Scalar-broadcast encoded API

These avoid materializing full constant arrays in Python:

* `rns.mul_u64(*ea, multiplier)`
* `rns.fma_u64(*ea, multiplier, addend)`
* `rns.affine_repeat_u64(*ea, multiplier, addend, iterations)`

### Raw fused uint64 API

These perform encode → exact op → decode in one native call:

* `rns.add_u64_io(x, addend)`
* `rns.sub_u64_io(x, subtrahend)`
* `rns.mul_u64_io(x, multiplier)`
* `rns.fma_u64_io(x, multiplier, addend)`
* `rns.affine_repeat_u64_io(x, multiplier, addend, iterations)`

OpenMP variants:

* `rns.add_u64_io_omp(...)`
* `rns.sub_u64_io_omp(...)`
* `rns.mul_u64_io_omp(...)`
* `rns.fma_u64_io_omp(...)`
* `rns.affine_repeat_u64_io_omp(...)`

Auto-dispatch variants:

* `rns.add_u64_auto(...)`
* `rns.sub_u64_auto(...)`
* `rns.mul_u64_auto(...)`
* `rns.fma_u64_auto(...)`
* `rns.affine_repeat_u64_auto(...)`

### High-level API

* `rns.Session`
* `rns.SignedSession`
* `rns.SessionCache`
* `rns.EncodedArray`

---

## Division constraint

Division requires the divisor to be invertible on **all four rails**:

* `b % 127 != 0`
* `b % 8191 != 0`
* `b` must be odd (for mod `65536`)
* `b % 524287 != 0`

A safe sanitizer looks like this:
```python
import numpy as np
import rns_engine as rns

M = int(rns.M)

def make_invertible_divisor(b):
    b = np.asarray(b, dtype=np.uint64) % np.uint64(M)
    b = np.where((b & np.uint64(1)) == 0, b + np.uint64(1), b)

    bad = (
        (b % np.uint64(127) == 0)
        | (b % np.uint64(8191) == 0)
        | (b % np.uint64(524287) == 0)
        | ((b & np.uint64(1)) == 0)
    )

    while np.any(bad):
        b = np.where(bad, (b + np.uint64(2)) % np.uint64(M), b)
        bad = (
            (b % np.uint64(127) == 0)
            | (b % np.uint64(8191) == 0)
            | (b % np.uint64(524287) == 0)
            | ((b & np.uint64(1)) == 0)
        )

    return b.astype(np.uint64)
```

---

## Data model

* input arrays to `encode(...)` are treated as `uint64`
* values outside `[0, M)` are reduced mod `M` during unsigned encode
* `encode_signed(...)` accepts only integers in `[-M/2, M/2 - 1]` and refuses silent wrapping
* `decode_signed(...)` returns the canonical centered `int64` view
* rails are returned as:
  * `r0`: `uint16`
  * `r1`: `uint16`
  * `r2`: `uint16`
  * `r3`: `uint32`
* high-level `EncodedArray` objects store four read-only rails

---

## Performance

### Verified benchmark

Verified on **Google Colab Linux x86_64**, with:

* `AVX2 = True`
* `omp_num_procs = 2`
* `omp_max_threads = 2`

Workloads were run against the **installed wheel**, not just an editable import.

Median throughput over **5 runs** on **1,000,000 uint64 values**:

#### Fused single-step affine

Workload: `fma_u64_io(x, 1_000_003, 7)`

* `fused fma_u64_io`: **47.8 million values/sec**
* `fma_u64_io_omp (1 thread)`: **80.7 million values/sec**
* `fma_u64_io_omp (2 threads)`: **84.6 million values/sec**

#### Repeated affine loop

Workload: `affine_repeat_u64_io(x, 1_000_003, 7, iterations=1000)`

* `affine_repeat_u64_io`: **61.19 billion ops/sec**
* `affine_repeat_u64_io_omp (1 thread)`: **82.80 billion ops/sec**
* `affine_repeat_u64_io_omp (2 threads)`: **94.86 billion ops/sec**

### Verification status

* correctness sanity checks passed
* the full suite is exercised on Linux, macOS, and Windows across Python 3.10–3.13
* built wheels are installed and tested before release

---

## Why RNS?

In a Residue Number System, addition and multiplication happen independently on each rail. There is no cross-rail carry propagation. That makes RNS attractive for:

* exact modular arithmetic within a fixed dynamic range
* SIMD-friendly kernels
* repeated arithmetic pipelines where decode can be delayed
* parallel execution

---

## How it works

### Encode
```text
x -> (x mod 127, x mod 8191, x mod 65536, x mod 524287)
```

### Operate

Each rail is processed independently.

### Decode

Garner-style CRT reconstruction combines the four residues back into a `uint64` value modulo `M`.

For the Mersenne moduli (`127 = 2^7 - 1`, `8191 = 2^13 - 1`, `524287 = 2^19 - 1`), the core uses fold-based reduction instead of general division.

---

## Building from source
```bash
git clone https://github.com/playfularchitect/rns_engine.git
cd rns_engine
pip install -e .
pytest tests/ -v
```

Requirements:

* Python 3.10–3.13
* C++17 compiler
* An internet-accessible Python package index containing the build dependencies

`pip` installs NumPy and pybind11 automatically from `pyproject.toml`. In an offline or restricted environment, those packages must already be available on the configured package index or installed locally before building.

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

### v0.5.0

* canonical centered signed encoding and decoding
* strict signed input validation with no silent modular wrapping
* signed uniqueness certificates using the law `|result| < M/2`
* exact dot-product and GEMM output-bound receipts
* `SignedSession` high-level API
* 4-rail engine (`127 × 8191 × 65536 × 524287`)
* AVX2-accelerated encoded kernels
* fused `fma(...)`
* scalar-broadcast encoded APIs: `mul_u64`, `fma_u64`, `affine_repeat_u64`
* fused raw uint64 APIs: `*_u64_io`
* OpenMP fused raw APIs: `*_u64_io_omp`
* auto-dispatch raw APIs: `*_u64_auto`
* high-level `Session`, `SessionCache`, and `EncodedArray`

---

## License
** This is licensed under AGPLv3. If you use it in a network service, you must make your modified source available to users. 
 Commercial Licensing
 If you need to use this software in a proprietary or closed-source product, commercial licenses are available. This allows you to bypass the AGPL requirements. For inquiries, please contact: ewesley541@gmail.com**


**AGPL-3.0-only**  
Copyright 2026 Evan Wesley
