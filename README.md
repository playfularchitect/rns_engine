# rns_engine
** This is licensed under AGPLv3. If you use it in a network service, you must make your modified source available to users. 
 Commercial Licensing
 If you need to use this software in a proprietary or closed-source product, commercial licenses are available. This allows you to bypass the AGPL requirements. For inquiries, please contact: ewesley541@gmail.com**


**Extremely fast and Exact integer arithmetic on a 2 core cpu**

No floating point. No approximation. Exact results modulo the engine's dynamic range.

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

## Core API

### Encoded rail API

* `rns.encode(x)` → `(r0, r1, r2, r3)`
* `rns.decode(r0, r1, r2, r3)` → `uint64[]`
* `rns.add(*ea, *eb)`
* `rns.sub(*ea, *eb)`
* `rns.mul(*ea, *eb)`
* `rns.div_(*ea, *eb)`
* `rns.fma(*ea, *eb, *ec)`
* `rns.op(*ea, *eb, code)` where `0=add 1=mul 2=sub 3=div`

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
* values outside `[0, M)` are reduced mod `M` during encode
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
* built wheel installed and executed successfully
* full test suite passed: **49 / 49**

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

* Python 3.10+
* C++17 compiler
* NumPy
* pybind11

---

## Introspection
```python
import rns_engine as rns

rns.info()

rns.M
rns.M0
rns.M1
rns.M2
rns.M3
rns.HAS_AVX2
```

---

## Current release

### v0.4.0rc1

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

**AGPL-3.0-only**  
Copyright 2026 Evan Wesley
