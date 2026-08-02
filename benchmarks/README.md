# Weighted INT32 baseline benchmark

This directory measures the current `accumulate_weighted_int32(...)` body before any fused C++ replacement is attempted.

The current path is:

```text
Python schedules each term
  -> native signed encode
  -> native scalar rail multiply
  -> native rail add
  -> optional native CRT decode
```

The benchmark reports:

- median and minimum accumulation time
- separate decode time
- partial values processed per second
- completed output values per second
- input memory size
- estimated Python-to-native call count
- signed uniqueness and remaining RNS headroom
- a NumPy `int64` control when the range certificate proves that comparison safe
- exact arbitrary-precision witnesses on sampled outputs

## Default sweep

```bash
python benchmarks/weighted_int32_baseline.py
```

Defaults:

- term counts: `1,2,4,8,16`
- output sizes: `1024,65536,1000000`
- partial magnitude: `127`
- weights: all `1`
- seven measured repeats after two warmups

## Radix-weighted planes

```bash
python benchmarks/weighted_int32_baseline.py \
  --terms 2,4,8 \
  --outputs 65536,1000000 \
  --weight-mode radix \
  --radix 128
```

Centered alternating radix weights:

```bash
python benchmarks/weighted_int32_baseline.py \
  --weight-mode centered-radix \
  --radix 128
```

## Save a machine-readable receipt

```bash
python benchmarks/weighted_int32_baseline.py \
  --json weighted_int32_baseline.json
```

The JSON includes environment, configuration, range certificates, timings, and throughput for every case.

## Decision rule for a fused kernel

Do not fuse merely because one kernel sounds cleaner. Compare the measured work:

```text
current cost
=
term scheduling
+ repeated encode cost
+ repeated weight-multiply cost
+ repeated rail-add cost
+ memory traffic
```

A fused C++ body earns its complexity only if the baseline shows that repeated dispatch or intermediate rail traffic is a material fraction of end-to-end time at the term counts and output sizes that WarpFrac actually needs.

The benchmark does not measure CUDA, Tensor Core GEMM, host-device transfer, or GPU reconstruction. Those require a separate GPU harness after the CPU bridge body is understood.
