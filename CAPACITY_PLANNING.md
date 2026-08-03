# Exact GEMM Capacity Planning

`rns_engine` keeps two different safety questions separate:

1. **Local accumulator safety:** does each raw grouped digit-plane coefficient fit the native signed accumulator, such as INT32?
2. **Global reconstruction safety:** after positional weights are applied, does the final wide result fit uniquely inside the RNS modulus product?

A workload can pass the first test and fail the second. Version 0.8 provides exact integer-only receipts for both.

## Local grouped-coefficient safety

For positional digit planes,

```text
A = sum(A_i * radix**i)
B = sum(B_j * radix**j)
```

raw coefficient `k` is:

```text
R_k = sum_{i+j=k} GEMM(A_i, B_j)
```

If the GEMM inner dimension is `K`, and plane entries obey magnitude bounds `L_i` and `R_j`, then:

```text
abs(R_k) <= sum_{i+j=k} K * L_i * R_j
```

Use:

```python
import rns_engine as rns

local = rns.plan_grouped_coefficient_capacity(
    inner_dimension=5120,
    left_digit_abs_bounds=[127] * 8,
    right_digit_abs_bounds=[127] * 8,
    accumulator_bits=32,
)

print(local.coefficient_pair_counts)
print(local.coefficient_abs_bounds)
print(local.max_abs_bound)
print(local.minimum_signed_accumulator_bits)
local.require_safe()
```

For eight full-magnitude INT8 planes at `K=5120`:

```text
largest grouped coefficient = 660,643,840
minimum signed width        = 31 bits
INT32 status                = safe
```

This proves the grouped Tensor Core or native INT32 outputs can remain exact locally.

## Global RNS reconstruction safety

The reconstructed operands obey:

```text
left_value_bound  = sum(L_i * radix**i)
right_value_bound = sum(R_j * radix**j)
```

One final GEMM output obeys:

```text
abs(C) <= K * left_value_bound * right_value_bound + addend_bound
```

A symmetric signed interval `[-bound, bound]` requires modulus:

```text
minimum_modulus = 2 * bound + 1
```

Use:

```python
wide = rns.plan_digit_plane_gemm_capacity(
    inner_dimension=5120,
    radix=128,
    left_digit_abs_bounds=[127] * 8,
    right_digit_abs_bounds=[127] * 8,
)

print(wide.max_abs_bound)
print(wide.capacity.minimum_required_modulus)
print(wide.capacity.required_modulus_bits)
print(wide.capacity.additional_bits_required)
print(wide.current_unique)
```

For eight full radix-128 planes per operand:

```text
operand magnitude bound     = 128**8 - 1
required signed modulus     = 126 bits
current additional deficit  = 71 product bits
current four-rail result    = modular-only in the worst case
```

So the same workload is locally INT32-safe but not globally unique in the current four-rail RNS range.

## Proposed additional rails

Candidate CRT moduli can be tested explicitly:

```python
base = rns.plan_digit_plane_gemm_capacity(
    5120,
    128,
    [127] * 8,
    [127] * 8,
)

candidate = base.capacity.minimum_single_coprime_modulus

expanded = rns.plan_digit_plane_gemm_capacity(
    5120,
    128,
    [127] * 8,
    [127] * 8,
    additional_moduli=[candidate],
)

expanded.require_unique()
```

Every proposed modulus must be greater than one and coprime with the existing modulus product and every earlier candidate.

`minimum_single_coprime_modulus` answers only the mathematical capacity question. It does **not** claim that the returned modulus is fast to reduce, appropriate for a GPU, or the best hardware rail. Reduction cost, storage width, vectorization, and reconstruction cost must be benchmarked separately.

## General weighted-sum planning

For arbitrary exact weighted partial bounds:

```python
plan = rns.plan_weighted_sum_capacity(
    weights=[1, 128, 128**2],
    term_abs_bounds=[100, 200, 300],
    addend_abs_bound=7,
)

print(plan.max_abs_bound)
print(plan.current_unique)
print(plan.modulus_shortfall)
```

For a known result bound directly:

```python
plan = rns.plan_signed_capacity(max_abs_bound=10**30)
print(plan.minimum_required_modulus)
print(plan.additional_bits_required)
```

## What these receipts prove

They prove conservative worst-case capacity from the supplied exact integer bounds. They can certify:

- grouped coefficient safety for a chosen signed accumulator width;
- the exact final-result magnitude bound;
- whether the current RNS product gives unique signed reconstruction;
- how much modulus product is missing;
- whether explicit proposed CRT rails close that deficit lawfully.

They do not prove:

- that the supplied input bounds describe a real dataset tightly;
- that a proposed modulus is computationally efficient;
- GPU, CUDA, or Tensor Core speed;
- that worst-case capacity is necessary for a narrower application distribution;
- that adding rails is cheaper than chunking, rescaling, or changing the digit representation.
