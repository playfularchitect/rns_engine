# Pre-CUDA Readiness Contract

Do not spend GPU credits until this repository's complete CPU-side readiness gate passes:

```bash
rns-pre-cuda
```

The command emits one JSON receipt and exits nonzero if any required condition fails.

## What must already be proven

The gate verifies all of the following without CUDA:

1. The eight-plane radix-128 target at `K=5120` needs a 126-bit signed modulus.
2. The existing four rails are short by 71 product bits.
3. Under a 31-bit-per-extra-rail ceiling, three added rails are mathematically necessary.
4. Both retained seven-rail profiles are pairwise coprime and close the capacity deficit:
   - smallest-product exponents `(11, 29, 31)`;
   - balanced exponents `(23, 24, 25)`.
5. Both profiles round-trip centered signed values, including their exact signed boundaries.
6. Wide rail addition and multiplication agree with Python big integers.
7. Signed radix-128 digit planes reconstruct their source integers exactly.
8. Grouped plane GEMMs reconstruct the direct Python-big-integer GEMM exactly.
9. Every grouped coefficient fits signed INT32 for the tested fixture.
10. Shared-scale rational GEMM preserves the exact numerator and multiplies scale metadata exactly.
11. The portable CUDA fixture survives a JSON round trip.
12. A backend must reproduce both:
    - grouped INT32 coefficient matrices;
    - weighted seven-rail residues.
13. Final centered reconstruction must equal the direct big-integer witness.

## What remains for the T4

Once the gate is green, the only intentionally missing body is the CUDA implementation of the backend contract:

```python
class ExactPipelineBackend:
    def grouped_partials(left_planes, right_planes):
        ...  # INT8 Tensor Core / cuBLASLt body

    def weighted_rails(grouped_partials, weights, moduli):
        ...  # CUDA modular weighting and seven-rail accumulation
```

The verifier, fixtures, scales, exact witnesses, range receipts, candidate rail profiles, and failure rules already exist.

## What a green gate does not prove

A green CPU gate does not prove:

- T4 Tensor Core execution;
- CUDA correctness;
- GPU reduction efficiency;
- FP16 parity or superiority;
- that either candidate rail profile is the fastest GPU profile;
- arbitrary per-element rational denominators.

Those claims require the T4 run.

## First T4 success condition

The first GPU milestone is not speed. It is exact equivalence:

```text
CUDA grouped partials == CPU grouped partials
CUDA seven rails       == CPU seven rails
CUDA decoded numerator == direct Python integer GEMM
CUDA output scale      == left scale * right scale
```

Only after that witness passes should timing begin.
