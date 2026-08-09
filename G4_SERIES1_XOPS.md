# G4 Series 1 XOPS / G4OPS

The public benchmark output is self-defining, but this note freezes the terminology used by `rns_engine` 0.12.0 and later Series 1 reports.

- **XOP** — one mathematically exact arithmetic operation.
- **XOPS** — exact arithmetic operations per second.
- **G4OPS** — XOPS delivered by a G4 implementation.

For GEMM, the accounting rule is the conventional `2*M*N*K` operation count used for FLOPS comparisons. A shape that fails exactness receives zero XOP credit.

For exact-rational Series 1 reporting, headline G4OPS uses the end-to-end exact-result timing boundary, including the rational metadata/bookkeeping required to produce the exact rational result. Kernel-only G4OPS may also be reported separately.

Integer and rational benchmark scores remain separate. `g4_benchmark()` runs them sequentially only as a convenience runner.
