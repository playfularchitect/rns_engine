# G4 Series 1 supported shapes

This folder is the human-facing catalog for the frozen **G4 Series 1** GEMM shape contract.

Series 1 supports exactly **1,024 `(M, N, K)` shapes** on NVIDIA Tesla T4. Unsupported shapes fail closed; they do not silently fall back to approximate floating point.

Files:

- [`shapes.csv`](shapes.csv) — the easiest list to scan: `shape_id,m,n,k`, one supported shape per row.
- [`shapes.json`](shapes.json) — the exact machine-readable mirror of the frozen Series 1 runtime manifest, including the retained Series 1 metadata for each shape.

The source of truth remains the cryptographically pinned runtime member `g4s1_matmul_shapes.json` used by `rns_engine` itself. Its certified SHA-256 is:

```text
ac612da046601fa0f2a6e23f3ddcf58d5d91699cef3a9263a5a00118ccbb2873
```

The checked-in `shapes.json` was exported from that pinned runtime member and verified byte-for-byte against this SHA-256 before publication. It contains **1,024 unique `(M, N, K)` shapes**.

Future Series should get their own sibling folder rather than modifying the frozen Series 1 catalog.
