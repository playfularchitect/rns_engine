# G4 Series 1 supported shapes

This folder is the human-facing catalog for the frozen **G4 Series 1** GEMM shape contract.

Series 1 supports exactly **1,024 `(M, N, K)` shapes** on NVIDIA Tesla T4. Unsupported shapes fail closed; they do not silently fall back to approximate floating point.

Files:

- `shapes.json` — exact machine-readable mirror of the frozen Series 1 shape manifest.
- `shapes.csv` — the same 1,024 shapes in a simple spreadsheet/text-friendly form.

The source of truth remains the cryptographically pinned runtime member `g4s1_matmul_shapes.json` used by `rns_engine` itself. Its certified SHA-256 is:

```text
ac612da046601fa0f2a6e23f3ddcf58d5d91699cef3a9263a5a00118ccbb2873
```

The public mirror in this folder must match that source exactly. Future Series should get their own sibling folder rather than modifying the frozen Series 1 catalog.
