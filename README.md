# rns_engine

Alpha prototype for **batched exact modular integer arithmetic** using a fixed 3-rail Residue Number System (RNS).

Current implementation:
- fixed moduli: `127`, `8191`, `65536`
- dynamic range: `[0, 68,174,282,752)`
- operations on batches of values encoded into residue rails
- AVX2 fast path on supported x86-64 CPUs
- scalar fallback on unsupported hardware

This package is a **prototype**, not a finished arbitrary-precision integer engine.

## What it currently does

`rns_engine` encodes `uint64` values into three residue arrays and performs arithmetic independently on each rail. Results can then be reconstructed with CRT / Garner decoding.

Supported operations:
- `encode`
- `decode`
- `add`
- `sub`
- `mul`
- `div_`
- `fma`

These operations are exact **modulo**:

```text
M = 127 × 8191 × 65536 = 68,174,282,752
