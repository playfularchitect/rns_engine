# G4 Series 1 — Frozen Tesla T4 Evidence

G4 Series 1 is frozen. This public layer exposes sanitized benchmark evidence plus a replay-only Tesla T4 runtime. It does **not** ship the private G4 optimizer, ACTG genomes, private search/learning state, generated grammar/profile files, private recovery machinery, or CUDA source.

## Two separate frozen claims

### Exact integer clean sweep

On the frozen `T4GL-1024-v1` suite, the final row-level decision ledger records:

- 1,024 declared GEMM shapes
- 938 exact integer wins
- 86 NVIDIA floating wins
- 91.60% exact-win coverage
- exact replay passed before and after timing on all 1,024 shapes
- Tesla T4, compute capability 7.5

The floating landscape uses FP16-input cuBLASLt configurations; the retained accumulation mode may be FP16 or FP32 for a given shape. This is a resident-data kernel benchmark.

### Dynamic exact-rational campaign

The later frozen G416 campaign records:

- 1,024 evaluated shapes
- 870 certified exact-rational wins
- 154 unresolved shapes at archive freeze
- 84.96% certified exact-rational coverage
- 31 paired timing blocks for every certified winner
- actual non-integer inputs for every certified winner
- range proof passed for every certified winner
- FP16 value-set proof passed for every certified winner

**Do not conflate the two percentages.** `938/1024` is the frozen exact-integer clean-sweep result. `870/1024` is the separately certified dynamic exact-rational result.

## Python API

```python
import rns_engine as rns

rns.g4_results()
rns.g4_results("integer")
rns.g4_results("rational")

# Tesla T4 only
rns.g4_benchmark("quick")
rns.g4_benchmark("standard")
rns.g4_benchmark("full")
```

`g4_results()` reads the frozen sanitized ledgers. `g4_benchmark()` physically replays the **dynamic exact-rational** comparison on a Tesla T4 and re-applies the frozen 31-paired-block certification rule. It does not run G4 search.

## Replay-config provenance boundary

Most full rational configs are preserved directly by the frozen work artifacts. Where the frozen result row retained the winner identity but omitted physical fields such as padding or stage ID, the one-time private build enumerates every full frozen candidate matching that retained identity. The public replay selects only within that identity class before measurement and does not claim which omitted member was historical. No timing-based historical config guess is permitted.

For floating cuBLASLt baselines, Series 1 originally retained the opaque algorithm object returned by `cublasLtMatmulAlgoGetHeuristic`. The public replay therefore re-queries that same heuristic API on the frozen T4 descriptor/padding context and accepts only returned algorithms whose visible identity and workspace match the archived option. It does not attempt to reconstruct NVIDIA's opaque algorithm object from visible fields with `cublasLtMatmulAlgoInit`.

The resulting provenance counts are stored in `g4s1_t4_runtime.json` and included in benchmark output.

## Public files

- `g4s1_public_summary.json` — summary, claim labels, privacy boundary, and provenance hashes
- `g4s1_integer_fp16_input_results.csv` — 1,024 sanitized integer clean-sweep rows
- `g4s1_dynamic_exact_rational_results.csv` — 1,024 sanitized rational campaign rows
- `g4s1_t4_replay.gz.b64` — packaged compressed replay payload; it decodes byte-for-byte to the stripped Tesla T4 executable that passed validation
- `g4s1_t4_runtime.json` — runtime integrity, build boundary, and config-provenance metadata
- `G4S1_SHA256SUMS` — hashes for packaged public data/runtime files

No internal optimizer identifiers, private search metadata, source-code lineage, or private G4 implementation state is present in the sanitized ledgers.
