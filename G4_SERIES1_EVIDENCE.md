# G4 Series 1 — Frozen Tesla T4 Evidence

G4 Series 1 is frozen. This public layer exposes the frozen headline benchmark evidence plus replay-only Tesla T4 runtimes.

Speedup ratios in this document are **NVIDIA time / G4 time**, so values above `1.0x` mean G4 is faster.

## Two separate frozen claims

### Exact integer clean sweep

On the frozen `T4GL-1024-v1` suite, the final decision record contains:

- 1,024 declared GEMM shapes
- 938 exact integer G4 wins
- 86 NVIDIA floating wins
- 0 statistical ties
- 0 errors
- 91.60% G4-win coverage
- exact replay passed before and after timing on all 1,024 shapes
- Tesla T4, compute capability 7.5
- resident-data GEMM kernel benchmark

Aggregate speed over **all 1,024 frozen shapes**, including NVIDIA-winning shapes:

- geometric mean: **1.235383x**
- median: **1.256704x**

Among the 938 G4-winning shapes:

- geometric mean: **1.311052x**
- median: **1.289235x**
- best frozen G4 win: **2.647061x**

Across the 86 NVIDIA-winning shapes, G4 retained **64.5959% of NVIDIA throughput on geometric average**. Equivalently, the execution-time ratio implied by that geometric mean is a **54.8086% G4 time penalty** on those losses.

That loss-side figure is exactly derivable from the frozen aggregate record because the suite contains only 938 G4 wins + 86 NVIDIA wins = 1,024 shapes with no ties/errors:

```text
all_shape_geomean^1024
    = g4_win_geomean^938 * nvidia_win_ratio_geomean^86
```

The floating landscape uses FP16-input cuBLASLt configurations; the retained accumulation mode may be FP16 or FP32 for a given shape. FP16 is the **speed baseline**, not the exactness baseline. Series 1 asks whether exact arithmetic can reach floating-point-class throughput; it does not claim to beat every possible NVIDIA integer GEMM configuration.

#### Signed-INT32 range contract

The frozen shape manifest has `K <= 4096`. For full-range signed-INT8 inputs, the largest possible product magnitude is `128 × 128 = 16,384`, so the conservative worst-case dot-product magnitude is:

```text
4096 × 128 × 128 = 67,108,864
```

That is safely inside signed INT32 range. Thus all 1,024 certified Series 1 shapes are safe for full-range signed-INT8 inputs under the published exact signed-INT32 output contract.

The historical integer archive preserves its actual measurement depth rather than pretending every shape used one block count:

- 827 shapes used 21 paired timing blocks
- 190 shapes used 31 paired timing blocks
- 7 shapes used 127 paired timing blocks

The current public integer replay uses a uniform 31-paired-block classification rule with 20,000 bootstrap resamples, a 95% confidence interval, a 1.002 promotion threshold, and at least 20 / 31 G4 block wins. Runtime or exactness failures fail closed.

### Dynamic exact-rational campaign

The separately frozen G416 campaign records:

- 1,024 evaluated shapes
- 870 certified exact-rational G4 wins
- 110 NVIDIA wins
- 41 statistical ties
- 3 final errors
- 84.96% certified exact-rational G4-win coverage
- 31 paired timing blocks for every certified winner
- actual non-integer inputs for every certified winner
- range proof passed for every certified winner
- FP16 value-set proof passed for every certified winner

Among the 870 certified G4-winning shapes:

- geometric mean: **1.417436x**
- median: **1.406441x**
- best certified G4 win: **2.978048x**
- slowest certified G4 win: **1.004345x**

The frozen public rational summary does **not** expose an all-1,024 speedup aggregate or a loss-side aggregate, so this document does not invent either one.

#### What the three final rational errors were

The three shapes left in final `ERROR` state were:

- `T4GL0140`
- `T4GL0879`
- `T4GL1020`

Each final row records the same post-measurement bookkeeping exception: `KeyError: 'forecast_debt_bits'`. These were not silent arithmetic mismatches. Each row had already produced 31 paired timings, used actual non-integer inputs, and recorded `range_proved=True` plus `fp16_value_set_proved=True`. The campaign conservatively retained them as `ERROR` rather than reclassifying them from partial evidence.

#### Shared-scale rational representation

`SharedScaleMatrix` represents one integer numerator matrix divided by one positive integer scale. In the public Series 1 API:

- numerator matrices must fit the signed-INT8 fast-input contract;
- scales are positive Python integers, so scale metadata is not limited by a fixed-width integer type;
- the output scale is exactly `left.scale * right.scale`;
- `reduce=True` optionally divides a common factor from the numerator matrix and scale;
- arbitrary per-element denominators are not part of the Series 1 fast-path contract.

**Do not conflate the two percentages.** `938/1024` is the frozen exact-integer clean-sweep result. `870/1024` is the separately certified dynamic exact-rational result.

## Benchmark timing boundary vs user-API wall time

The frozen **integer** headline is a resident-data GEMM kernel benchmark: GPU data movement, Python-call overhead, and one-shot CPU-array wrapper/runtime work are outside that timing boundary.

The frozen **rational** benchmark has its own documented exact-result timing boundary and includes the rational metadata/bookkeeping required to produce the exact rational result. It still is not a one-shot CPU-NumPy `g4_matmul()` wall-time claim.

The public `g4_matmul()` API has a different boundary. It accepts CPU NumPy arrays (or `SharedScaleMatrix` objects backed by CPU arrays) and returns a CPU NumPy-backed result. End-to-end `g4_matmul()` wall time therefore includes data movement and wrapper/runtime work that the resident-data integer benchmark excludes.

A benchmark that times PyTorch only after tensors are already resident on the GPU must be compared against the same resident-data boundary on the G4 side. Comparing that PyTorch kernel time directly with CPU→CPU `g4_matmul()` wall time mixes two different measurements.

Series 1 does **not** claim that one-shot CPU-NumPy→CPU-NumPy `g4_matmul()` wall time beats a pre-resident PyTorch GPU GEMM call.

## Python API

```python
import rns_engine as rns

# Frozen evidence; no GPU required.
rns.g4_results()
rns.g4_results("integer")
rns.g4_results("rational")

# Tesla T4 only: physical replay of frozen implementations.
rns.g4_benchmark("quick")
rns.g4_benchmark("standard")
rns.g4_benchmark("full")

# Or run either species independently.
rns.g4_integer_benchmark("full")
rns.g4_rational_benchmark("full")
```

`g4_results()` reports the frozen scorecards above.

`g4_benchmark()` is a convenience runner that physically replays **both** public Series 1 benchmarks on a Tesla T4: G4 INTEGERS first, then G4 RATIONALS. The scores remain separate and no combined win percentage is reported. It does not run G4 search.

`g4_integer_benchmark()` and `g4_rational_benchmark()` replay either benchmark independently.

## Replay-config provenance boundary

Most full rational configs are preserved directly by the frozen work artifacts. Where the frozen result row retained the winner identity but omitted physical fields such as padding or stage ID, the one-time private build enumerated every full frozen candidate matching that retained identity. The public replay selects only within that identity class before measurement and does not claim which omitted member was historical. No timing-based historical config guess is permitted.

For floating cuBLASLt baselines, Series 1 originally retained the opaque algorithm object returned by `cublasLtMatmulAlgoGetHeuristic`. The public replay therefore re-queries that same heuristic API on the frozen T4 descriptor/padding context and accepts only returned algorithms whose visible identity and workspace match the archived option. It does not attempt to reconstruct NVIDIA's opaque algorithm object from visible fields with `cublasLtMatmulAlgoInit`.

The resulting provenance counts are stored in the packaged runtime metadata and included in benchmark output.

## Packaged public evidence/runtime

The public package contains:

- `g4s1_public_summary.json` — the two frozen headline claims, full scorecard fields available from the archive, and provenance metadata.
- `g4s1_public_t4_runtime.json` — integer/runtime integrity, hardware boundary, source-family metadata, shape manifest, compile contract, and public-source hashes.
- rational replay runtime metadata and payload chunks used by the frozen rational replay path.
- public execution sources required for the Series 1 integer and `g4_matmul()` paths.

Every packaged execution/runtime component is SHA-256 integrity checked before use. If the packaged hashes do not match, the public path refuses to run.

The detailed G4 search machinery is not needed to reproduce the public replay result. The public package contains the frozen evidence and runtime required by `g4_results()`, `g4_benchmark()`, and the certified Series 1 `g4_matmul()` path.

## Claim boundary

Series 1 claims only the frozen Tesla T4 / compute capability 7.5 contract documented above. Results on another GPU are a different experiment. Future G4 generations are kept as separate Series rather than rewriting Series 1 evidence.
