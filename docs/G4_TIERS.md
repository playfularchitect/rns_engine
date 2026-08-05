# G4 exact range tiers

`G416`, `G432`, and `G464` are the public numerical contracts for RNS Engine.
The number communicates the familiar IEEE binary floating-point range tier. It
does **not** expose the physical representation used underneath.

| G4 tier | Matched floating opponent | Maximum finite magnitude | Minimum positive magnitude |
|---|---:|---:|---:|
| `G416` | FP16 | `65,504` | `2^-24` |
| `G432` | FP32 | `(2^24 - 1) * 2^104` | `2^-149` |
| `G464` | FP64 | `(2^53 - 1) * 2^971` | `2^-1074` |
| `G4X` | arbitrary precision | unbounded | unbounded |

Within a tier's range, values and results are exact. They are not restricted to
the discrete values representable by the matching floating type. For example,
`G416("1/3")` is an exact one third while still carrying the G416 range
contract.

## Strict by default

Strict behavior is mandatory by default:

```python
import rns_engine as rns

x = rns.G416(rns.G416.max_finite)
x + rns.G416(1)  # raises ExactRangeError
```

There is no silent wrap, infinity, rounding, or dtype promotion.

Advanced callers may explicitly opt into promotion:

```python
with rns.promote_exact():
    y = rns.G416(rns.G416.max_finite) + rns.G416(1)

assert y.tier is rns.G432
```

Promotion proceeds `G416 -> G432 -> G464 -> G4X` and ends when the smallest
sufficient exact tier is found. Leaving the context restores strict mode.

## Hidden implementation freedom

A public `G464` value may internally use a native integer, fixed-point body,
dyadic representation, RNS rails, multiple limbs, a shared denominator, or a
specialized CPU/GPU kernel. Those details are private implementation choices.
The stable public contract is range, exactness, overflow behavior, and the
operation result.

## Unified benchmark

Run all matched tiers together:

```bash
rns-tier-bench --size 256 --repeats 7 --output tier_report.json
```

The report always includes:

- `G416` versus FP16;
- `G432` versus FP32;
- `G464` versus FP64;
- exactness evidence;
- timing evidence under one declared contract.

The first harness is a correctness reference and explicitly refuses a
performance claim. Optimized G4 bodies will replace the reference adapter while
preserving the same tier and benchmark schema.
