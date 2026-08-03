#!/usr/bin/env python3
"""Reproducible fused-vs-staged benchmark for weighted signed-INT32 RNS accumulation.

The public path uses one fused native call for signed INT32 reading, per-term
magnitude collection, positional weighting, and four-rail accumulation. The
retained staged reference uses the pre-fusion encode -> scale -> add sequence.
Both paths are checked against each other and against exact sampled witnesses.

This benchmark is CPU-only. It does not measure CUDA or Tensor Cores and makes
no universal hardware-performance claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import rns_engine as rns
from rns_engine.weighted import (
    _accumulate_weighted_int32_staged,
    _validate_weighted_inputs,
)


@dataclass(frozen=True, slots=True)
class CaseResult:
    terms: int
    outputs: int
    input_mib: float
    weight_mode: str
    max_abs_partial: int
    max_abs_bound: int
    unique_signed_result: bool
    signed_headroom: int
    fused_native_calls: int
    staged_native_calls: int
    fused_median_seconds: float
    fused_min_seconds: float
    staged_median_seconds: float
    staged_min_seconds: float
    fused_speedup_over_staged: float
    decode_median_seconds: float
    fused_partial_values_per_second: float
    staged_partial_values_per_second: float
    fused_output_values_per_second: float
    numpy_int64_control_median_seconds: float | None
    fused_over_numpy_ratio: float | None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(_positive_int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one value is required")
    return parsed


def _weights(term_count: int, mode: str, radix: int) -> tuple[int, ...]:
    if mode == "unit":
        return (1,) * term_count
    if mode == "alternating":
        return tuple(1 if position % 2 == 0 else -1 for position in range(term_count))
    if mode == "radix":
        return tuple(radix**position for position in range(term_count))
    if mode == "centered-radix":
        return tuple(
            (radix**position) * (1 if position % 2 == 0 else -1)
            for position in range(term_count)
        )
    raise ValueError(f"unsupported weight mode: {mode}")


def _median_timing(function, repeats: int) -> tuple[float, float, object]:
    elapsed: list[float] = []
    last_result = None
    for _ in range(repeats):
        started = time.perf_counter()
        last_result = function()
        elapsed.append(time.perf_counter() - started)
    return statistics.median(elapsed), min(elapsed), last_result


def _run_staged(partials: np.ndarray, weights: tuple[int, ...]):
    flat, exact_weights, output_shape, output_size, _ = _validate_weighted_inputs(
        partials,
        weights,
    )
    rails, bounds = _accumulate_weighted_int32_staged(
        flat,
        exact_weights,
        output_size,
    )
    certificate = rns.certify_weighted_sum_bound(exact_weights, bounds)
    return rns.WeightedInt32Result(
        encoded=rns.EncodedArray(*rails),
        certificate=certificate,
        output_shape=output_shape,
        weights=exact_weights,
        term_abs_bounds=bounds,
    )


def _verify_receipts_match(
    fused: rns.WeightedInt32Result,
    staged: rns.WeightedInt32Result,
) -> None:
    assert fused.output_shape == staged.output_shape
    assert fused.weights == staged.weights
    assert fused.term_abs_bounds == staged.term_abs_bounds
    assert fused.certificate == staged.certificate
    for fused_rail, staged_rail in zip(fused.encoded.rails(), staged.encoded.rails()):
        np.testing.assert_array_equal(fused_rail, staged_rail)


def _verify_sample(
    partials: np.ndarray,
    weights: tuple[int, ...],
    receipt: rns.WeightedInt32Result,
    sample_count: int,
) -> None:
    outputs = partials.shape[1]
    if outputs == 0:
        return

    indices = np.linspace(
        0,
        outputs - 1,
        num=min(sample_count, outputs),
        dtype=np.int64,
    )
    modular = receipt.decode_modular()
    signed = receipt.decode_signed() if receipt.certificate.unique else None

    for index_value in indices:
        output_index = int(index_value)
        exact = sum(
            weight * int(partials[term_index, output_index])
            for term_index, weight in enumerate(weights)
        )
        expected_modular = exact % int(rns.M)
        actual_modular = int(modular[output_index])
        if actual_modular != expected_modular:
            raise AssertionError(
                "modular witness mismatch at output "
                f"{output_index}: {actual_modular} != {expected_modular}"
            )
        if signed is not None and int(signed[output_index]) != exact:
            raise AssertionError(
                "signed witness mismatch at output "
                f"{output_index}: {int(signed[output_index])} != {exact}"
            )


def _numpy_control(
    partials: np.ndarray,
    weights: tuple[int, ...],
    repeats: int,
) -> float:
    widened_weights = np.asarray(weights, dtype=np.int64).reshape(-1, 1)
    widened_partials = partials.astype(np.int64, copy=False)

    def run() -> np.ndarray:
        return np.sum(widened_partials * widened_weights, axis=0, dtype=np.int64)

    median, _, _ = _median_timing(run, repeats)
    return median


def run_case(
    *,
    terms: int,
    outputs: int,
    max_abs: int,
    weight_mode: str,
    radix: int,
    repeats: int,
    warmups: int,
    sample_count: int,
    rng: np.random.Generator,
) -> CaseResult:
    partials = rng.integers(
        -max_abs,
        max_abs + 1,
        size=(terms, outputs),
        dtype=np.int32,
    )
    weights = _weights(terms, weight_mode, radix)

    for _ in range(warmups):
        rns.accumulate_weighted_int32(partials, weights)
        _run_staged(partials, weights)

    fused_median, fused_min, fused_object = _median_timing(
        lambda: rns.accumulate_weighted_int32(partials, weights),
        repeats,
    )
    staged_median, staged_min, staged_object = _median_timing(
        lambda: _run_staged(partials, weights),
        repeats,
    )
    if not isinstance(fused_object, rns.WeightedInt32Result):
        raise TypeError("fused accumulator returned an unexpected result type")
    if not isinstance(staged_object, rns.WeightedInt32Result):
        raise TypeError("staged accumulator returned an unexpected result type")

    fused = fused_object
    staged = staged_object
    _verify_receipts_match(fused, staged)
    _verify_sample(partials, weights, fused, sample_count)

    decode_median, _, _ = _median_timing(fused.decode_modular, repeats)
    nonzero_terms = sum(weight != 0 for weight in weights)
    partial_values = terms * outputs

    numpy_median: float | None = None
    fused_over_numpy: float | None = None
    int64_max = np.iinfo(np.int64).max
    if fused.certificate.max_abs_bound <= int64_max and all(
        -int64_max - 1 <= weight <= int64_max for weight in weights
    ):
        numpy_median = _numpy_control(partials, weights, repeats)
        if numpy_median > 0:
            fused_over_numpy = fused_median / numpy_median

    return CaseResult(
        terms=terms,
        outputs=outputs,
        input_mib=partials.nbytes / (1024 * 1024),
        weight_mode=weight_mode,
        max_abs_partial=max_abs,
        max_abs_bound=fused.certificate.max_abs_bound,
        unique_signed_result=fused.certificate.unique,
        signed_headroom=fused.certificate.headroom,
        fused_native_calls=1,
        staged_native_calls=1 + 3 * nonzero_terms,
        fused_median_seconds=fused_median,
        fused_min_seconds=fused_min,
        staged_median_seconds=staged_median,
        staged_min_seconds=staged_min,
        fused_speedup_over_staged=(
            staged_median / fused_median if fused_median > 0 else float("inf")
        ),
        decode_median_seconds=decode_median,
        fused_partial_values_per_second=(
            partial_values / fused_median if fused_median > 0 else float("inf")
        ),
        staged_partial_values_per_second=(
            partial_values / staged_median if staged_median > 0 else float("inf")
        ),
        fused_output_values_per_second=(
            outputs / fused_median if fused_median > 0 else float("inf")
        ),
        numpy_int64_control_median_seconds=numpy_median,
        fused_over_numpy_ratio=fused_over_numpy,
    )


def _print_result(result: CaseResult) -> None:
    print(
        f"terms={result.terms:>3} outputs={result.outputs:>9,} "
        f"input={result.input_mib:>8.2f} MiB unique="
        f"{'yes' if result.unique_signed_result else 'no'}"
    )
    print(
        f"  fused: calls={result.fused_native_calls:>3} "
        f"time={result.fused_median_seconds * 1e3:>10.3f} ms "
        f"partial-rate={result.fused_partial_values_per_second / 1e6:>10.2f} M/s"
    )
    print(
        f"  staged: calls≈{result.staged_native_calls:>3} "
        f"time={result.staged_median_seconds * 1e3:>10.3f} ms "
        f"partial-rate={result.staged_partial_values_per_second / 1e6:>10.2f} M/s"
    )
    print(
        f"  fused speedup={result.fused_speedup_over_staged:.2f}x; "
        f"decode={result.decode_median_seconds * 1e3:.3f} ms"
    )
    if result.numpy_int64_control_median_seconds is not None:
        print(
            "  int64 control: "
            f"{result.numpy_int64_control_median_seconds * 1e3:.3f} ms; "
            f"fused/control={result.fused_over_numpy_ratio:.2f}x"
        )
    else:
        print("  int64 control: skipped because the exact bound or weights exceed int64")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terms",
        type=_csv_positive_ints,
        default=(1, 2, 4, 8, 16),
        help="comma-separated term counts (default: 1,2,4,8,16)",
    )
    parser.add_argument(
        "--outputs",
        type=_csv_positive_ints,
        default=(1_024, 65_536, 1_000_000),
        help="comma-separated output sizes (default: 1024,65536,1000000)",
    )
    parser.add_argument(
        "--max-abs",
        type=_positive_int,
        default=127,
        help="maximum absolute generated INT32 partial value (default: 127)",
    )
    parser.add_argument(
        "--weight-mode",
        choices=("unit", "alternating", "radix", "centered-radix"),
        default="unit",
        help="positional weight family (default: unit)",
    )
    parser.add_argument(
        "--radix",
        type=_positive_int,
        default=128,
        help="radix used by radix weight modes (default: 128)",
    )
    parser.add_argument("--repeats", type=_positive_int, default=7)
    parser.add_argument("--warmups", type=_nonnegative_int, default=2)
    parser.add_argument("--sample-count", type=_positive_int, default=16)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="optional path for a machine-readable result file",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_abs > np.iinfo(np.int32).max:
        raise SystemExit("--max-abs must fit in signed INT32")

    print(f"rns_engine={rns.__version__}")
    print(f"python={platform.python_version()} numpy={np.__version__}")
    print(f"platform={platform.platform()}")
    print(f"M={rns.M} HALF_M={rns.HALF_M} AVX2={rns.HAS_AVX2}")
    print(
        f"weight_mode={args.weight_mode} radix={args.radix} "
        f"max_abs={args.max_abs} repeats={args.repeats} warmups={args.warmups}"
    )
    print()

    rng = np.random.default_rng(args.seed)
    results: list[CaseResult] = []
    for terms in args.terms:
        for outputs in args.outputs:
            result = run_case(
                terms=terms,
                outputs=outputs,
                max_abs=args.max_abs,
                weight_mode=args.weight_mode,
                radix=args.radix,
                repeats=args.repeats,
                warmups=args.warmups,
                sample_count=args.sample_count,
                rng=rng,
            )
            results.append(result)
            _print_result(result)

    if args.json is not None:
        payload = {
            "environment": {
                "rns_engine": rns.__version__,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "modulus": int(rns.M),
                "half_modulus": int(rns.HALF_M),
                "avx2": bool(rns.HAS_AVX2),
            },
            "configuration": {
                "terms": list(args.terms),
                "outputs": list(args.outputs),
                "max_abs": args.max_abs,
                "weight_mode": args.weight_mode,
                "radix": args.radix,
                "repeats": args.repeats,
                "warmups": args.warmups,
                "sample_count": args.sample_count,
                "seed": args.seed,
            },
            "results": [asdict(result) for result in results],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
