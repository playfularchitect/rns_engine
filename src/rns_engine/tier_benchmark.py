"""One-run G416/G432/G464 benchmark contract.

This first version is deliberately a correctness-first reference harness. It
runs every public tier against the matching NumPy floating tier and emits one
machine-readable report. Optimized CPU/CUDA/G4 search bodies can replace the
reference adapter without changing the public benchmark law.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Callable, Iterable

import numpy as np

from .tiers import G416, G432, G464, G4Array, G4Tier, tensor


TIERS: tuple[G4Tier, ...] = (G416, G432, G464)


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    tier: str
    floating_opponent: str
    operation: str
    size: int
    repeats: int
    g4_reference_median_seconds: float
    floating_median_seconds: float
    floating_over_g4_ratio: float
    floating_exact_match: bool
    floating_mismatch_count: int
    g4_exact_match: bool
    status: str


def _median_seconds(fn: Callable[[], Any], repeats: int, warmups: int) -> float:
    for _ in range(warmups):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1_000_000_000.0)
    return statistics.median(samples)


def _fraction_of_numpy_float(value: np.generic[Any] | float) -> Fraction:
    numerator, denominator = float(value).as_integer_ratio()
    return Fraction(numerator, denominator)


def _exact_flat(values: np.ndarray[Any, Any]) -> list[Fraction]:
    return [Fraction(value) for value in np.asarray(values, dtype=object).flat]


def _float_mismatch_count(float_values: np.ndarray[Any, Any], exact_values: Iterable[Fraction]) -> int:
    return sum(
        _fraction_of_numpy_float(observed) != expected
        for observed, expected in zip(np.asarray(float_values).flat, exact_values)
    )


def _workload(size: int, seed: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    rng = np.random.default_rng(seed)
    left = rng.integers(-3, 4, size=size, dtype=np.int64)
    right = rng.integers(-3, 4, size=size, dtype=np.int64)
    return left, right


def _run_elementwise(
    tier: G4Tier,
    operation: str,
    size: int,
    repeats: int,
    warmups: int,
    seed: int,
) -> BenchmarkRow:
    left_i64, right_i64 = _workload(size, seed)
    float_dtype = tier.numpy_float_dtype
    assert float_dtype is not None
    left_float = left_i64.astype(float_dtype)
    right_float = right_i64.astype(float_dtype)
    left_g4 = tensor(left_i64, dtype=tier)
    right_g4 = tensor(right_i64, dtype=tier)

    if operation == "add":
        g4_fn = lambda: left_g4 + right_g4
        float_fn = lambda: np.add(left_float, right_float)
    elif operation == "mul":
        g4_fn = lambda: left_g4 * right_g4
        float_fn = lambda: np.multiply(left_float, right_float)
    else:
        raise ValueError(f"unsupported elementwise operation {operation!r}")

    exact_result = g4_fn()
    assert isinstance(exact_result, G4Array)
    floating_result = np.asarray(float_fn())
    exact_values = _exact_flat(exact_result.fractions(copy=False))
    mismatch_count = _float_mismatch_count(floating_result, exact_values)

    g4_seconds = _median_seconds(g4_fn, repeats, warmups)
    float_seconds = _median_seconds(float_fn, repeats, warmups)
    ratio = float_seconds / g4_seconds if g4_seconds else float("inf")

    return BenchmarkRow(
        tier=tier.name,
        floating_opponent=f"FP{tier.bits}",
        operation=operation,
        size=size,
        repeats=repeats,
        g4_reference_median_seconds=g4_seconds,
        floating_median_seconds=float_seconds,
        floating_over_g4_ratio=ratio,
        floating_exact_match=(mismatch_count == 0),
        floating_mismatch_count=mismatch_count,
        g4_exact_match=True,
        status="REFERENCE_ONLY_NO_PERFORMANCE_CLAIM",
    )


def _run_dot(
    tier: G4Tier,
    size: int,
    repeats: int,
    warmups: int,
    seed: int,
) -> BenchmarkRow:
    left_i64, right_i64 = _workload(size, seed)
    float_dtype = tier.numpy_float_dtype
    assert float_dtype is not None
    left_float = left_i64.astype(float_dtype)
    right_float = right_i64.astype(float_dtype)
    left_g4 = tensor(left_i64, dtype=tier)
    right_g4 = tensor(right_i64, dtype=tier)

    g4_fn = lambda: (left_g4 * right_g4).sum()
    float_fn = lambda: np.dot(left_float, right_float)

    exact_result = g4_fn().fractions(copy=False).reshape(-1)[0]
    floating_result = float_fn()
    mismatch_count = int(_fraction_of_numpy_float(floating_result) != Fraction(exact_result))

    g4_seconds = _median_seconds(g4_fn, repeats, warmups)
    float_seconds = _median_seconds(float_fn, repeats, warmups)
    ratio = float_seconds / g4_seconds if g4_seconds else float("inf")

    return BenchmarkRow(
        tier=tier.name,
        floating_opponent=f"FP{tier.bits}",
        operation="dot",
        size=size,
        repeats=repeats,
        g4_reference_median_seconds=g4_seconds,
        floating_median_seconds=float_seconds,
        floating_over_g4_ratio=ratio,
        floating_exact_match=(mismatch_count == 0),
        floating_mismatch_count=mismatch_count,
        g4_exact_match=True,
        status="REFERENCE_ONLY_NO_PERFORMANCE_CLAIM",
    )


def run_all_tiers(
    *,
    size: int = 256,
    repeats: int = 7,
    warmups: int = 2,
    seed: int = 20260805,
    operations: tuple[str, ...] = ("add", "mul", "dot"),
) -> dict[str, Any]:
    if size <= 0:
        raise ValueError("size must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmups < 0:
        raise ValueError("warmups must be nonnegative")

    rows: list[BenchmarkRow] = []
    for tier_index, tier in enumerate(TIERS):
        tier_seed = seed + tier_index * 1000
        for operation in operations:
            if operation in {"add", "mul"}:
                row = _run_elementwise(
                    tier,
                    operation,
                    size,
                    repeats,
                    warmups,
                    tier_seed,
                )
            elif operation == "dot":
                row = _run_dot(tier, size, repeats, warmups, tier_seed)
            else:
                raise ValueError(f"unknown operation {operation!r}")
            rows.append(row)

    return {
        "schema": "rns_engine.g4_tier_benchmark.v1",
        "strict_default": True,
        "promotion_is_explicit_opt_in": True,
        "performance_claim_admitted": False,
        "implementation_stage": "correctness_reference",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "contract": {
            "G416": "exact range tier versus FP16",
            "G432": "exact range tier versus FP32",
            "G464": "exact range tier versus FP64",
            "all_tiers_run_together": True,
            "timed_costs": "public operation call on already-created arrays",
        },
        "rows": [asdict(row) for row in rows],
    }


def _parse_operations(raw: str) -> tuple[str, ...]:
    operations = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    allowed = {"add", "mul", "dot"}
    unknown = sorted(set(operations) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown operations: {', '.join(unknown)}")
    if not operations:
        raise argparse.ArgumentTypeError("at least one operation is required")
    return operations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run G416/FP16, G432/FP32, and G464/FP64 in one benchmark report."
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--operations", type=_parse_operations, default=("add", "mul", "dot"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    report = run_all_tiers(
        size=args.size,
        repeats=args.repeats,
        warmups=args.warmups,
        seed=args.seed,
        operations=args.operations,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
