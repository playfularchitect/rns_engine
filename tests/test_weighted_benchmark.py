from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "weighted_int32_baseline.py"
)


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "weighted_int32_baseline",
        BENCHMARK_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load weighted INT32 benchmark module")

    module = importlib.util.module_from_spec(spec)
    # Dataclasses with postponed annotations resolve their defining module
    # through sys.modules while the class decorator executes.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def test_weight_families_are_deterministic():
    benchmark = _load_benchmark_module()

    assert benchmark._weights(4, "unit", 128) == (1, 1, 1, 1)
    assert benchmark._weights(4, "alternating", 128) == (1, -1, 1, -1)
    assert benchmark._weights(4, "radix", 8) == (1, 8, 64, 512)
    assert benchmark._weights(4, "centered-radix", 8) == (1, -8, 64, -512)


def test_tiny_benchmark_case_runs_with_exact_witnesses():
    benchmark = _load_benchmark_module()

    result = benchmark.run_case(
        terms=2,
        outputs=8,
        max_abs=7,
        weight_mode="alternating",
        radix=128,
        repeats=1,
        warmups=0,
        sample_count=8,
        rng=np.random.default_rng(12345),
    )

    assert result.terms == 2
    assert result.outputs == 8
    assert result.fused_native_calls == 1
    assert result.staged_native_calls == 7
    assert result.fused_speedup_over_staged > 0
    assert result.max_abs_bound <= 14
    assert result.unique_signed_result is True
    assert result.fused_median_seconds >= 0
    assert result.staged_median_seconds >= 0
    assert result.decode_median_seconds >= 0
    assert result.numpy_int64_control_median_seconds is not None
