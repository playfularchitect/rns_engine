from copy import deepcopy
from io import StringIO

import rns_engine.g4_trust as trust


def _result():
    return {
        "schema": "RNS-ENGINE-G4S1-LIVE-BENCHMARK-2",
        "series": 1,
        "campaign": "dynamic_exact_rational_vs_fp16",
        "mode": "full",
        "gpu": {"name": "Tesla T4", "compute_capability": "7.5", "driver_version": "580.82.07"},
        "runtime": {
            "binary_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
            "source_archive_sha256": "c" * 64,
            "build_nvcc": "Cuda compilation tools, release 12.8, V12.8.93",
            "config_provenance": {},
        },
        "timing_seconds": {"gpu_test": 1.0, "result_verification": 1.0, "total": 2.0},
        "summary": {
            "shapes_run": 2,
            "runtime_errors": 0,
            "exact_replay_failures": 0,
            "live_exact_wins": 1,
            "live_floating_wins": 1,
            "live_statistical_ties": 0,
            "frozen_decision_matches": 2,
        },
        "rows": [
            {"shape_id": "A", "exact_replay_passed": True, "live_decision": "EXACT_WIN"},
            {"shape_id": "B", "exact_replay_passed": True, "live_decision": "FLOATING_WIN"},
        ],
    }


def _snapshot():
    return {
        "available": True,
        "pstate": "P0",
        "temperature_c": "45",
        "power_w": "38.50",
        "sm_clock_mhz": "1590",
        "memory_clock_mhz": "5001",
    }


def _fake_core(mode, *, display, stream):
    if display:
        trust._box(
            "EXACTNESS - FRESH RUN",
            ["Exact rational calculations correct: 2 / 2", "Result: PASS"],
            stream,
        )
        trust._box(
            "SPEED - FRESH RUN DIRECTLY AGAINST NVIDIA FP16",
            [
                "Exact rational faster: 1 / 2  (50.00%)",
                "NVIDIA FP16 faster:    1 / 2  (50.00%)",
                "Too close to call:     0 / 2  (0.00%)",
                "Median speedup when exact wins: 1.400x",
                "Median speedup including rational bookkeeping: 1.350x",
            ],
            stream,
        )
        trust._box(
            "REPRODUCIBILITY",
            [
                "Same winner/tie result as original: 2 / 2  (100.00%)",
                "Validated replay binary SHA-256: " + "a" * 64,
                "Total elapsed: 00:02",
            ],
            stream,
        )
    return deepcopy(_result())


def test_live_report_is_explicitly_rational(monkeypatch):
    monkeypatch.setattr(trust, "_core_g4_benchmark", _fake_core)
    monkeypatch.setattr(trust, "_gpu_state_snapshot", _snapshot)
    monkeypatch.setattr(trust, "_environment_name", lambda: "Google Colab")
    monkeypatch.setattr(trust, "_package_version", lambda: "0.11.2")
    monkeypatch.setattr(trust, "_current_cuda_version", lambda: "12.8")
    times = iter(["2026-08-09T06:00:00Z", "2026-08-09T06:10:00Z"])
    monkeypatch.setattr(trust, "_utc_now", lambda: next(times))

    output = StringIO()
    result = trust.g4_benchmark("full", display=True, stream=output)
    text = output.getvalue()

    assert result["live_test"] == "G4_RATIONALS_vs_NVIDIA_FP16_cuBLASLt"
    assert result["summary"]["rational_frozen_decision_matches"] == 2
    assert "G4 INTEGERS  vs NVIDIA FP16 cuBLASLt" in text
    assert "G4 RATIONALS vs NVIDIA FP16 cuBLASLt" in text
    assert "LIVE BENCHMARK BELOW: G4 RATIONALS vs NVIDIA FP16 ONLY." in text

    assert "G4 RATIONALS - EXACTNESS - FRESH RUN" in text
    assert "G4 rational calculations correct: 2 / 2" in text
    assert "G4 RATIONALS - SPEED vs NVIDIA FP16 - FRESH RUN" in text
    assert "G4 RATIONALS faster than NVIDIA FP16: 1 / 2" in text
    assert "NVIDIA FP16 faster than G4 RATIONALS:" in text
    assert "G4 RATIONALS / NVIDIA FP16 statistical ties:" in text
    assert "G4 RATIONALS - REPRODUCIBILITY" in text
    assert "RATIONAL winner/tie classification matches original RATIONAL benchmark:" in text

    assert "Same winner/tie result as original:" not in text
    assert "Too close to call:" not in text
    assert "TRUST PACK - G4 RATIONAL RUN RECEIPT" in text