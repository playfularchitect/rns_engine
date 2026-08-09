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


def test_live_report_is_explicitly_rational(monkeypatch):
    monkeypatch.setattr(trust, "_core_g4_benchmark", lambda *args, **kwargs: deepcopy(_result()))
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
    assert "G4 RATIONALS - FRESH RUN SUMMARY" in text
    assert "RATIONAL winner/tie classification matches original rational benchmark: 2 / 2 (100.00%)" in text
    assert "Frozen G4 INTEGER result was not rerun by this command." in text
    assert "TRUST PACK - G4 RATIONAL RUN RECEIPT" in text