from __future__ import annotations

from copy import deepcopy
from io import StringIO

import rns_engine.g4_trust as trust


def _fake_result() -> dict:
    return {
        "schema": "RNS-ENGINE-G4S1-LIVE-BENCHMARK-2",
        "series": 1,
        "campaign": "dynamic_exact_rational_vs_fp16",
        "mode": "full",
        "gpu": {
            "name": "Tesla T4",
            "compute_capability": "7.5",
            "driver_version": "580.82.07",
        },
        "runtime": {
            "binary_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
            "source_archive_sha256": "c" * 64,
            "build_nvcc": "Cuda compilation tools, release 12.8, V12.8.93",
            "config_provenance": {},
        },
        "timing_seconds": {
            "gpu_test": 12.5,
            "result_verification": 3.0,
            "total": 15.5,
        },
        "summary": {
            "shapes_run": 2,
            "runtime_errors": 0,
            "exact_replay_failures": 0,
            "live_exact_wins": 1,
            "live_floating_wins": 1,
            "live_statistical_ties": 0,
        },
        "rows": [
            {"shape_id": "A", "exact_replay_passed": True, "live_decision": "EXACT_WIN"},
            {"shape_id": "B", "exact_replay_passed": True, "live_decision": "FLOATING_WIN"},
        ],
    }


def _snapshot() -> dict:
    return {
        "available": True,
        "pstate": "P0",
        "temperature_c": "45",
        "power_w": "38.50",
        "sm_clock_mhz": "1590",
        "memory_clock_mhz": "5001",
    }


def test_trust_pack_hashes_all_rows_and_commits_to_receipt():
    result = _fake_result()
    pack = trust._make_trust_pack(
        result,
        started_utc="2026-08-09T06:00:00Z",
        completed_utc="2026-08-09T06:10:00Z",
        environment="Google Colab",
        package_version="0.11.1",
        current_cuda="12.8",
        gpu_before=_snapshot(),
        gpu_after=_snapshot(),
    )

    assert pack["schema"] == "RNS-ENGINE-G4S1-TRUST-PACK-1"
    assert pack["package"]["release_tag"] == "v0.11.1"
    assert pack["exactness"] == {
        "rows_exact": 2,
        "rows_total": 2,
        "runtime_errors": 0,
        "exact_replay_failures": 0,
    }
    assert len(pack["result_rows_sha256"]) == 64
    assert len(pack["run_receipt_sha256"]) == 64

    changed = _fake_result()
    changed["rows"][0]["live_decision"] = "STATISTICAL_TIE"
    changed_pack = trust._make_trust_pack(
        changed,
        started_utc="2026-08-09T06:00:00Z",
        completed_utc="2026-08-09T06:10:00Z",
        environment="Google Colab",
        package_version="0.11.1",
        current_cuda="12.8",
        gpu_before=_snapshot(),
        gpu_after=_snapshot(),
    )
    assert changed_pack["result_rows_sha256"] != pack["result_rows_sha256"]
    assert changed_pack["run_receipt_sha256"] != pack["run_receipt_sha256"]


def test_public_wrapper_adds_and_prints_trust_pack(monkeypatch):
    monkeypatch.setattr(trust, "_core_g4_benchmark", lambda *args, **kwargs: deepcopy(_fake_result()))
    monkeypatch.setattr(trust, "_gpu_state_snapshot", _snapshot)
    monkeypatch.setattr(trust, "_environment_name", lambda: "Google Colab")
    monkeypatch.setattr(trust, "_package_version", lambda: "0.11.1")
    monkeypatch.setattr(trust, "_current_cuda_version", lambda: "12.8")

    timestamps = iter(["2026-08-09T06:00:00Z", "2026-08-09T06:10:00Z"])
    monkeypatch.setattr(trust, "_utc_now", lambda: next(timestamps))

    output = StringIO()
    result = trust.g4_benchmark("full", display=True, stream=output)
    text = output.getvalue()

    assert "trust_pack" in result
    assert "TRUST PACK - G4 RATIONAL RUN RECEIPT" in text
    assert "Environment: Google Colab | rns_engine 0.11.1 | release v0.11.1" in text
    assert "Replay binary SHA-256:   " + "a" * 64 in text
    assert "Replay payload SHA-256:  " + "b" * 64 in text
    assert "Source archive SHA-256:   " + "c" * 64 in text
    assert result["trust_pack"]["result_rows_sha256"] in text
    assert result["trust_pack"]["run_receipt_sha256"] in text