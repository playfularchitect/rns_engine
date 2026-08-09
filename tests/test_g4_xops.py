from io import StringIO

import pytest

from rns_engine.g4_xops import (
    build_xops_summary,
    format_xops,
    gemm_xop_count,
    print_xops_key,
    xops_per_second,
)


def test_xops_uses_conventional_gemm_count():
    assert gemm_xop_count(2, 3, 4) == 48
    assert xops_per_second(2, 3, 4, 2.0) == 24_000.0


def test_failed_exactness_gets_zero_xop_credit():
    rows = [
        {"shape_id": "A", "m": 2, "n": 3, "k": 4, "exact_replay_passed": True, "exact_median_ms": 2.0},
        {"shape_id": "B", "m": 2, "n": 3, "k": 4, "exact_replay_passed": False, "exact_median_ms": 2.0},
    ]
    got = build_xops_summary(
        rows,
        species="TEST",
        headline_time_key="exact_median_ms",
        timing_boundary="test",
    )
    assert got["rows_exact_eligible"] == 1
    assert got["xops_credited"] == 48
    assert got["per_shape"][0]["g4ops_per_second"] == 24_000.0
    assert got["per_shape"][1]["g4ops_per_second"] == 0.0
    assert len(got["xops_receipt_sha256"]) == 64


def test_xops_key_is_self_explaining():
    stream = StringIO()
    print_xops_key(stream)
    text = stream.getvalue()
    assert "XOP   = one mathematically exact arithmetic operation." in text
    assert "XOPS  = exact arithmetic operations per second." in text
    assert "G4OPS = XOPS delivered by a G4 implementation." in text
    assert "2*M*N*K" in text
    assert "fails exactness earns 0 XOPS" in text


def test_format_xops_si_prefixes():
    assert format_xops(12.0).endswith(" XOPS")
    assert format_xops(2.5e9) == "2.500 GXOPS"
    assert format_xops(4.2e12) == "4.200 TXOPS"


def test_bad_time_rejected():
    with pytest.raises(ValueError):
        xops_per_second(1, 1, 1, 0)
