from rns_engine.g4_xops import build_xops_summary


def test_rational_headline_xops_can_include_bookkeeping_while_kernel_is_reported_separately():
    rows = [{
        "shape_id": "R",
        "m": 2,
        "n": 2,
        "k": 2,
        "exact_replay_passed": True,
        "exact_median_ms": 1.0,
        "fraction_end_to_end_median_ms": 2.0,
    }]
    got = build_xops_summary(
        rows,
        species="G4_RATIONALS",
        headline_time_key="fraction_end_to_end_median_ms",
        kernel_time_key="exact_median_ms",
        timing_boundary="end-to-end",
    )
    assert got["suite_g4ops_per_second"] == 8_000.0
    assert got["suite_kernel_g4ops_per_second"] == 16_000.0
