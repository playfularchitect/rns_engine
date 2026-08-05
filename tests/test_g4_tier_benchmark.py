from rns_engine.tier_benchmark import run_all_tiers


def test_one_run_contains_all_three_matched_float_opponents():
    report = run_all_tiers(size=16, repeats=1, warmups=0, operations=("add",))
    rows = report["rows"]
    assert report["strict_default"] is True
    assert report["performance_claim_admitted"] is False
    assert [(row["tier"], row["floating_opponent"]) for row in rows] == [
        ("G416", "FP16"),
        ("G432", "FP32"),
        ("G464", "FP64"),
    ]
    assert all(row["g4_exact_match"] for row in rows)
