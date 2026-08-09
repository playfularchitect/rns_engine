from io import StringIO

import rns_engine.g4_public as public


def _fake_species(name):
    rows = [{
        "shape_id": name,
        "m": 2, "n": 2, "k": 2,
        "exact_replay_passed": True,
        "exact_median_ms": 1.0,
        "fraction_end_to_end_median_ms": 1.2,
        "speedup_fp16_over_exact": 1.25,
        "live_decision": "EXACT_WIN",
    }]
    return {
        "summary": {
            "live_exact_wins": 1,
            "live_floating_wins": 0,
            "live_statistical_ties": 0,
        },
        "rows": rows,
        "xops": {"suite_g4ops_per_second": 16000.0, "xops_receipt_sha256": name[0] * 64},
        "trust_pack": {"run_receipt_sha256": name[-1] * 64},
    }


def test_combined_benchmark_runs_integer_then_rational(monkeypatch):
    order = []
    integer = _fake_species("integer")
    rational = _fake_species("rational")

    def fake_integer(*args, **kwargs):
        order.append("integer")
        return integer

    def fake_rational(*args, **kwargs):
        order.append("rational")
        return rational

    monkeypatch.setattr(public, "_g4_integer_benchmark", fake_integer)
    monkeypatch.setattr(public, "_g4_rational_benchmark", fake_rational)
    got = public.g4_benchmark("quick", display=False)
    assert order == ["integer", "rational"]
    assert got["integer"] is integer
    assert got["rational"] is rational
    assert len(got["combined_receipt_sha256"]) == 64


def test_combined_output_prints_xops_key_once_and_integer_distribution(monkeypatch):
    monkeypatch.setattr(public, "_g4_integer_benchmark", lambda *a, **k: _fake_species("integer"))
    monkeypatch.setattr(public, "_g4_rational_benchmark", lambda *a, **k: _fake_species("rational"))
    stream = StringIO()
    public.g4_benchmark("quick", display=True, stream=stream)
    text = stream.getvalue()
    assert text.count("XOPS / G4OPS KEY") == 1
    assert "PART 1 / 2: G4 INTEGERS" in text
    assert "G4 INTEGERS - SPEED DISTRIBUTION - FRESH RUN" in text
    assert "All 1 shapes: 1.250x geomean | 1.250x median" in text
    assert "G4-winning shapes: 1.250x geomean | 1.250x median | 1.250x best" in text
    assert "NVIDIA-winning shapes: none in this run" in text
    assert "PART 2 / 2: G4 RATIONALS" in text
    assert "No combined win percentage" in text
