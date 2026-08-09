from __future__ import annotations

import io

import pytest

from rns_engine.g4_results import g4_results


def test_g4_results_keeps_integer_and_rational_claims_distinct():
    evidence = g4_results(display=False)
    integer = evidence["campaigns"]["integer_fp16_input_clean_sweep"]
    rational = evidence["campaigns"]["dynamic_exact_rational"]

    assert integer["declared_shapes"] == 1024
    assert integer["exact_wins"] == 938
    assert integer["floating_wins"] == 86
    assert integer["statistical_ties"] == 0
    assert integer["errors"] == 0
    assert integer["exact_win_rate"] == pytest.approx(938 / 1024)
    assert integer["all_shapes_exact_replay_pre_and_post"] is True
    assert integer["overall_speedup_geometric_mean"] == pytest.approx(1.235383405277301)
    assert integer["overall_speedup_median"] == pytest.approx(1.256704092)
    assert integer["exact_win_speedup_geometric_mean"] == pytest.approx(1.311051665330303)
    assert integer["nvidia_win_g4_throughput_retained_geometric_mean"] == pytest.approx(0.6459590179226226)
    assert integer["nvidia_win_g4_execution_time_penalty_from_geomean"] == pytest.approx(0.5480858262741786)

    assert rational["target_shapes"] == 1024
    assert rational["evaluated_shapes"] == 1024
    assert rational["certified_exact_wins"] == 870
    assert rational["nvidia_wins"] == 110
    assert rational["statistical_ties"] == 41
    assert rational["errors"] == 3
    assert rational["remaining_unresolved"] == 154
    assert rational["certified_exact_win_rate"] == pytest.approx(870 / 1024)
    assert rational["certified_speedup_geometric_mean"] == pytest.approx(1.417436071329068)
    assert rational["certified_speedup_median"] == pytest.approx(1.4064410325)
    assert rational["all_certified_actual_noninteger_inputs"] is True
    assert rational["all_certified_range_proved"] is True
    assert rational["all_certified_fp16_value_set_proved"] is True


def test_g4_results_report_labels_both_species_and_full_scorecards():
    stream = io.StringIO()
    g4_results(stream=stream)
    text = stream.getvalue()
    assert "G4 INTEGERS vs NVIDIA FP16 cuBLASLt" in text
    assert "G4 RATIONALS vs NVIDIA FP16 cuBLASLt" in text
    assert "G4 exact integers faster: 938 / 1024 (91.60%)" in text
    assert "Overall integer speedup across all 1,024 shapes: 1.235x geometric mean | 1.257x median" in text
    assert "Among 938 G4 wins: 1.311x geometric mean | 1.289x median | 2.647x best" in text
    assert "Across 86 NVIDIA wins, G4 retained 64.60% of NVIDIA throughput" in text
    assert "G4 exact rationals faster (certified): 870 / 1024 (84.96%)" in text
    assert "NVIDIA FP16 wins: 110 / 1024" in text
    assert "Statistical ties / errors: 41 / 3" in text
    assert "Among 870 certified G4 wins: 1.417x geometric mean | 1.406x median | 2.978x best" in text
    assert "INTEGER RESULT:  938 G4 wins / 86 NVIDIA wins / 0 ties / 0 errors" in text
    assert "RATIONAL RESULT: 870 G4 wins / 110 NVIDIA wins / 41 ties / 3 errors" in text
    assert "separate benchmark results" in text


def test_g4_results_bad_campaign_rejected():
    with pytest.raises(ValueError):
        g4_results("series2", display=False)
