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
    assert integer["exact_win_rate"] == pytest.approx(938 / 1024)
    assert integer["all_shapes_exact_replay_pre_and_post"] is True

    assert rational["target_shapes"] == 1024
    assert rational["evaluated_shapes"] == 1024
    assert rational["certified_exact_wins"] == 870
    assert rational["remaining_unresolved"] == 154
    assert rational["certified_exact_win_rate"] == pytest.approx(870 / 1024)
    assert rational["all_certified_actual_noninteger_inputs"] is True
    assert rational["all_certified_range_proved"] is True
    assert rational["all_certified_fp16_value_set_proved"] is True


def test_g4_results_report_labels_both_species_and_percentages():
    stream = io.StringIO()
    g4_results(stream=stream)
    text = stream.getvalue()
    assert "G4 INTEGERS vs NVIDIA FP16 cuBLASLt" in text
    assert "G4 RATIONALS vs NVIDIA FP16 cuBLASLt" in text
    assert "G4 exact integers faster: 938 / 1024 (91.60%)" in text
    assert "G4 exact rationals faster (certified): 870 / 1024 (84.96%)" in text
    assert "INTEGER RESULT:  938 / 1024 = 91.60%" in text
    assert "RATIONAL RESULT: 870 / 1024 = 84.96%" in text
    assert "separate benchmark results" in text


def test_g4_results_bad_campaign_rejected():
    with pytest.raises(ValueError):
        g4_results("series2", display=False)