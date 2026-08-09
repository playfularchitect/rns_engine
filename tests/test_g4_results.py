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


def test_g4_results_report_labels_both_percentages():
    stream = io.StringIO()
    g4_results(stream=stream)
    text = stream.getvalue()
    assert "938 / 1024 (91.60%)" in text
    assert "870 / 1024 (84.96%)" in text
    assert "separate claims" in text


def test_g4_results_shape_lookup_is_sanitized():
    result = g4_results("rational", shape_id="T4GL0021", display=False)
    row = result["shape_rows"][0]
    assert row["shape_id"] == "T4GL0021"
    assert row["certified_exact_win"] is True
    assert row["speedup_fp16_over_exact"] > 1.0
    assert set(row) == {
        "campaign", "actual_noninteger_inputs", "bootstrap_high", "bootstrap_low",
        "category", "certified_exact_win", "exact_block_wins", "exact_median_ms",
        "final_decision", "fp16_median_ms", "fp16_value_set_proved", "k", "m", "n",
        "paired_blocks", "range_proved", "shape_id", "speedup_fp16_over_exact",
    }


def test_g4_results_bad_campaign_rejected():
    with pytest.raises(ValueError):
        g4_results("series2", display=False)
