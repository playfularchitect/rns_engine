from pathlib import Path

from rns_engine.g4.master_key_demo import run_demo


def test_master_key_demo_closes_the_full_architecture_loop(tmp_path: Path):
    report = run_demo(tmp_path / "capsule")
    assert report["pattern_count"] == 1
    assert report["mutation_proposal_count"] >= 3
    assert report["generated_candidate_ids"] == ["direct_row_increase"]
    assert report["promotion"]["learner_decision"] == "PROMOTED"
    assert report["promotion"]["mdl_accepted"] is True
    assert report["predictive_class_count"] == 1
    assert report["deterministic_edge_count"] == 2
    assert report["superstep"]["start_state"] == "generate_one_launch_family"
    assert report["capsule"]["ok"] is True
