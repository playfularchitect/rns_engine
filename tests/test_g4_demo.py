from pathlib import Path

from rns_engine.g4.demo import run_demo


def test_demo_runs_two_boundaries_and_persists_capsule(tmp_path: Path):
    report = run_demo(tmp_path / "capsule", budget=8)
    assert report["capsule"]["ok"] is True
    assert report["first_boundary"]["evaluated"] == 8
    assert report["second_boundary"]["evaluated"] == 8
    assert report["experience_count"] == 16
    assert len(report["transfer_top5_before_second_run"]) == 5
