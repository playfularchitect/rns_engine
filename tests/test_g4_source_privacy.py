from pathlib import Path

import rns_engine.g4_runtime as runtime


def test_public_execution_bundle_has_no_search_controller_or_learner():
    root = runtime.extracted_runtime_root() / "source"
    forbidden = (
        "collect_candidates",
        "search_candidates",
        "run_search",
        "grammar",
        "learner",
        "genome",
        "northstar",
        "north_star",
        "autonomous",
    )
    for path in root.glob("*.cu"):
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for token in forbidden:
            assert token not in text, f"{token} unexpectedly present in {path.name}"
