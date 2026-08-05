from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from rns_engine.g4 import (
    Candidate,
    G4Capsule,
    G4Learner,
    JudgeLaw,
    MidRunSupervisor,
    Observation,
    SearchBoundary,
    actg_to_bytes,
    bytes_to_actg,
)


def boundary(operation="matmul"):
    return SearchBoundary.create(
        environment="T4",
        domain="exact_gemm",
        operation=operation,
        constraints={"k_band": "deep"},
    )


def candidate(identifier, *features, **costs):
    return Candidate.create(identifier, features=features, **costs)


def win(speedup=1.10):
    return Observation(True, True, True, speedup, speedup - 0.01, 27, 31)


def loss(speedup=0.90):
    return Observation(True, True, True, speedup, speedup - 0.01, 10, 31)


def test_actg_round_trip_uses_frozen_mapping():
    payload = b"G4\x00\xff"
    genome = bytes_to_actg(payload)
    assert set(genome) <= set("ACTG")
    assert actg_to_bytes(genome) == payload
    assert bytes_to_actg(b"\x00") == "AAAA"
    assert bytes_to_actg(b"\xff") == "GGGG"


def test_judge_is_frozen_and_digest_changes_with_law():
    judge = JudgeLaw()
    with pytest.raises(FrozenInstanceError):
        judge.minimum_speedup = 0.5
    assert judge.digest != JudgeLaw(minimum_speedup=1.01).digest


def test_positive_and_negative_evidence_reorders_candidates():
    learner = G4Learner()
    target = boundary()
    direct = candidate("direct", "direct", "row_major")
    shared = candidate("shared", "shared", "row_major")
    learner.observe(target, direct, win())
    learner.observe(target, shared, loss())
    new_direct = candidate("direct2", "direct", "row_major", "split2")
    new_shared = candidate("shared2", "shared", "row_major", "split2")
    ranked = learner.rank(target, [new_shared, new_direct])
    assert ranked[0].candidate is new_direct


def test_no_repeat_ledger_is_enforced():
    learner = G4Learner()
    target = boundary()
    body = candidate("one", "direct")
    learner.observe(target, body, win())
    assert learner.rank(target, [body]) == []
    with pytest.raises(ValueError):
        learner.observe(target, body, win())


def test_parent_child_credit_focuses_changed_features():
    learner = G4Learner()
    target = boundary()
    parent = candidate("parent", "direct", "reuse_a")
    child = Candidate.create("child", features=("direct", "reuse_ab"), parent_id="parent")
    learner.observe(target, child, win(), parent=parent)
    keys = "\n".join(learner.weights)
    assert "reuse_ab" in keys
    assert "reuse_a" in keys


def test_capsule_checkpoint_resume_and_judge_guard(tmp_path: Path):
    capsule = G4Capsule(tmp_path / "G4_Capsule")
    learner = G4Learner(capsule=capsule)
    target = boundary()
    body = candidate("one", "direct", "row_major")
    learner.observe(target, body, win())
    assert capsule.verify()["ok"] is True
    resumed = G4Learner(capsule=capsule)
    assert resumed.evaluation_key(target, body) in resumed.evaluated
    with pytest.raises(ValueError):
        G4Learner(judge=JudgeLaw(minimum_speedup=1.5), capsule=capsule)


def test_supervisor_updates_mid_run_and_never_repeats():
    learner = G4Learner()
    supervisor = MidRunSupervisor(learner)
    target = boundary()
    candidates = [
        candidate("a", "shared"),
        candidate("b", "direct"),
        candidate("c", "direct", "reuse_ab"),
        candidate("d", "shared", "reuse_ab"),
    ]

    def evaluate(_boundary, body):
        return win(1.2) if "direct" in body.features else loss(0.8)

    summary = supervisor.run(target, candidates, evaluate, budget=4)
    assert summary.evaluated == 4
    assert len(set(summary.selection_order)) == 4
    assert summary.champion is not None


def test_residue_patterns_extract_repeated_failure_structure():
    learner = G4Learner()
    target = boundary()
    for index in range(4):
        learner.observe(target, candidate(f"s{index}", "shared", f"variant{index}"), loss())
    patterns = learner.patterns(target, minimum_support=3, minimum_rate=0.75)
    assert any(pattern.feature == "shared" and pattern.residue == "BELOW_SPEED" for pattern in patterns)
