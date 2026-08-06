from pathlib import Path

import pytest

from rns_engine.g4 import (
    BoundaryContract,
    Candidate,
    CompleteCost,
    G4Capsule,
    G4Learner,
    MutationProposal,
    MutationRegistry,
    Observation,
    PhysicalEvidence,
    PredictiveOutcome,
    ProofObject,
    ProofStore,
    ResidueDrivenTeacher,
    SearchBoundary,
    compile_superstep,
    deterministic_edge_is_valid,
    discover_exact_patterns,
    discover_predictive_classes,
    evaluate_merge,
    evaluate_split,
    extract_residue,
    propose_mutations,
)
from rns_engine.g4.supersteps import DeterministicEdge


def search_boundary(name="b0"):
    return SearchBoundary.create(
        environment="T4",
        domain="exact_gemm",
        operation="matmul",
        constraints={"shape_id": name, "k_band": "tiny"},
    )


def contract_for(boundary):
    return ResidueDrivenTeacher.bind_contract(
        boundary,
        {"exact": True, "timing": "31 paired blocks", "shape": dict(boundary.constraints)},
    )


def body(name="base", *features):
    return Candidate.create(name, features=features or ("direct",), parameters={"tile": 128})


def observation(speedup, exact=True):
    return Observation(True, True, exact, speedup, max(0.01, speedup - 0.01), 27 if speedup > 1 else 10, 31)


def physical(contract, candidate, latency, *, exact=True, launch=0.8):
    return PhysicalEvidence.create(
        boundary=contract,
        rule_execution_fingerprint=candidate.execution_fingerprint,
        exact=exact,
        metrics={"direct_latency_ms": latency},
        shape=(4096, 4096, 16),
        launch_fraction=launch,
        padding_fraction=0.02,
        memory_fraction=0.2,
        reduction_fraction=0.01,
        workspace_bytes=0,
    )


def test_boundary_contract_and_evidence_are_content_addressed():
    boundary = search_boundary()
    contract = contract_for(boundary)
    candidate = body()
    evidence = physical(contract, candidate, 1.2)
    assert contract.boundary_id == boundary.fingerprint
    assert len(contract.contract_hash) == 64
    assert len(evidence.evidence_hash) == 64
    assert dict(evidence.signature)["launch_band"] == "dominant"


def test_inexact_evidence_fails_admission_before_residue():
    boundary = search_boundary()
    contract = contract_for(boundary)
    candidate = body()
    evidence = physical(contract, candidate, 1.2, exact=False)
    with pytest.raises(ValueError):
        extract_residue(
            boundary=contract,
            rule_execution_fingerprint=candidate.execution_fingerprint,
            evidence=evidence,
            metric_name="direct_latency_ms",
            target=1.0,
            objective="baseline",
        )


def test_cross_boundary_residue_discovers_one_exact_pattern():
    candidate = body()
    residues = []
    for index in range(3):
        boundary = search_boundary(f"b{index}")
        contract = contract_for(boundary)
        residues.append(
            extract_residue(
                boundary=contract,
                rule_execution_fingerprint=candidate.execution_fingerprint,
                evidence=physical(contract, candidate, 1.2 + index * 0.01),
                metric_name="direct_latency_ms",
                target=0.8,
                objective="baseline",
            )
        )
    patterns = discover_exact_patterns(residues, minimum_members=2)
    assert len(patterns) == 1
    assert len(patterns[0].member_boundaries) == 3
    assert patterns[0].proposed_distinction == "launch-bound"


def test_mdl_merge_and_split_laws_are_exact():
    merge = evaluate_merge(
        separate_structure_bits=300,
        merged_structure_bits=150,
        restoring_residue_bits=20,
    )
    split = evaluate_split(
        shared_structure_bits=150,
        current_correction_bits=100,
        split_structure_bits=175,
        remaining_correction_bits=10,
    )
    assert merge.accepted and merge.savings_bits == 130
    assert split.accepted and split.savings_bits == 65


def test_predictive_class_merges_future_equivalence_and_preserves_history():
    outcomes = [
        PredictiveOutcome.create(
            boundary_id=f"b{index}",
            history_signature={"history": index},
            continuation_signature=("mutate", "certify"),
            conclusion_signature={"rule": "winner"},
            exact_evidence_hash=f"e{index}",
        )
        for index in range(2)
    ]
    classes = discover_predictive_classes(outcomes)
    assert len(classes) == 1
    assert len(classes[0].restoring_history_residue) == 2


def test_residue_mutation_is_an_executable_search_object():
    candidate = body("parent", "direct")
    boundary = search_boundary()
    contract = contract_for(boundary)
    residue = extract_residue(
        boundary=contract,
        rule_execution_fingerprint=candidate.execution_fingerprint,
        evidence=physical(contract, candidate, 1.2),
        metric_name="direct_latency_ms",
        target=0.8,
        objective="baseline",
    )
    assert discover_exact_patterns([residue, residue], minimum_members=2) == []

    second_boundary = search_boundary("b1")
    second_contract = contract_for(second_boundary)
    second_residue = extract_residue(
        boundary=second_contract,
        rule_execution_fingerprint=candidate.execution_fingerprint,
        evidence=physical(second_contract, candidate, 1.3),
        metric_name="direct_latency_ms",
        target=0.8,
        objective="baseline",
    )
    pattern = discover_exact_patterns([residue, second_residue], minimum_members=2)[0]
    proposal = next(item for item in propose_mutations(pattern, candidate) if item.operation == "increase_work_per_launch")
    registry = MutationRegistry()

    def mutate(parent, _proposal: MutationProposal):
        return Candidate.create("child", features=(*parent.features, "grouped_launch"))

    registry.register("increase_work_per_launch", mutate)
    materialized = registry.materialize(proposal, candidate)
    assert materialized[0].candidate_id == "child"


def test_exact_superstep_requires_current_contract_and_exact_proofs():
    store = ProofStore()
    proof_one = ProofObject.create(
        proof_type="edge",
        premises=(),
        conclusion="a->b",
        evidence_hash="e1",
        exact=True,
    )
    proof_two = ProofObject.create(
        proof_type="edge",
        premises=(proof_one.proof_id,),
        conclusion="b->c",
        evidence_hash="e2",
        exact=True,
    )
    store.add(proof_one)
    store.add(proof_two)
    edge_one = DeterministicEdge("a", "b", proof_one.proof_id, "contract")
    edge_two = DeterministicEdge("b", "c", proof_two.proof_id, "contract")
    assert deterministic_edge_is_valid(edge_one, current_contract_hash="contract", proof_store=store)
    assert not deterministic_edge_is_valid(edge_one, current_contract_hash="other", proof_store=store)
    superstep = compile_superstep((edge_one, edge_two), proof_store=store, contract_hash="contract")
    assert superstep.start_state == "a" and superstep.end_state == "c"
    assert store.require_exact(superstep.proof_id).exact


def test_teacher_turns_residue_into_pattern_mutation_promotion_and_certificate(tmp_path: Path):
    capsule = G4Capsule(tmp_path / "capsule")
    learner = G4Learner(capsule=capsule)
    teacher = ResidueDrivenTeacher(learner=learner, capsule=capsule)
    incumbent = body("parent", "direct", "one_launch")
    boundaries = [search_boundary("a"), search_boundary("b")]
    contracts = [contract_for(item) for item in boundaries]

    for index, (boundary, contract) in enumerate(zip(boundaries, contracts)):
        cost = CompleteCost(exact=True, live_structure_bits=100, direct_latency_ms=1.2 + index * 0.01)
        teacher.register_incumbent(boundary=contract, candidate=incumbent, cost=cost)
        teacher.ingest(
            search_boundary=boundary,
            boundary=contract,
            candidate=incumbent,
            observation=observation(0.8),
            physical_evidence=physical(contract, incumbent, cost.direct_latency_ms),
            residue_target=0.8,
            challenger_cost=cost,
        )

    assert len(teacher.patterns) == 1
    assert any(proposal.operation == "increase_work_per_launch" for proposal in teacher.mutation_proposals.values())
    registry = MutationRegistry()

    def mutate(parent, proposal):
        return Candidate.create("child", features=(*parent.features, proposal.operation))

    registry.register("increase_work_per_launch", mutate)
    generated = teacher.generator(registry)(boundaries[0], learner, (incumbent,))
    assert len(generated) == 1
    challenger = generated[0]
    challenger_cost = CompleteCost(exact=True, live_structure_bits=110, direct_latency_ms=0.7)
    result = teacher.ingest(
        search_boundary=boundaries[0],
        boundary=contracts[0],
        candidate=challenger,
        observation=observation(1.2),
        physical_evidence=physical(contracts[0], challenger, 0.7, launch=0.2),
        residue_target=0.6,
        challenger_cost=challenger_cost,
        parent=incumbent,
    )
    assert result.promotion_decision and result.promotion_decision.accepted
    assert result.promotion_proof_id
    certificate = teacher.build_certificate(mode="timed", cursor="after-test")
    assert certificate.continuation.mode == "timed"
    assert len(certificate.actg_stamp) > 32
    assert capsule.verify()["ok"] is True
