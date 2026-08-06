"""Deterministic end-to-end demonstration of the G4 Self-Teaching Master Key."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

from .capsule import G4Capsule
from .evidence import PhysicalEvidence
from .learner import G4Learner
from .mdl import CompleteCost, evaluate_merge, evaluate_split
from .model import Candidate, Observation, SearchBoundary
from .mutations import MutationProposal, MutationRegistry
from .predictive import PredictiveOutcome
from .proofs import ProofObject
from .teacher import ResidueDrivenTeacher


def _evidence_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _boundary(index: int) -> SearchBoundary:
    return SearchBoundary.create(
        environment="synthetic_t4",
        domain="exact_gemm",
        operation="matmul",
        constraints={
            "shape_id": f"launch_{index}",
            "k_band": "tiny",
            "output_band": "very_large",
        },
    )


def _observation(speedup: float, *, exact: bool = True) -> Observation:
    return Observation(
        compile_ok=True,
        legal=True,
        exact=exact,
        speedup=speedup,
        confidence_lower=max(0.01, speedup - 0.01),
        wins=27 if speedup > 1.002 else 12,
        blocks=31,
    )


def _launch_mutation(parent: Candidate, proposal: MutationProposal) -> Candidate:
    return Candidate.create(
        f"{parent.candidate_id}_{proposal.operation[:8]}",
        features=(*parent.features, "grouped_launch"),
        parameters={**dict(parent.parameters), "launch_grouping": 4},
        mutation_ops=(*parent.mutation_ops, proposal.operation),
        parent_id=parent.candidate_id,
        description_cost=parent.description_cost + 0.2,
        expected_work_cost=max(0.0, parent.expected_work_cost - 0.1),
        expected_memory_cost=parent.expected_memory_cost,
    )


def run_demo(capsule_path: Path) -> dict[str, object]:
    if capsule_path.exists():
        shutil.rmtree(capsule_path)
    capsule = G4Capsule(capsule_path)
    learner = G4Learner(capsule=capsule)
    teacher = ResidueDrivenTeacher(learner=learner, capsule=capsule, minimum_pattern_members=2)

    incumbent = Candidate.create(
        "direct_row",
        features=("direct", "row_major", "one_launch_per_boundary"),
        parameters={"tile": 128},
        mutation_ops=("seed",),
        description_cost=1.0,
        expected_work_cost=1.0,
    )

    boundaries: list[SearchBoundary] = []
    contracts = []
    for index in range(3):
        search_boundary = _boundary(index)
        contract = teacher.bind_contract(
            search_boundary,
            {
                "exact": True,
                "timing": "31 paired direct blocks",
                "shape_id": f"launch_{index}",
            },
        )
        boundaries.append(search_boundary)
        contracts.append(contract)
        incumbent_cost = CompleteCost(
            exact=True,
            live_structure_bits=100,
            correction_residue_bits=20,
            direct_latency_ms=1.20 + index * 0.01,
            workspace_bytes=0,
        )
        teacher.register_incumbent(boundary=contract, candidate=incumbent, cost=incumbent_cost)
        evidence = PhysicalEvidence.create(
            boundary=contract,
            rule_execution_fingerprint=incumbent.execution_fingerprint,
            exact=True,
            metrics={"direct_latency_ms": incumbent_cost.direct_latency_ms},
            shape=(4096, 4096, 16),
            launch_fraction=0.78,
            padding_fraction=0.05,
            memory_fraction=0.20,
            reduction_fraction=0.02,
            workspace_bytes=0,
        )
        teacher.ingest(
            search_boundary=search_boundary,
            boundary=contract,
            candidate=incumbent,
            observation=_observation(0.82),
            physical_evidence=evidence,
            residue_target=0.75,
            challenger_cost=incumbent_cost,
        )

    patterns = tuple(teacher.patterns.values())
    registry = MutationRegistry()
    registry.register("increase_work_per_launch", _launch_mutation)
    generated = teacher.generator(registry)(boundaries[0], learner, (incumbent,))
    if not generated:
        raise RuntimeError("residue pattern did not become an executable mutation")
    challenger = generated[0]

    challenger_cost = CompleteCost(
        exact=True,
        live_structure_bits=112,
        correction_residue_bits=8,
        mutation_bits=4,
        direct_latency_ms=0.68,
        workspace_bytes=0,
    )
    challenger_evidence = PhysicalEvidence.create(
        boundary=contracts[0],
        rule_execution_fingerprint=challenger.execution_fingerprint,
        exact=True,
        metrics={"direct_latency_ms": challenger_cost.direct_latency_ms},
        shape=(4096, 4096, 16),
        launch_fraction=0.18,
        padding_fraction=0.05,
        memory_fraction=0.20,
        reduction_fraction=0.02,
        workspace_bytes=0,
    )
    promotion = teacher.ingest(
        search_boundary=boundaries[0],
        boundary=contracts[0],
        candidate=challenger,
        observation=_observation(1.20),
        physical_evidence=challenger_evidence,
        residue_target=0.65,
        challenger_cost=challenger_cost,
        parent=incumbent,
    )

    new_predictive_classes = ()
    for index in range(2):
        new_predictive_classes = teacher.record_predictive_outcome(
            PredictiveOutcome.create(
                boundary_id=contracts[index].boundary_id,
                history_signature={"shape_id": f"launch_{index}", "old_rule": "direct_row"},
                continuation_signature=("increase_work_per_launch", "certify_exact"),
                conclusion_signature={"next_rule": "grouped_launch", "result": "promoted"},
                exact_evidence_hash=_evidence_hash(f"predictive-{index}"),
            )
        )

    merge = evaluate_merge(
        separate_structure_bits=240,
        merged_structure_bits=130,
        restoring_residue_bits=24,
    )
    split = evaluate_split(
        shared_structure_bits=130,
        current_correction_bits=90,
        split_structure_bits=155,
        remaining_correction_bits=20,
    )
    teacher.record_merge_decision(merge)
    teacher.record_split_decision(split)

    proof_one = ProofObject.create(
        proof_type="exact-family-elimination",
        premises=(contracts[0].contract_hash,),
        conclusion="launch_bound_class rejects one_launch_per_boundary family",
        evidence_hash=_evidence_hash("edge-one"),
        exact=True,
    )
    edge_one = teacher.install_deterministic_edge(
        boundary=contracts[0],
        source_state="generate_one_launch_family",
        destination_state="family_dominated",
        proof=proof_one,
        dominated_region=True,
    )
    proof_two = ProofObject.create(
        proof_type="exact-ledger-update",
        premises=(edge_one.proof_id,),
        conclusion="family_dominated updates coverage ledger",
        evidence_hash=_evidence_hash("edge-two"),
        exact=True,
    )
    edge_two = teacher.install_deterministic_edge(
        boundary=contracts[0],
        source_state="family_dominated",
        destination_state="coverage_updated",
        proof=proof_two,
    )
    superstep = teacher.compile_superstep(boundary=contracts[0], edges=(edge_one, edge_two))

    certificate = teacher.build_certificate(
        mode="timed",
        cursor="after_launch_residue_class",
        candidate_ledger={"incumbent": incumbent.to_dict(), "challenger": challenger.to_dict()},
        coverage_ledger={"tested_boundaries": [item.fingerprint for item in boundaries]},
    )
    capsule_result = capsule.verify()

    return {
        "schema": "rns_engine.g4_master_key_demo.v1",
        "residue_count": len(teacher.residues.all()),
        "unresolved_residue_count": len(teacher.residues.unresolved()),
        "pattern_count": len(patterns),
        "pattern_distinctions": [pattern.proposed_distinction for pattern in patterns],
        "mutation_proposal_count": len(teacher.mutation_proposals),
        "generated_candidate_ids": [candidate.candidate_id for candidate in generated],
        "promotion": {
            "learner_decision": promotion.experience.decision,
            "mdl_accepted": bool(promotion.promotion_decision and promotion.promotion_decision.accepted),
            "proof_id": promotion.promotion_proof_id,
        },
        "predictive_class_count": len(teacher.predictive_classes),
        "new_predictive_class_count": len(new_predictive_classes),
        "merge": asdict(merge),
        "split": asdict(split),
        "deterministic_edge_count": len(teacher.deterministic_edges),
        "superstep": superstep.to_dict(),
        "certificate_id": certificate.certificate_id,
        "certificate_actg_stamp_prefix": certificate.actg_stamp[:32],
        "continuation": certificate.continuation.to_dict(),
        "capsule": capsule_result,
        "note": "Deterministic architecture proof; not a hardware performance claim.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the G4 Self-Teaching Master Key demonstration.")
    parser.add_argument("--capsule", type=Path, default=Path("G4_Master_Key_Capsule"))
    parser.add_argument("--output", type=Path, default=Path("g4_master_key_demo.json"))
    args = parser.parse_args(argv)
    report = run_demo(args.capsule)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
