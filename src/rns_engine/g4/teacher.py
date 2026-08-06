"""Residue-driven self-teaching orchestrator for G4.

The learner ranks known decisions. This teacher converts exact remaining cost
into cross-boundary patterns, executable mutations, predictive classes, proofs,
and certificate-native continuation state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from .capsule import G4Capsule
from .certificates import CertificateRecord, ContinuationState, build_certificate
from .evidence import BoundaryContract, PhysicalEvidence
from .learner import G4Learner
from .mdl import CompleteCost, MergeDecision, PromotionDecision, SplitDecision, evaluate_promotion
from .model import Candidate, Experience, Observation, SearchBoundary
from .mutations import MutationProposal, MutationRegistry, propose_mutations
from .patterns import ExactPattern, discover_exact_patterns
from .predictive import PredictiveClass, PredictiveOutcome, discover_predictive_classes
from .proofs import ProofObject, ProofStore
from .residue import ExactResidue, ResidueLedger, extract_residue
from .supersteps import DeterministicEdge, Superstep, compile_superstep


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class TeachingResult:
    experience: Experience
    residue: ExactResidue | None
    new_patterns: tuple[ExactPattern, ...]
    new_mutation_proposals: tuple[MutationProposal, ...]
    promotion_decision: PromotionDecision | None
    promotion_proof_id: str | None


class ResidueDrivenTeacher:
    STATE_VERSION = 1

    def __init__(
        self,
        *,
        learner: G4Learner | None = None,
        capsule: G4Capsule | None = None,
        minimum_pattern_members: int = 2,
    ):
        if minimum_pattern_members < 2:
            raise ValueError("minimum_pattern_members must be at least two")
        self.capsule = capsule
        self.learner = learner or G4Learner(capsule=capsule)
        if learner is not None and capsule is not None and learner.capsule is not capsule:
            raise ValueError("learner and teacher must share the same capsule")
        self.minimum_pattern_members = minimum_pattern_members
        self.residues = ResidueLedger()
        self.patterns: dict[str, ExactPattern] = {}
        self.mutation_proposals: dict[str, MutationProposal] = {}
        self.emitted_mutations: set[str] = set()
        self.predictive_outcomes: list[PredictiveOutcome] = []
        self.predictive_classes: dict[str, PredictiveClass] = {}
        self.proof_store = ProofStore()
        self.deterministic_edges: dict[str, DeterministicEdge] = {}
        self.supersteps: dict[str, Superstep] = {}
        self.current_rules: dict[str, Candidate] = {}
        self.current_costs: dict[str, CompleteCost] = {}
        self.boundary_contracts: dict[str, BoundaryContract] = {}
        self.physical_evidence_hashes: set[str] = set()
        self.complete_cost_records: list[dict[str, Any]] = []
        self.merge_decisions: list[MergeDecision] = []
        self.split_decisions: list[SplitDecision] = []
        self.dominated_region_proof_ids: set[str] = set()
        if capsule is not None:
            capsule.initialize()

    @staticmethod
    def bind_contract(search_boundary: SearchBoundary, contract: Mapping[str, Any]) -> BoundaryContract:
        return BoundaryContract.create(search_boundary.fingerprint, contract, search_boundary.to_dict())

    def register_incumbent(self, *, boundary: BoundaryContract, candidate: Candidate, cost: CompleteCost) -> None:
        if not cost.exact:
            raise ValueError("an inexact body cannot be installed as the current rule")
        self.boundary_contracts[boundary.boundary_id] = boundary
        self.current_rules[boundary.boundary_id] = candidate
        self.current_costs[boundary.boundary_id] = cost
        self.complete_cost_records.append(cost.to_dict())
        self._checkpoint()

    def _refresh_patterns(self) -> tuple[ExactPattern, ...]:
        discovered = discover_exact_patterns(
            self.residues.unresolved(),
            minimum_members=self.minimum_pattern_members,
        )
        new: list[ExactPattern] = []
        for pattern in discovered:
            superseded = [
                pattern_id
                for pattern_id, existing in self.patterns.items()
                if existing.shared_signature == pattern.shared_signature
                and pattern_id != pattern.pattern_id
            ]
            for pattern_id in superseded:
                del self.patterns[pattern_id]
                for mutation_id, proposal in list(self.mutation_proposals.items()):
                    if proposal.target_pattern_id == pattern_id and mutation_id not in self.emitted_mutations:
                        del self.mutation_proposals[mutation_id]
            if pattern.pattern_id not in self.patterns:
                self.patterns[pattern.pattern_id] = pattern
                new.append(pattern)
        return tuple(new)

    def _propose_for_patterns(
        self,
        patterns: Iterable[ExactPattern],
        *,
        fallback_parent: Candidate,
    ) -> tuple[MutationProposal, ...]:
        new: list[MutationProposal] = []
        for pattern in patterns:
            parent = next(
                (
                    self.current_rules[boundary_id]
                    for boundary_id in pattern.member_boundaries
                    if boundary_id in self.current_rules
                ),
                fallback_parent,
            )
            for proposal in propose_mutations(pattern, parent):
                if proposal.mutation_id not in self.mutation_proposals:
                    self.mutation_proposals[proposal.mutation_id] = proposal
                    new.append(proposal)
        return tuple(new)

    def ingest(
        self,
        *,
        search_boundary: SearchBoundary,
        boundary: BoundaryContract,
        candidate: Candidate,
        observation: Observation,
        physical_evidence: PhysicalEvidence,
        residue_target: float,
        residue_metric: str = "direct_latency_ms",
        residue_objective: str = "external_baseline_latency",
        challenger_cost: CompleteCost | None = None,
        confidence_passed: bool = True,
        parent: Candidate | None = None,
        workspace_ceiling_bytes: int | None = None,
    ) -> TeachingResult:
        if boundary.boundary_id != search_boundary.fingerprint:
            raise ValueError("boundary contract must be bound to the SearchBoundary fingerprint")
        physical_evidence.require_matches(boundary, candidate.execution_fingerprint)
        if physical_evidence.exact != observation.exact:
            raise ValueError("judge observation and physical evidence disagree on exactness")
        self.boundary_contracts[boundary.boundary_id] = boundary

        experience = self.learner.observe(search_boundary, candidate, observation, parent=parent)
        self.physical_evidence_hashes.add(physical_evidence.evidence_hash)

        residue: ExactResidue | None = None
        new_patterns: tuple[ExactPattern, ...] = ()
        new_proposals: tuple[MutationProposal, ...] = ()
        if physical_evidence.exact:
            residue = extract_residue(
                boundary=boundary,
                rule_execution_fingerprint=candidate.execution_fingerprint,
                evidence=physical_evidence,
                metric_name=residue_metric,
                target=residue_target,
                objective=residue_objective,
            )
            self.residues.add(residue)
            new_patterns = self._refresh_patterns()
            new_proposals = self._propose_for_patterns(new_patterns, fallback_parent=candidate)

        promotion_decision: PromotionDecision | None = None
        promotion_proof_id: str | None = None
        incumbent = self.current_costs.get(boundary.boundary_id)
        if challenger_cost is not None:
            self.complete_cost_records.append(challenger_cost.to_dict())
        if incumbent is not None and challenger_cost is not None:
            promotion_decision = evaluate_promotion(
                incumbent=incumbent,
                challenger=challenger_cost,
                confidence_passed=confidence_passed,
                workspace_ceiling_bytes=workspace_ceiling_bytes,
            )
            if experience.decision == "PROMOTED" and promotion_decision.accepted:
                evidence_hash = _digest(
                    {
                        "boundary": boundary.to_dict(),
                        "candidate": candidate.to_dict(),
                        "experience": experience.to_dict(),
                        "incumbent_cost": incumbent.to_dict(),
                        "challenger_cost": challenger_cost.to_dict(),
                        "physical_evidence": physical_evidence.evidence_hash,
                    }
                )
                proof = ProofObject.create(
                    proof_type="exact-rule-promotion",
                    premises=(boundary.contract_hash, physical_evidence.evidence_hash, experience.fingerprint),
                    conclusion=f"{boundary.boundary_id}: install {candidate.execution_fingerprint} as current rule",
                    evidence_hash=evidence_hash,
                    exact=True,
                    metadata={
                        "latency_improvement_ms": promotion_decision.latency_improvement_ms,
                        "description_savings_bits": promotion_decision.description_savings_bits,
                    },
                )
                promotion_proof_id = self.proof_store.add(proof)
                self.current_rules[boundary.boundary_id] = candidate
                self.current_costs[boundary.boundary_id] = challenger_cost
        elif incumbent is None and challenger_cost is not None and challenger_cost.exact:
            self.current_rules[boundary.boundary_id] = candidate
            self.current_costs[boundary.boundary_id] = challenger_cost

        self._checkpoint()
        return TeachingResult(
            experience=experience,
            residue=residue,
            new_patterns=new_patterns,
            new_mutation_proposals=new_proposals,
            promotion_decision=promotion_decision,
            promotion_proof_id=promotion_proof_id,
        )

    def record_predictive_outcome(self, outcome: PredictiveOutcome) -> tuple[PredictiveClass, ...]:
        self.predictive_outcomes.append(outcome)
        discovered = discover_predictive_classes(
            self.predictive_outcomes,
            minimum_members=self.minimum_pattern_members,
        )
        new: list[PredictiveClass] = []
        for predictive_class in discovered:
            if predictive_class.class_id not in self.predictive_classes:
                self.predictive_classes[predictive_class.class_id] = predictive_class
                new.append(predictive_class)
        self._checkpoint()
        return tuple(new)

    def record_merge_decision(self, decision: MergeDecision) -> None:
        self.merge_decisions.append(decision)
        self._checkpoint()

    def record_split_decision(self, decision: SplitDecision) -> None:
        self.split_decisions.append(decision)
        self._checkpoint()

    def install_deterministic_edge(
        self,
        *,
        boundary: BoundaryContract,
        source_state: str,
        destination_state: str,
        proof: ProofObject,
        dominated_region: bool = False,
    ) -> DeterministicEdge:
        if not proof.exact:
            raise ValueError("deterministic conclusions require exact proof")
        proof_id = self.proof_store.add(proof)
        edge = DeterministicEdge(
            source_state=str(source_state),
            destination_state=str(destination_state),
            proof_id=proof_id,
            boundary_contract_hash=boundary.contract_hash,
        )
        self.deterministic_edges[edge.edge_id] = edge
        if dominated_region:
            self.dominated_region_proof_ids.add(proof_id)
        self._checkpoint()
        return edge

    def compile_superstep(
        self,
        *,
        boundary: BoundaryContract,
        edges: Iterable[DeterministicEdge],
    ) -> Superstep:
        superstep = compile_superstep(edges, proof_store=self.proof_store, contract_hash=boundary.contract_hash)
        self.supersteps[superstep.superstep_id] = superstep
        self._checkpoint()
        return superstep

    def generator(self, registry: MutationRegistry):
        """Return a generator compatible with MidRunSupervisor."""

        def generate(
            boundary: SearchBoundary,
            _learner: G4Learner,
            pool: Sequence[Candidate],
        ) -> tuple[Candidate, ...]:
            pool_by_execution = {candidate.execution_fingerprint: candidate for candidate in pool}
            generated_by_execution: dict[str, Candidate] = {}
            for proposal in self.mutation_proposals.values():
                if proposal.mutation_id in self.emitted_mutations:
                    continue
                if boundary.fingerprint not in proposal.target_boundaries:
                    continue
                parent = pool_by_execution.get(proposal.parent_execution_fingerprint)
                if parent is None:
                    continue
                materialized = registry.materialize(proposal, parent)
                if materialized:
                    for candidate in materialized:
                        generated_by_execution.setdefault(candidate.execution_fingerprint, candidate)
                    self.emitted_mutations.add(proposal.mutation_id)
            self._checkpoint()
            return tuple(generated_by_execution.values())

        return generate

    def to_state(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "judge_digest": self.learner.judge.digest,
            "boundary_contracts": {key: value.to_dict() for key, value in sorted(self.boundary_contracts.items())},
            "residues": self.residues.to_state(),
            "patterns": {key: value.to_dict() for key, value in sorted(self.patterns.items())},
            "mutation_proposals": {key: value.to_dict() for key, value in sorted(self.mutation_proposals.items())},
            "emitted_mutations": sorted(self.emitted_mutations),
            "predictive_classes": {key: value.to_dict() for key, value in sorted(self.predictive_classes.items())},
            "proofs": self.proof_store.to_state(),
            "deterministic_edges": {key: value.to_dict() for key, value in sorted(self.deterministic_edges.items())},
            "supersteps": {key: value.to_dict() for key, value in sorted(self.supersteps.items())},
            "current_rules": {key: value.to_dict() for key, value in sorted(self.current_rules.items())},
            "current_costs": {key: value.to_dict() for key, value in sorted(self.current_costs.items())},
            "physical_evidence_hashes": sorted(self.physical_evidence_hashes),
            "complete_cost_records": list(self.complete_cost_records),
            "merge_decisions": [asdict(value) for value in self.merge_decisions],
            "split_decisions": [asdict(value) for value in self.split_decisions],
            "dominated_region_proof_ids": sorted(self.dominated_region_proof_ids),
        }

    def build_certificate(
        self,
        *,
        mode: str,
        cursor: str,
        parent_certificate_id: str | None = None,
        candidate_ledger: Any | None = None,
        coverage_ledger: Any | None = None,
    ) -> CertificateRecord:
        learner_state = self.learner.to_state()
        teacher_state = self.to_state()
        continuation = ContinuationState.create(
            mode=mode,
            cursor=cursor,
            parent_certificate_id=parent_certificate_id,
            learner_state=learner_state,
            teacher_state=teacher_state,
            pending_search_objects=(
                proposal_id
                for proposal_id in self.mutation_proposals
                if proposal_id not in self.emitted_mutations
            ),
            completed_execution_keys=self.learner.evaluated,
        )
        champions = {
            boundary_id: candidate.execution_fingerprint
            for boundary_id, candidate in self.current_rules.items()
        }
        certificate = build_certificate(
            judge_digest=self.learner.judge.digest,
            boundary_contracts={key: value.contract_hash for key, value in self.boundary_contracts.items()},
            candidate_ledger=(candidate_ledger if candidate_ledger is not None else learner_state),
            coverage_ledger=(coverage_ledger if coverage_ledger is not None else sorted(self.learner.evaluated)),
            champions=champions,
            correctness_evidence_hashes=(proof.evidence_hash for proof in self.proof_store.all() if proof.exact),
            physical_measurement_hashes=self.physical_evidence_hashes,
            complete_cost_records=self.complete_cost_records,
            unresolved_residue_ids=(residue.residue_id for residue in self.residues.unresolved()),
            pattern_ids=self.patterns,
            predictive_class_ids=self.predictive_classes,
            merge_decisions=(asdict(value) for value in self.merge_decisions),
            split_decisions=(asdict(value) for value in self.split_decisions),
            dominated_region_proof_ids=self.dominated_region_proof_ids,
            mutation_proposal_ids=self.mutation_proposals,
            deterministic_edge_ids=self.deterministic_edges,
            superstep_ids=self.supersteps,
            proof_ids=(proof.proof_id for proof in self.proof_store.all()),
            continuation=continuation,
        )
        if self.capsule is not None:
            certificate.save(self.capsule)
        return certificate

    def _checkpoint(self) -> None:
        if self.capsule is None:
            return
        self.capsule.save_state(self.to_state(), name="master_key_teacher")


__all__ = ["TeachingResult", "ResidueDrivenTeacher"]
