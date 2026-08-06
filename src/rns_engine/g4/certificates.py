"""Certificate-native continuation records for G4 Timed and Lazy modes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

from .actg import object_to_actg
from .capsule import G4Capsule


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(str(value) for value in values)))


@dataclass(frozen=True, slots=True)
class ContinuationState:
    mode: str
    cursor: str
    parent_certificate_id: str | None
    learner_state_hash: str
    teacher_state_hash: str
    pending_search_objects: tuple[str, ...]
    completed_execution_keys: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        mode: str,
        cursor: str,
        parent_certificate_id: str | None,
        learner_state: Mapping[str, Any],
        teacher_state: Mapping[str, Any],
        pending_search_objects: Iterable[str] = (),
        completed_execution_keys: Iterable[str] = (),
    ) -> "ContinuationState":
        normalized = mode.strip().lower()
        if normalized not in {"timed", "lazy"}:
            raise ValueError("continuation mode must be 'timed' or 'lazy'")
        return cls(
            mode=normalized,
            cursor=str(cursor),
            parent_certificate_id=parent_certificate_id,
            learner_state_hash=_digest(dict(learner_state)),
            teacher_state_hash=_digest(dict(teacher_state)),
            pending_search_objects=_sorted_unique(pending_search_objects),
            completed_execution_keys=_sorted_unique(completed_execution_keys),
        )

    @property
    def continuation_id(self) -> str:
        return _digest(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        output = {
            "mode": self.mode,
            "cursor": self.cursor,
            "parent_certificate_id": self.parent_certificate_id,
            "learner_state_hash": self.learner_state_hash,
            "teacher_state_hash": self.teacher_state_hash,
            "pending_search_objects": list(self.pending_search_objects),
            "completed_execution_keys": list(self.completed_execution_keys),
        }
        if include_id:
            output["continuation_id"] = self.continuation_id
        return output


@dataclass(frozen=True, slots=True)
class CertificateRecord:
    schema: str
    judge_digest: str
    frozen_boundary_contracts: tuple[tuple[str, str], ...]
    candidate_ledger_hash: str
    coverage_ledger_hash: str
    champions: tuple[tuple[str, str], ...]
    correctness_evidence_hashes: tuple[str, ...]
    physical_measurement_hashes: tuple[str, ...]
    complete_cost_hashes: tuple[str, ...]
    unresolved_residue_ids: tuple[str, ...]
    pattern_ids: tuple[str, ...]
    predictive_class_ids: tuple[str, ...]
    merge_decision_hashes: tuple[str, ...]
    split_decision_hashes: tuple[str, ...]
    dominated_region_proof_ids: tuple[str, ...]
    mutation_proposal_ids: tuple[str, ...]
    deterministic_edge_ids: tuple[str, ...]
    superstep_ids: tuple[str, ...]
    proof_ids: tuple[str, ...]
    continuation: ContinuationState

    @property
    def certificate_id(self) -> str:
        return _digest(self.to_dict(include_id=False))

    @property
    def actg_stamp(self) -> str:
        return object_to_actg(
            {
                "certificate_id": self.certificate_id,
                "judge_digest": self.judge_digest,
                "continuation_id": self.continuation.continuation_id,
            }
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        output = {
            "schema": self.schema,
            "judge_digest": self.judge_digest,
            "frozen_boundary_contracts": dict(self.frozen_boundary_contracts),
            "candidate_ledger_hash": self.candidate_ledger_hash,
            "coverage_ledger_hash": self.coverage_ledger_hash,
            "champions": dict(self.champions),
            "correctness_evidence_hashes": list(self.correctness_evidence_hashes),
            "physical_measurement_hashes": list(self.physical_measurement_hashes),
            "complete_cost_hashes": list(self.complete_cost_hashes),
            "unresolved_residue_ids": list(self.unresolved_residue_ids),
            "pattern_ids": list(self.pattern_ids),
            "predictive_class_ids": list(self.predictive_class_ids),
            "merge_decision_hashes": list(self.merge_decision_hashes),
            "split_decision_hashes": list(self.split_decision_hashes),
            "dominated_region_proof_ids": list(self.dominated_region_proof_ids),
            "mutation_proposal_ids": list(self.mutation_proposal_ids),
            "deterministic_edge_ids": list(self.deterministic_edge_ids),
            "superstep_ids": list(self.superstep_ids),
            "proof_ids": list(self.proof_ids),
            "continuation": self.continuation.to_dict(),
        }
        if include_id:
            output["certificate_id"] = self.certificate_id
            output["actg_stamp"] = self.actg_stamp
        return output

    def save(self, capsule: G4Capsule, *, name: str = "master_key_certificate") -> str:
        return capsule.save_state(self.to_dict(), name=name)


def build_certificate(
    *,
    judge_digest: str,
    boundary_contracts: Mapping[str, str] | Iterable[tuple[str, str]],
    candidate_ledger: Any,
    coverage_ledger: Any,
    champions: Mapping[str, str] | Iterable[tuple[str, str]],
    correctness_evidence_hashes: Iterable[str],
    physical_measurement_hashes: Iterable[str],
    complete_cost_records: Iterable[Any],
    unresolved_residue_ids: Iterable[str],
    pattern_ids: Iterable[str],
    predictive_class_ids: Iterable[str],
    merge_decisions: Iterable[Any],
    split_decisions: Iterable[Any],
    dominated_region_proof_ids: Iterable[str],
    mutation_proposal_ids: Iterable[str],
    deterministic_edge_ids: Iterable[str],
    superstep_ids: Iterable[str],
    proof_ids: Iterable[str],
    continuation: ContinuationState,
) -> CertificateRecord:
    boundary_items = boundary_contracts.items() if isinstance(boundary_contracts, Mapping) else boundary_contracts
    champion_items = champions.items() if isinstance(champions, Mapping) else champions
    return CertificateRecord(
        schema="rns_engine.g4_master_key_certificate.v1",
        judge_digest=str(judge_digest),
        frozen_boundary_contracts=tuple(sorted((str(k), str(v)) for k, v in boundary_items)),
        candidate_ledger_hash=_digest(candidate_ledger),
        coverage_ledger_hash=_digest(coverage_ledger),
        champions=tuple(sorted((str(k), str(v)) for k, v in champion_items)),
        correctness_evidence_hashes=_sorted_unique(correctness_evidence_hashes),
        physical_measurement_hashes=_sorted_unique(physical_measurement_hashes),
        complete_cost_hashes=_sorted_unique(_digest(value) for value in complete_cost_records),
        unresolved_residue_ids=_sorted_unique(unresolved_residue_ids),
        pattern_ids=_sorted_unique(pattern_ids),
        predictive_class_ids=_sorted_unique(predictive_class_ids),
        merge_decision_hashes=_sorted_unique(_digest(value) for value in merge_decisions),
        split_decision_hashes=_sorted_unique(_digest(value) for value in split_decisions),
        dominated_region_proof_ids=_sorted_unique(dominated_region_proof_ids),
        mutation_proposal_ids=_sorted_unique(mutation_proposal_ids),
        deterministic_edge_ids=_sorted_unique(deterministic_edge_ids),
        superstep_ids=_sorted_unique(superstep_ids),
        proof_ids=_sorted_unique(proof_ids),
        continuation=continuation,
    )


__all__ = ["ContinuationState", "CertificateRecord", "build_certificate"]
