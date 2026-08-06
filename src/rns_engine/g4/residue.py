"""First-class exact residue objects and ledgers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable

from .evidence import BoundaryContract, PhysicalEvidence


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ExactResidue:
    boundary_id: str
    boundary_contract_hash: str
    rule_execution_fingerprint: str
    objective: str
    amount: float
    measured: float
    target: float
    signature: tuple[tuple[str, Any], ...]
    exact_evidence_hash: str
    residue_id: str

    @property
    def unresolved(self) -> bool:
        return self.amount > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "boundary_contract_hash": self.boundary_contract_hash,
            "rule_execution_fingerprint": self.rule_execution_fingerprint,
            "objective": self.objective,
            "amount": self.amount,
            "measured": self.measured,
            "target": self.target,
            "signature": dict(self.signature),
            "exact_evidence_hash": self.exact_evidence_hash,
            "residue_id": self.residue_id,
        }


def extract_residue(
    *,
    boundary: BoundaryContract,
    rule_execution_fingerprint: str,
    evidence: PhysicalEvidence,
    metric_name: str,
    target: float,
    objective: str,
) -> ExactResidue:
    """Preserve the exact unexplained remainder above a declared target.

    Incorrect candidates are not assigned an attractive or infinite score. They
    fail admission and cannot create lawful residue patterns.
    """

    evidence.require_matches(boundary, rule_execution_fingerprint)
    if not evidence.exact:
        raise ValueError("inexact evidence fails admission and cannot form exact residue")
    if not math.isfinite(target) or target < 0:
        raise ValueError("residue target must be finite and nonnegative")
    measured = evidence.metric(metric_name)
    if not math.isfinite(measured) or measured < 0:
        raise ValueError("measured physical cost must be finite and nonnegative")
    amount = max(0.0, measured - target)
    payload = {
        "boundary_id": boundary.boundary_id,
        "boundary_contract_hash": boundary.contract_hash,
        "rule_execution_fingerprint": rule_execution_fingerprint,
        "objective": objective,
        "amount": amount,
        "measured": measured,
        "target": target,
        "signature": dict(evidence.signature),
        "exact_evidence_hash": evidence.evidence_hash,
    }
    return ExactResidue(
        boundary_id=boundary.boundary_id,
        boundary_contract_hash=boundary.contract_hash,
        rule_execution_fingerprint=rule_execution_fingerprint,
        objective=str(objective),
        amount=amount,
        measured=measured,
        target=target,
        signature=evidence.signature,
        exact_evidence_hash=evidence.evidence_hash,
        residue_id=_digest(payload),
    )


def external_baseline_residue(
    *,
    boundary: BoundaryContract,
    rule_execution_fingerprint: str,
    evidence: PhysicalEvidence,
    baseline_latency_ms: float,
) -> ExactResidue:
    return extract_residue(
        boundary=boundary,
        rule_execution_fingerprint=rule_execution_fingerprint,
        evidence=evidence,
        metric_name="direct_latency_ms",
        target=baseline_latency_ms,
        objective="external_baseline_latency",
    )


def physical_floor_residue(
    *,
    boundary: BoundaryContract,
    rule_execution_fingerprint: str,
    evidence: PhysicalEvidence,
    physical_lower_bound_ms: float,
) -> ExactResidue:
    return extract_residue(
        boundary=boundary,
        rule_execution_fingerprint=rule_execution_fingerprint,
        evidence=evidence,
        metric_name="direct_latency_ms",
        target=physical_lower_bound_ms,
        objective="physical_lower_bound_latency",
    )


class ResidueLedger:
    """Content-addressed exact residue memory."""

    def __init__(self, residues: Iterable[ExactResidue] = ()):
        self._residues: dict[str, ExactResidue] = {}
        for residue in residues:
            self.add(residue)

    def add(self, residue: ExactResidue) -> bool:
        is_new = residue.residue_id not in self._residues
        self._residues.setdefault(residue.residue_id, residue)
        return is_new

    def unresolved(self) -> tuple[ExactResidue, ...]:
        return tuple(residue for residue in self._residues.values() if residue.unresolved)

    def all(self) -> tuple[ExactResidue, ...]:
        return tuple(self._residues.values())

    def to_state(self) -> dict[str, Any]:
        return {key: value.to_dict() for key, value in sorted(self._residues.items())}


__all__ = [
    "ExactResidue",
    "ResidueLedger",
    "extract_residue",
    "external_baseline_residue",
    "physical_floor_residue",
]
