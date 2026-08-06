"""Exact predictive equivalence classes with restoring residue."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping


def _pairs(values: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> tuple[tuple[str, Any], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    return tuple(sorted((str(key), value) for key, value in items))


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PredictiveOutcome:
    boundary_id: str
    history_signature: tuple[tuple[str, Any], ...]
    continuation_signature: tuple[str, ...]
    conclusion_signature: tuple[tuple[str, Any], ...]
    exact_evidence_hash: str

    @classmethod
    def create(
        cls,
        *,
        boundary_id: str,
        history_signature: Mapping[str, Any] | Iterable[tuple[str, Any]],
        continuation_signature: Iterable[str],
        conclusion_signature: Mapping[str, Any] | Iterable[tuple[str, Any]],
        exact_evidence_hash: str,
    ) -> "PredictiveOutcome":
        return cls(
            boundary_id=str(boundary_id),
            history_signature=_pairs(history_signature),
            continuation_signature=tuple(str(item) for item in continuation_signature),
            conclusion_signature=_pairs(conclusion_signature),
            exact_evidence_hash=str(exact_evidence_hash),
        )


@dataclass(frozen=True, slots=True)
class PredictiveClass:
    class_id: str
    member_boundaries: tuple[str, ...]
    continuation_signature: tuple[str, ...]
    conclusion_signature: tuple[tuple[str, Any], ...]
    restoring_history_residue: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...]
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "member_boundaries": list(self.member_boundaries),
            "continuation_signature": list(self.continuation_signature),
            "conclusion_signature": dict(self.conclusion_signature),
            "restoring_history_residue": {
                boundary: dict(signature)
                for boundary, signature in self.restoring_history_residue
            },
            "evidence_hash": self.evidence_hash,
        }


def discover_predictive_classes(
    outcomes: Iterable[PredictiveOutcome],
    *,
    minimum_members: int = 2,
) -> list[PredictiveClass]:
    groups: dict[
        tuple[tuple[str, ...], tuple[tuple[str, Any], ...]],
        list[PredictiveOutcome],
    ] = defaultdict(list)
    for outcome in outcomes:
        groups[(outcome.continuation_signature, outcome.conclusion_signature)].append(outcome)

    classes: list[PredictiveClass] = []
    for (continuation, conclusion), members in groups.items():
        unique = {member.boundary_id: member for member in members}
        if len(unique) < minimum_members:
            continue
        ordered = [unique[key] for key in sorted(unique)]
        restoring = tuple((member.boundary_id, member.history_signature) for member in ordered)
        evidence_hash = _digest(
            [
                {
                    "boundary": member.boundary_id,
                    "evidence": member.exact_evidence_hash,
                    "history": dict(member.history_signature),
                }
                for member in ordered
            ]
        )
        payload = {
            "boundaries": [member.boundary_id for member in ordered],
            "continuation": list(continuation),
            "conclusion": dict(conclusion),
            "restoring": {boundary: dict(signature) for boundary, signature in restoring},
            "evidence_hash": evidence_hash,
        }
        classes.append(
            PredictiveClass(
                class_id=_digest(payload),
                member_boundaries=tuple(member.boundary_id for member in ordered),
                continuation_signature=continuation,
                conclusion_signature=conclusion,
                restoring_history_residue=restoring,
                evidence_hash=evidence_hash,
            )
        )
    classes.sort(key=lambda item: (-len(item.member_boundaries), item.class_id))
    return classes


__all__ = ["PredictiveOutcome", "PredictiveClass", "discover_predictive_classes"]
