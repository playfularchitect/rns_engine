"""Exact cross-boundary residue pattern discovery."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

from .residue import ExactResidue


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def describe_signature(signature: tuple[tuple[str, Any], ...]) -> str:
    values = dict(signature)
    labels: list[str] = []
    if values.get("launch_band") in {"high", "dominant"}:
        labels.append("launch-bound")
    if values.get("padding_band") in {"high", "dominant"}:
        labels.append("tile-waste")
    if values.get("reduction_band") in {"high", "dominant"}:
        labels.append("reduction-bound")
    if values.get("memory_band") in {"high", "dominant"}:
        labels.append("memory-bound")
    if not labels:
        labels.extend(f"{key}={value}" for key, value in signature)
    return " + ".join(labels)


@dataclass(frozen=True, slots=True)
class ExactPattern:
    pattern_id: str
    member_boundaries: tuple[str, ...]
    member_residue_ids: tuple[str, ...]
    shared_signature: tuple[tuple[str, Any], ...]
    evidence_hash: str
    proposed_distinction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "member_boundaries": list(self.member_boundaries),
            "member_residue_ids": list(self.member_residue_ids),
            "shared_signature": dict(self.shared_signature),
            "evidence_hash": self.evidence_hash,
            "proposed_distinction": self.proposed_distinction,
        }


def discover_exact_patterns(
    residues: Iterable[ExactResidue],
    *,
    minimum_members: int = 2,
) -> list[ExactPattern]:
    if minimum_members < 2:
        raise ValueError("minimum_members must be at least two")
    groups: dict[tuple[tuple[str, Any], ...], list[ExactResidue]] = defaultdict(list)
    for residue in residues:
        if residue.unresolved:
            groups[residue.signature].append(residue)

    patterns: list[ExactPattern] = []
    for signature, members in groups.items():
        unique_boundaries = sorted({member.boundary_id for member in members})
        if len(unique_boundaries) < minimum_members:
            continue
        ordered = sorted(members, key=lambda item: (item.boundary_id, item.residue_id))
        evidence_payload = [
            {
                "boundary_id": item.boundary_id,
                "rule_execution_fingerprint": item.rule_execution_fingerprint,
                "amount": item.amount,
                "evidence_hash": item.exact_evidence_hash,
            }
            for item in ordered
        ]
        evidence_hash = _digest(evidence_payload)
        residue_ids = tuple(item.residue_id for item in ordered)
        payload = {
            "signature": dict(signature),
            "boundaries": unique_boundaries,
            "residue_ids": residue_ids,
            "evidence_hash": evidence_hash,
        }
        patterns.append(
            ExactPattern(
                pattern_id=_digest(payload),
                member_boundaries=tuple(unique_boundaries),
                member_residue_ids=residue_ids,
                shared_signature=signature,
                evidence_hash=evidence_hash,
                proposed_distinction=describe_signature(signature),
            )
        )
    patterns.sort(key=lambda pattern: (-len(pattern.member_boundaries), pattern.pattern_id))
    return patterns


__all__ = ["ExactPattern", "discover_exact_patterns", "describe_signature"]
