"""Residue-directed mutation proposals that can become executable search objects."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Iterable

from .model import Candidate
from .patterns import ExactPattern


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class MutationProposal:
    mutation_id: str
    parent_execution_fingerprint: str
    target_pattern_id: str
    target_boundaries: tuple[str, ...]
    operation: str
    parameters: tuple[tuple[str, Any], ...]

    @property
    def search_key(self) -> str:
        return self.mutation_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "parent_execution_fingerprint": self.parent_execution_fingerprint,
            "target_pattern_id": self.target_pattern_id,
            "target_boundaries": list(self.target_boundaries),
            "operation": self.operation,
            "parameters": dict(self.parameters),
        }


def propose_mutations(pattern: ExactPattern, parent: Candidate) -> list[MutationProposal]:
    signature = dict(pattern.shared_signature)
    raw: list[tuple[str, dict[str, Any]]] = []

    def add(operation: str, **parameters: Any) -> None:
        raw.append((operation, parameters))

    if signature.get("launch_band") in {"high", "dominant"}:
        add("increase_work_per_launch")
        add("group_boundaries")
        add("persistent_dispatch")
    if signature.get("padding_band") in {"high", "dominant"}:
        add("change_tile_geometry")
        add("add_exact_edge_tile")
    if signature.get("reduction_band") in {"high", "dominant"}:
        add("change_reduction_structure")
        add("fuse_reconstruction")
    if signature.get("memory_band") in {"high", "dominant"}:
        add("reduce_materialization")
        add("change_layout")
        add("increase_reuse")
    if not raw:
        add("generic_local_mutation", distinction=pattern.proposed_distinction)

    proposals: list[MutationProposal] = []
    for operation, parameters in raw:
        parameter_pairs = tuple(sorted((str(key), value) for key, value in parameters.items()))
        payload = {
            "parent": parent.execution_fingerprint,
            "pattern": pattern.pattern_id,
            "boundaries": list(pattern.member_boundaries),
            "operation": operation,
            "parameters": dict(parameter_pairs),
        }
        proposals.append(
            MutationProposal(
                mutation_id=_digest(payload),
                parent_execution_fingerprint=parent.execution_fingerprint,
                target_pattern_id=pattern.pattern_id,
                target_boundaries=pattern.member_boundaries,
                operation=operation,
                parameters=parameter_pairs,
            )
        )
    return proposals


MutationOperator = Callable[[Candidate, MutationProposal], Candidate | Iterable[Candidate]]


class MutationRegistry:
    """Maps residue-derived operations to domain-specific executable mutations."""

    def __init__(self):
        self._operators: dict[str, MutationOperator] = {}

    def register(self, operation: str, operator: MutationOperator) -> None:
        if not operation:
            raise ValueError("mutation operation cannot be empty")
        self._operators[str(operation)] = operator

    def materialize(self, proposal: MutationProposal, parent: Candidate) -> tuple[Candidate, ...]:
        if parent.execution_fingerprint != proposal.parent_execution_fingerprint:
            raise ValueError("mutation parent does not match proposal")
        operator = self._operators.get(proposal.operation)
        if operator is None:
            return ()
        result = operator(parent, proposal)
        if isinstance(result, Candidate):
            return (result,)
        return tuple(result)

    def supported_operations(self) -> tuple[str, ...]:
        return tuple(sorted(self._operators))


__all__ = [
    "MutationProposal",
    "MutationRegistry",
    "MutationOperator",
    "propose_mutations",
]
