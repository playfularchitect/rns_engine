"""Proof-guarded deterministic edges and exact search supersteps."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

from .proofs import ProofObject, ProofStore


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class DeterministicEdge:
    source_state: str
    destination_state: str
    proof_id: str
    boundary_contract_hash: str

    @property
    def edge_id(self) -> str:
        return _digest(
            {
                "source": self.source_state,
                "destination": self.destination_state,
                "proof_id": self.proof_id,
                "boundary_contract_hash": self.boundary_contract_hash,
            }
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "edge_id": self.edge_id,
            "source_state": self.source_state,
            "destination_state": self.destination_state,
            "proof_id": self.proof_id,
            "boundary_contract_hash": self.boundary_contract_hash,
        }


def deterministic_edge_is_valid(
    edge: DeterministicEdge,
    *,
    current_contract_hash: str,
    proof_store: ProofStore,
) -> bool:
    proof = proof_store.get(edge.proof_id)
    return bool(
        proof is not None
        and proof.exact
        and edge.boundary_contract_hash == current_contract_hash
    )


@dataclass(frozen=True, slots=True)
class Superstep:
    superstep_id: str
    start_state: str
    end_state: str
    internal_edges: tuple[str, ...]
    proof_id: str
    boundary_contract_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "superstep_id": self.superstep_id,
            "start_state": self.start_state,
            "end_state": self.end_state,
            "internal_edges": list(self.internal_edges),
            "proof_id": self.proof_id,
            "boundary_contract_hash": self.boundary_contract_hash,
        }


def compile_superstep(
    edges: Iterable[DeterministicEdge],
    *,
    proof_store: ProofStore,
    contract_hash: str,
) -> Superstep:
    ordered = tuple(edges)
    if not ordered:
        raise ValueError("cannot compile an empty deterministic path")
    for index, edge in enumerate(ordered):
        if not deterministic_edge_is_valid(
            edge,
            current_contract_hash=contract_hash,
            proof_store=proof_store,
        ):
            raise ValueError("path contains an unproved or stale deterministic edge")
        if index and ordered[index - 1].destination_state != edge.source_state:
            raise ValueError("deterministic edges do not form a continuous path")
    start = ordered[0].source_state
    end = ordered[-1].destination_state
    edge_ids = tuple(edge.edge_id for edge in ordered)
    evidence_hash = _digest(
        {
            "start": start,
            "end": end,
            "edge_ids": list(edge_ids),
            "proofs": [edge.proof_id for edge in ordered],
            "contract_hash": contract_hash,
        }
    )
    proof = ProofObject.create(
        proof_type="exact-superstep",
        premises=tuple(edge.proof_id for edge in ordered),
        conclusion=f"{start}->{end}",
        evidence_hash=evidence_hash,
        exact=True,
        metadata={"contract_hash": contract_hash, "edge_count": len(ordered)},
    )
    proof_store.add(proof)
    return Superstep(
        superstep_id=_digest({"start": start, "end": end, "edge_ids": list(edge_ids)}),
        start_state=start,
        end_state=end,
        internal_edges=edge_ids,
        proof_id=proof.proof_id,
        boundary_contract_hash=contract_hash,
    )


__all__ = [
    "DeterministicEdge",
    "Superstep",
    "deterministic_edge_is_valid",
    "compile_superstep",
]
