"""Content-addressed exact proof objects and ACTG proof stamps."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

from .actg import object_to_actg


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ProofObject:
    proof_type: str
    premises: tuple[str, ...]
    conclusion: str
    evidence_hash: str
    exact: bool
    metadata: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def create(
        cls,
        *,
        proof_type: str,
        premises: Iterable[str],
        conclusion: str,
        evidence_hash: str,
        exact: bool,
        metadata: Mapping[str, Any] | Iterable[tuple[str, Any]] = (),
    ) -> "ProofObject":
        items = metadata.items() if isinstance(metadata, Mapping) else metadata
        return cls(
            proof_type=str(proof_type),
            premises=tuple(str(item) for item in premises),
            conclusion=str(conclusion),
            evidence_hash=str(evidence_hash),
            exact=bool(exact),
            metadata=tuple(sorted((str(key), value) for key, value in items)),
        )

    @property
    def proof_id(self) -> str:
        return _digest(self.to_dict(include_id=False))

    @property
    def actg_stamp(self) -> str:
        return object_to_actg({"proof_id": self.proof_id, "exact": self.exact})

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        output = {
            "proof_type": self.proof_type,
            "premises": list(self.premises),
            "conclusion": self.conclusion,
            "evidence_hash": self.evidence_hash,
            "exact": self.exact,
            "metadata": dict(self.metadata),
        }
        if include_id:
            output["proof_id"] = self.proof_id
            output["actg_stamp"] = self.actg_stamp
        return output


class ProofStore:
    def __init__(self):
        self._proofs: dict[str, ProofObject] = {}

    def add(self, proof: ProofObject) -> str:
        existing = self._proofs.get(proof.proof_id)
        if existing is not None and existing != proof:
            raise RuntimeError("proof hash collision")
        self._proofs[proof.proof_id] = proof
        return proof.proof_id

    def get(self, proof_id: str) -> ProofObject | None:
        return self._proofs.get(proof_id)

    def require_exact(self, proof_id: str) -> ProofObject:
        proof = self._proofs.get(proof_id)
        if proof is None:
            raise KeyError(f"unknown proof: {proof_id}")
        if not proof.exact:
            raise ValueError(f"proof is not exact: {proof_id}")
        return proof

    def all(self) -> tuple[ProofObject, ...]:
        return tuple(self._proofs[key] for key in sorted(self._proofs))

    def to_state(self) -> dict[str, Any]:
        return {proof.proof_id: proof.to_dict() for proof in self.all()}


__all__ = ["ProofObject", "ProofStore"]
