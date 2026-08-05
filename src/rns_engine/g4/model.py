"""Canonical boundary, candidate, observation, and pattern models."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Iterable, Mapping

from .actg import candidate_genome, canonical_json


def _pairs(values: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> tuple[tuple[str, Any], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    return tuple(sorted((str(key), value) for key, value in items))


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SearchBoundary:
    environment: str
    domain: str
    operation: str
    constraints: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def create(
        cls,
        *,
        environment: str,
        domain: str,
        operation: str,
        constraints: Mapping[str, Any] | Iterable[tuple[str, Any]] = (),
    ) -> "SearchBoundary":
        return cls(environment, domain, operation, _pairs(constraints))

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict())

    def context_tokens(self) -> tuple[str, ...]:
        tokens = [
            "global",
            f"environment:{self.environment}",
            f"domain:{self.domain}",
            f"operation:{self.operation}",
            f"environment_domain:{self.environment}|{self.domain}",
            f"domain_operation:{self.domain}|{self.operation}",
        ]
        tokens.extend(f"constraint:{key}={value}" for key, value in self.constraints)
        return tuple(tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "domain": self.domain,
            "operation": self.operation,
            "constraints": dict(self.constraints),
        }


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    features: tuple[str, ...]
    parameters: tuple[tuple[str, Any], ...] = ()
    mutation_ops: tuple[str, ...] = ()
    parent_id: str | None = None
    description_cost: float = 0.0
    expected_work_cost: float = 0.0
    expected_memory_cost: float = 0.0
    genome: str = field(default="")

    @classmethod
    def create(
        cls,
        candidate_id: str,
        *,
        features: Iterable[str],
        parameters: Mapping[str, Any] | Iterable[tuple[str, Any]] = (),
        mutation_ops: Iterable[str] = (),
        parent_id: str | None = None,
        description_cost: float = 0.0,
        expected_work_cost: float = 0.0,
        expected_memory_cost: float = 0.0,
    ) -> "Candidate":
        feature_tuple = tuple(sorted(set(str(feature) for feature in features)))
        parameter_pairs = _pairs(parameters)
        mutations = tuple(str(item) for item in mutation_ops)
        genome = candidate_genome(
            features=feature_tuple,
            parameters=dict(parameter_pairs),
            mutation_ops=mutations,
        )
        return cls(
            candidate_id=str(candidate_id),
            features=feature_tuple,
            parameters=parameter_pairs,
            mutation_ops=mutations,
            parent_id=parent_id,
            description_cost=float(description_cost),
            expected_work_cost=float(expected_work_cost),
            expected_memory_cost=float(expected_memory_cost),
            genome=genome,
        )

    @property
    def fingerprint(self) -> str:
        """Genotype identity including mutation history."""
        return _digest({"genome": self.genome, "parent_id": self.parent_id})

    @property
    def execution_fingerprint(self) -> str:
        """Phenotype identity used by the exact no-repeat ledger."""
        return _digest(
            {
                "features": list(self.features),
                "parameters": dict(self.parameters),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "features": list(self.features),
            "parameters": dict(self.parameters),
            "mutation_ops": list(self.mutation_ops),
            "parent_id": self.parent_id,
            "description_cost": self.description_cost,
            "expected_work_cost": self.expected_work_cost,
            "expected_memory_cost": self.expected_memory_cost,
            "genome": self.genome,
            "fingerprint": self.fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class Observation:
    compile_ok: bool
    legal: bool
    exact: bool
    speedup: float
    confidence_lower: float
    wins: int
    blocks: int
    elapsed_seconds: float = 0.0
    actual_memory_cost: float = 0.0
    actual_work_cost: float = 0.0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "compile_ok": self.compile_ok,
            "legal": self.legal,
            "exact": self.exact,
            "speedup": self.speedup,
            "confidence_lower": self.confidence_lower,
            "wins": self.wins,
            "blocks": self.blocks,
            "elapsed_seconds": self.elapsed_seconds,
            "actual_memory_cost": self.actual_memory_cost,
            "actual_work_cost": self.actual_work_cost,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class Experience:
    boundary: SearchBoundary
    candidate: Candidate
    observation: Observation
    decision: str
    reward: float
    sequence: int

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "boundary": self.boundary.fingerprint,
                "candidate": self.candidate.fingerprint,
                "observation": self.observation.to_dict(),
                "decision": self.decision,
                "sequence": self.sequence,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary.to_dict(),
            "boundary_fingerprint": self.boundary.fingerprint,
            "candidate": self.candidate.to_dict(),
            "observation": self.observation.to_dict(),
            "decision": self.decision,
            "reward": self.reward,
            "sequence": self.sequence,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ResiduePattern:
    boundary_fingerprint: str
    context_token: str
    feature: str
    residue: str
    support: int
    rate: float
    suggested_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_fingerprint": self.boundary_fingerprint,
            "context_token": self.context_token,
            "feature": self.feature,
            "residue": self.residue,
            "support": self.support,
            "rate": self.rate,
            "suggested_action": self.suggested_action,
        }


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: Candidate
    score: float
    evidence_score: float
    exploration_bonus: float
    cost_penalty: float


__all__ = [
    "SearchBoundary",
    "Candidate",
    "Observation",
    "Experience",
    "ResiduePattern",
    "RankedCandidate",
]
