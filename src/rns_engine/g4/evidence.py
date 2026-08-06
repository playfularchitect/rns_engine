"""Certificate-native boundary and physical evidence records for G4.

This module follows the Master Key law that reality establishes the boundary
and that residue must retain exact measured evidence, not merely a score.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping


def _pairs(values: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> tuple[tuple[str, Any], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    return tuple(sorted((str(key), value) for key, value in items))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def size_band(value: int) -> str:
    if value < 0:
        raise ValueError("size values must be nonnegative")
    if value == 0:
        return "zero"
    if value <= 16:
        return "tiny"
    if value <= 64:
        return "small"
    if value <= 256:
        return "medium"
    if value <= 1024:
        return "large"
    if value <= 4096:
        return "very_large"
    if value <= 16384:
        return "huge"
    return "extreme"


def ratio_band(value: float) -> str:
    if not math.isfinite(value) or value < 0:
        raise ValueError("ratio values must be finite and nonnegative")
    if value == 0:
        return "none"
    if value < 0.10:
        return "low"
    if value < 0.35:
        return "medium"
    if value < 0.70:
        return "high"
    return "dominant"


def aspect_band(m: int, n: int) -> str:
    if m <= 0 or n <= 0:
        raise ValueError("matrix dimensions must be positive")
    ratio = max(m, n) / min(m, n)
    if ratio < 1.5:
        return "square"
    if ratio < 4:
        return "rectangular"
    if m > n:
        return "tall"
    return "wide"


@dataclass(frozen=True, slots=True)
class BoundaryContract:
    """The immutable world in which one claim must hold."""

    boundary_id: str
    contract_hash: str
    features: tuple[tuple[str, Any], ...]

    @classmethod
    def create(
        cls,
        boundary_id: str,
        contract: bytes | str | Mapping[str, Any],
        features: Mapping[str, Any] | Iterable[tuple[str, Any]],
    ) -> "BoundaryContract":
        if isinstance(contract, bytes):
            payload = contract
        elif isinstance(contract, str):
            payload = contract.encode("utf-8")
        else:
            payload = _canonical(dict(contract))
        return cls(
            boundary_id=str(boundary_id),
            contract_hash=hashlib.sha256(payload).hexdigest(),
            features=_pairs(features),
        )

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "contract_hash": self.contract_hash,
            "features": dict(self.features),
        }


@dataclass(frozen=True, slots=True)
class PhysicalEvidence:
    """Exact correctness and physical measurements for one executable rule."""

    boundary_id: str
    boundary_contract_hash: str
    rule_execution_fingerprint: str
    exact: bool
    metrics: tuple[tuple[str, float], ...]
    signature: tuple[tuple[str, Any], ...]
    failure_tags: tuple[str, ...]
    evidence_hash: str

    @classmethod
    def create(
        cls,
        *,
        boundary: BoundaryContract,
        rule_execution_fingerprint: str,
        exact: bool,
        metrics: Mapping[str, float] | Iterable[tuple[str, float]],
        signature: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
        shape: tuple[int, int, int] | None = None,
        launch_fraction: float | None = None,
        padding_fraction: float | None = None,
        memory_fraction: float | None = None,
        reduction_fraction: float | None = None,
        workspace_bytes: int | None = None,
        failure_tags: Iterable[str] = (),
    ) -> "PhysicalEvidence":
        metric_pairs = tuple(sorted((str(key), float(value)) for key, value in (
            metrics.items() if isinstance(metrics, Mapping) else metrics
        )))
        signature_map: dict[str, Any] = {}
        if signature is not None:
            signature_map.update(dict(signature))
        if shape is not None:
            m, n, k = shape
            signature_map.update(
                {
                    "m_band": size_band(m),
                    "n_band": size_band(n),
                    "k_band": size_band(k),
                    "aspect_band": aspect_band(m, n),
                }
            )
        if launch_fraction is not None:
            signature_map["launch_band"] = ratio_band(launch_fraction)
        if padding_fraction is not None:
            signature_map["padding_band"] = ratio_band(padding_fraction)
        if memory_fraction is not None:
            signature_map["memory_band"] = ratio_band(memory_fraction)
        if reduction_fraction is not None:
            signature_map["reduction_band"] = ratio_band(reduction_fraction)
        if workspace_bytes is not None:
            signature_map["workspace_band"] = size_band(workspace_bytes)
        signature_pairs = _pairs(signature_map)
        tags = tuple(sorted(set(str(tag) for tag in failure_tags)))
        payload = {
            "boundary_id": boundary.boundary_id,
            "boundary_contract_hash": boundary.contract_hash,
            "rule_execution_fingerprint": str(rule_execution_fingerprint),
            "exact": bool(exact),
            "metrics": dict(metric_pairs),
            "signature": dict(signature_pairs),
            "failure_tags": list(tags),
        }
        return cls(
            boundary_id=boundary.boundary_id,
            boundary_contract_hash=boundary.contract_hash,
            rule_execution_fingerprint=str(rule_execution_fingerprint),
            exact=bool(exact),
            metrics=metric_pairs,
            signature=signature_pairs,
            failure_tags=tags,
            evidence_hash=_digest(payload),
        )

    def metric(self, name: str) -> float:
        values = dict(self.metrics)
        if name not in values:
            raise KeyError(f"missing physical metric: {name}")
        return values[name]

    def require_matches(self, boundary: BoundaryContract, rule_execution_fingerprint: str) -> None:
        if self.boundary_id != boundary.boundary_id:
            raise ValueError("physical evidence belongs to a different boundary")
        if self.boundary_contract_hash != boundary.contract_hash:
            raise ValueError("physical evidence belongs to a different frozen contract")
        if self.rule_execution_fingerprint != rule_execution_fingerprint:
            raise ValueError("physical evidence belongs to a different executable rule")

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "boundary_contract_hash": self.boundary_contract_hash,
            "rule_execution_fingerprint": self.rule_execution_fingerprint,
            "exact": self.exact,
            "metrics": dict(self.metrics),
            "signature": dict(self.signature),
            "failure_tags": list(self.failure_tags),
            "evidence_hash": self.evidence_hash,
        }


__all__ = [
    "BoundaryContract",
    "PhysicalEvidence",
    "size_band",
    "ratio_band",
    "aspect_band",
]
