"""Complete MDL and physical cost accounting for G4 rule survival."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CompleteCost:
    """Keep descriptive and physical costs explicit instead of hiding either."""

    exact: bool
    live_structure_bits: int = 0
    input_residue_bits: int = 0
    correction_residue_bits: int = 0
    ambiguity_bits: int = 0
    mutation_bits: int = 0
    nondeterministic_work_bits: int = 0
    direct_latency_ms: float = 0.0
    graph_latency_ms: float | None = None
    workspace_bytes: int = 0
    energy_joules: float | None = None

    def __post_init__(self) -> None:
        integer_fields = (
            self.live_structure_bits,
            self.input_residue_bits,
            self.correction_residue_bits,
            self.ambiguity_bits,
            self.mutation_bits,
            self.nondeterministic_work_bits,
            self.workspace_bytes,
        )
        if any(value < 0 for value in integer_fields):
            raise ValueError("complete-cost fields must be nonnegative")
        if self.direct_latency_ms < 0:
            raise ValueError("direct latency must be nonnegative")
        if self.graph_latency_ms is not None and self.graph_latency_ms < 0:
            raise ValueError("graph latency must be nonnegative")
        if self.energy_joules is not None and self.energy_joules < 0:
            raise ValueError("energy must be nonnegative")

    @property
    def description_bits(self) -> int:
        return (
            self.live_structure_bits
            + self.input_residue_bits
            + self.correction_residue_bits
            + self.ambiguity_bits
            + self.mutation_bits
            + self.nondeterministic_work_bits
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact": self.exact,
            "live_structure_bits": self.live_structure_bits,
            "input_residue_bits": self.input_residue_bits,
            "correction_residue_bits": self.correction_residue_bits,
            "ambiguity_bits": self.ambiguity_bits,
            "mutation_bits": self.mutation_bits,
            "nondeterministic_work_bits": self.nondeterministic_work_bits,
            "description_bits": self.description_bits,
            "direct_latency_ms": self.direct_latency_ms,
            "graph_latency_ms": self.graph_latency_ms,
            "workspace_bytes": self.workspace_bytes,
            "energy_joules": self.energy_joules,
        }


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    accepted: bool
    reason: str
    incumbent: CompleteCost
    challenger: CompleteCost
    latency_improvement_ms: float
    description_savings_bits: int


def evaluate_promotion(
    *,
    incumbent: CompleteCost,
    challenger: CompleteCost,
    confidence_passed: bool,
    workspace_ceiling_bytes: int | None = None,
    require_description_nonincrease: bool = False,
) -> PromotionDecision:
    if not challenger.exact:
        reason = "INEXACT"
    elif not confidence_passed:
        reason = "CONFIDENCE_FAILED"
    elif challenger.direct_latency_ms >= incumbent.direct_latency_ms:
        reason = "PRIMARY_OBJECTIVE_NOT_IMPROVED"
    elif workspace_ceiling_bytes is not None and challenger.workspace_bytes > workspace_ceiling_bytes:
        reason = "WORKSPACE_CONTRACT_VIOLATION"
    elif require_description_nonincrease and challenger.description_bits > incumbent.description_bits:
        reason = "DESCRIPTION_COST_INCREASED"
    else:
        reason = "PROMOTED"
    return PromotionDecision(
        accepted=reason == "PROMOTED",
        reason=reason,
        incumbent=incumbent,
        challenger=challenger,
        latency_improvement_ms=incumbent.direct_latency_ms - challenger.direct_latency_ms,
        description_savings_bits=incumbent.description_bits - challenger.description_bits,
    )


@dataclass(frozen=True, slots=True)
class MergeDecision:
    accepted: bool
    separate_cost_bits: int
    merged_cost_bits: int
    restoring_residue_bits: int
    savings_bits: int


def evaluate_merge(
    *,
    separate_structure_bits: int,
    merged_structure_bits: int,
    restoring_residue_bits: int,
) -> MergeDecision:
    if min(separate_structure_bits, merged_structure_bits, restoring_residue_bits) < 0:
        raise ValueError("merge costs must be nonnegative")
    separate = separate_structure_bits
    merged = merged_structure_bits + restoring_residue_bits
    return MergeDecision(
        accepted=merged < separate,
        separate_cost_bits=separate,
        merged_cost_bits=merged,
        restoring_residue_bits=restoring_residue_bits,
        savings_bits=separate - merged,
    )


@dataclass(frozen=True, slots=True)
class SplitDecision:
    accepted: bool
    shared_cost_bits: int
    split_cost_bits: int
    correction_bits_removed: int
    savings_bits: int


def evaluate_split(
    *,
    shared_structure_bits: int,
    current_correction_bits: int,
    split_structure_bits: int,
    remaining_correction_bits: int,
) -> SplitDecision:
    if min(
        shared_structure_bits,
        current_correction_bits,
        split_structure_bits,
        remaining_correction_bits,
    ) < 0:
        raise ValueError("split costs must be nonnegative")
    before = shared_structure_bits + current_correction_bits
    after = split_structure_bits + remaining_correction_bits
    return SplitDecision(
        accepted=after < before,
        shared_cost_bits=before,
        split_cost_bits=after,
        correction_bits_removed=current_correction_bits - remaining_correction_bits,
        savings_bits=before - after,
    )


__all__ = [
    "CompleteCost",
    "PromotionDecision",
    "evaluate_promotion",
    "MergeDecision",
    "evaluate_merge",
    "SplitDecision",
    "evaluate_split",
]
