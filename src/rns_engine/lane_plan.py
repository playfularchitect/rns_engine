from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .capacity import SignedCapacityPlan
from .rail_planning import MersenneRailSetPlan, search_mersenne_rails_for_capacity


@dataclass(frozen=True, slots=True)
class RailLearningLane:
    """One lawful candidate lane sharing the same exact capacity prior."""

    lane_id: str
    objective: Literal["smallest_product", "balanced"]
    plan: MersenneRailSetPlan

    @property
    def structural_score(self) -> tuple[int, int, tuple[int, ...], str]:
        if self.objective == "smallest_product":
            return (
                self.plan.additional_product,
                self.plan.exponent_span,
                self.plan.exponents,
                self.lane_id,
            )
        return (
            self.plan.exponent_span,
            self.plan.additional_product,
            self.plan.exponents,
            self.lane_id,
        )


@dataclass(frozen=True, slots=True)
class ParallelRailPrior:
    required_additional_product: int
    lanes: tuple[RailLearningLane, ...]

    def require_lawful(self) -> ParallelRailPrior:
        if not self.lanes:
            raise ValueError("at least one lane is required")
        for lane in self.lanes:
            lane.plan.require_sufficient()
            if lane.plan.required_additional_product != self.required_additional_product:
                raise ValueError("all lanes must share the same capacity prior")
        return self


def build_parallel_rail_prior(capacity: SignedCapacityPlan) -> ParallelRailPrior:
    search = search_mersenne_rails_for_capacity(
        capacity,
        max_exponent=31,
        max_rails=4,
    )
    smallest = search.smallest_product_plan
    balanced = search.most_balanced_plan
    if smallest is None or balanced is None:
        raise OverflowError("no lawful rail lanes close the requested capacity")
    return ParallelRailPrior(
        required_additional_product=capacity.minimum_additional_product_factor,
        lanes=(
            RailLearningLane("lane-smallest-product", "smallest_product", smallest),
            RailLearningLane("lane-balanced", "balanced", balanced),
        ),
    ).require_lawful()
