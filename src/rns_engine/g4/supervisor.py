"""Synchronous mid-run adaptive supervisor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .learner import G4Learner
from .model import Candidate, Experience, Observation, ResiduePattern, SearchBoundary


Evaluator = Callable[[SearchBoundary, Candidate], Observation]
Generator = Callable[[SearchBoundary, G4Learner, Sequence[Candidate]], Iterable[Candidate]]


@dataclass(frozen=True, slots=True)
class RunSummary:
    boundary_fingerprint: str
    budget: int
    evaluated: int
    selection_order: tuple[str, ...]
    decisions: tuple[str, ...]
    champion: dict[str, object] | None
    patterns: tuple[ResiduePattern, ...]
    exhausted: bool


class MidRunSupervisor:
    def __init__(self, learner: G4Learner):
        self.learner = learner

    def run(
        self,
        boundary: SearchBoundary,
        candidates: Iterable[Candidate],
        evaluator: Evaluator,
        *,
        budget: int,
        generator: Generator | None = None,
        generation_interval: int = 5,
    ) -> RunSummary:
        if budget <= 0:
            raise ValueError("budget must be positive")
        pool: dict[str, Candidate] = {candidate.fingerprint: candidate for candidate in candidates}
        order: list[str] = []
        decisions: list[str] = []
        evaluated = 0

        while evaluated < budget:
            ranked = self.learner.rank(boundary, pool.values())
            if not ranked:
                break
            candidate = ranked[0].candidate
            parent = None
            if candidate.parent_id is not None:
                parent = next((item for item in pool.values() if item.candidate_id == candidate.parent_id), None)
            observation = evaluator(boundary, candidate)
            experience: Experience = self.learner.observe(
                boundary,
                candidate,
                observation,
                parent=parent,
            )
            order.append(candidate.candidate_id)
            decisions.append(experience.decision)
            evaluated += 1

            if generator is not None and evaluated % generation_interval == 0:
                generated = generator(boundary, self.learner, tuple(pool.values()))
                for new_candidate in generated:
                    pool.setdefault(new_candidate.fingerprint, new_candidate)

        remaining = self.learner.rank(boundary, pool.values())
        patterns = tuple(self.learner.patterns(boundary))
        champion = self.learner.champions.get(boundary.fingerprint)
        return RunSummary(
            boundary_fingerprint=boundary.fingerprint,
            budget=budget,
            evaluated=evaluated,
            selection_order=tuple(order),
            decisions=tuple(decisions),
            champion=None if champion is None else dict(champion),
            patterns=patterns,
            exhausted=not remaining,
        )


__all__ = ["Evaluator", "Generator", "RunSummary", "MidRunSupervisor"]
