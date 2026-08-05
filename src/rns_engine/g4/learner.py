"""Contextual certified learner updated after every experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .capsule import G4Capsule
from .law import JudgeLaw
from .model import Candidate, Experience, Observation, RankedCandidate, ResiduePattern, SearchBoundary


@dataclass(frozen=True, slots=True)
class LearningConfig:
    learning_rate: float = 0.22
    interaction_learning_rate: float = 0.08
    exploration_strength: float = 0.75
    description_cost_weight: float = 0.02
    work_cost_weight: float = 0.02
    memory_cost_weight: float = 0.02
    promotion_bonus: float = 1.0
    exactness_failure_penalty: float = 4.0
    invalid_penalty: float = 2.0


class G4Learner:
    STATE_VERSION = 1

    def __init__(
        self,
        *,
        judge: JudgeLaw | None = None,
        config: LearningConfig | None = None,
        capsule: G4Capsule | None = None,
    ):
        self.judge = judge or JudgeLaw()
        self.config = config or LearningConfig()
        self.capsule = capsule
        self.weights: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.interaction_weights: dict[str, float] = {}
        self.interaction_counts: dict[str, int] = {}
        self.evaluated: set[str] = set()
        self.proved_impossible: set[str] = set()
        self.residue_counts: dict[str, int] = {}
        self.feature_counts: dict[str, int] = {}
        self.experiences: list[Experience] = []
        self.experience_count = 0
        self.champions: dict[str, dict[str, Any]] = {}
        if capsule is not None:
            capsule.initialize()
            state = capsule.load_state()
            if state is not None:
                self.load_state(state)

    @staticmethod
    def evaluation_key(boundary: SearchBoundary, candidate: Candidate) -> str:
        return f"{boundary.fingerprint}:{candidate.fingerprint}"

    @staticmethod
    def _weight_key(context: str, feature: str) -> str:
        return f"{context}\u241f{feature}"

    @staticmethod
    def _interaction_key(context: str, left: str, right: str) -> str:
        a, b = sorted((left, right))
        return f"{context}\u241f{a}\u241f{b}"

    @staticmethod
    def _residue_key(boundary: SearchBoundary, context: str, feature: str, residue: str) -> str:
        return f"{boundary.fingerprint}\u241f{context}\u241f{feature}\u241f{residue}"

    @staticmethod
    def _feature_count_key(boundary: SearchBoundary, context: str, feature: str) -> str:
        return f"{boundary.fingerprint}\u241f{context}\u241f{feature}"

    def _reward(self, candidate: Candidate, observation: Observation, decision: str) -> float:
        if decision in {"COMPILE_FAILURE", "ILLEGAL"}:
            base = -self.config.invalid_penalty
        elif decision == "INEXACT":
            base = -self.config.exactness_failure_penalty
        else:
            speed = max(observation.speedup, 1e-12)
            base = 4.0 * math.log2(speed)
            if decision == "PROMOTED":
                base += self.config.promotion_bonus
            elif decision in {"LOW_WIN_RATE", "LOW_CONFIDENCE"}:
                base *= 0.5
        cost = (
            self.config.description_cost_weight * candidate.description_cost
            + self.config.work_cost_weight * (candidate.expected_work_cost + observation.actual_work_cost)
            + self.config.memory_cost_weight * (candidate.expected_memory_cost + observation.actual_memory_cost)
        )
        return base - cost

    def _credited_features(self, candidate: Candidate, parent: Candidate | None) -> tuple[str, ...]:
        if parent is None:
            return candidate.features
        changed = sorted(set(candidate.features).symmetric_difference(parent.features))
        return tuple(changed or candidate.features)

    def observe(
        self,
        boundary: SearchBoundary,
        candidate: Candidate,
        observation: Observation,
        *,
        parent: Candidate | None = None,
    ) -> Experience:
        key = self.evaluation_key(boundary, candidate)
        if key in self.evaluated:
            raise ValueError("candidate already evaluated inside this boundary")
        decision = self.judge.decide(observation)
        reward = self._reward(candidate, observation, decision)
        credited = self._credited_features(candidate, parent)
        contexts = boundary.context_tokens()

        for context in contexts:
            for feature in credited:
                weight_key = self._weight_key(context, feature)
                old = self.weights.get(weight_key, 0.0)
                self.weights[weight_key] = old + self.config.learning_rate * (reward - old)
                self.counts[weight_key] = self.counts.get(weight_key, 0) + 1
                feature_key = self._feature_count_key(boundary, context, feature)
                self.feature_counts[feature_key] = self.feature_counts.get(feature_key, 0) + 1
                residue_key = self._residue_key(boundary, context, feature, decision)
                self.residue_counts[residue_key] = self.residue_counts.get(residue_key, 0) + 1
            if len(credited) >= 2:
                for index, left in enumerate(credited):
                    for right in credited[index + 1 :]:
                        interaction_key = self._interaction_key(context, left, right)
                        old = self.interaction_weights.get(interaction_key, 0.0)
                        rate = self.config.interaction_learning_rate
                        self.interaction_weights[interaction_key] = old + rate * (reward - old)
                        self.interaction_counts[interaction_key] = self.interaction_counts.get(interaction_key, 0) + 1

        self.evaluated.add(key)
        experience = Experience(
            boundary=boundary,
            candidate=candidate,
            observation=observation,
            decision=decision,
            reward=reward,
            sequence=self.experience_count + 1,
        )
        self.experiences.append(experience)
        self.experience_count += 1

        if decision == "PROMOTED":
            current = self.champions.get(boundary.fingerprint)
            if current is None or observation.speedup > float(current["speedup"]):
                self.champions[boundary.fingerprint] = {
                    "candidate_id": candidate.candidate_id,
                    "candidate_fingerprint": candidate.fingerprint,
                    "speedup": observation.speedup,
                    "experience": experience.fingerprint,
                }

        if self.capsule is not None:
            self.capsule.store_experience(experience.to_dict())
            self.capsule.save_state(self.to_state())
        return experience

    def prove_impossible(self, boundary: SearchBoundary, candidate: Candidate, *, reason: str) -> None:
        key = self.evaluation_key(boundary, candidate)
        self.proved_impossible.add(key)
        if self.capsule is not None:
            self.capsule.store_experience(
                {
                    "kind": "PROVED_IMPOSSIBLE",
                    "boundary": boundary.to_dict(),
                    "candidate": candidate.to_dict(),
                    "reason": reason,
                }
            )
            self.capsule.save_state(self.to_state())

    def score(self, boundary: SearchBoundary, candidate: Candidate) -> RankedCandidate:
        contexts = boundary.context_tokens()
        evidence = 0.0
        evidence_count = 0
        for context in contexts:
            for feature in candidate.features:
                key = self._weight_key(context, feature)
                evidence += self.weights.get(key, 0.0)
                evidence_count += self.counts.get(key, 0)
            for index, left in enumerate(candidate.features):
                for right in candidate.features[index + 1 :]:
                    key = self._interaction_key(context, left, right)
                    evidence += self.interaction_weights.get(key, 0.0)
                    evidence_count += self.interaction_counts.get(key, 0)
        total = max(1, self.experience_count)
        exploration = self.config.exploration_strength * math.sqrt(math.log(total + 1) / (evidence_count + 1))
        cost = (
            self.config.description_cost_weight * candidate.description_cost
            + self.config.work_cost_weight * candidate.expected_work_cost
            + self.config.memory_cost_weight * candidate.expected_memory_cost
        )
        return RankedCandidate(candidate, evidence + exploration - cost, evidence, exploration, cost)

    def rank(self, boundary: SearchBoundary, candidates: Iterable[Candidate]) -> list[RankedCandidate]:
        ranked: list[RankedCandidate] = []
        for candidate in candidates:
            key = self.evaluation_key(boundary, candidate)
            if key in self.evaluated or key in self.proved_impossible:
                continue
            ranked.append(self.score(boundary, candidate))
        ranked.sort(key=lambda item: (-item.score, item.candidate.candidate_id))
        return ranked

    def patterns(
        self,
        boundary: SearchBoundary,
        *,
        minimum_support: int = 3,
        minimum_rate: float = 0.75,
    ) -> list[ResiduePattern]:
        output: list[ResiduePattern] = []
        prefix = boundary.fingerprint + "\u241f"
        for key, support in self.residue_counts.items():
            if not key.startswith(prefix) or support < minimum_support:
                continue
            _, context, feature, residue = key.split("\u241f", 3)
            total = self.feature_counts.get(self._feature_count_key(boundary, context, feature), 0)
            if not total:
                continue
            rate = support / total
            if rate < minimum_rate:
                continue
            if residue == "PROMOTED":
                action = "increase_sampling"
            elif residue in {"INEXACT", "ILLEGAL", "COMPILE_FAILURE"}:
                action = "suppress_or_prove"
            else:
                action = "deprioritize_and_mutate"
            output.append(
                ResiduePattern(
                    boundary_fingerprint=boundary.fingerprint,
                    context_token=context,
                    feature=feature,
                    residue=residue,
                    support=support,
                    rate=rate,
                    suggested_action=action,
                )
            )
        output.sort(key=lambda pattern: (-pattern.rate, -pattern.support, pattern.feature, pattern.residue))
        return output

    def to_state(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "judge_digest": self.judge.digest,
            "weights": self.weights,
            "counts": self.counts,
            "interaction_weights": self.interaction_weights,
            "interaction_counts": self.interaction_counts,
            "evaluated": sorted(self.evaluated),
            "proved_impossible": sorted(self.proved_impossible),
            "residue_counts": self.residue_counts,
            "feature_counts": self.feature_counts,
            "champions": self.champions,
            "experience_count": self.experience_count,
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        if int(state.get("state_version", -1)) != self.STATE_VERSION:
            raise ValueError("unsupported G4 learner state version")
        if state.get("judge_digest") != self.judge.digest:
            raise ValueError("capsule judge law does not match the active immutable judge")
        self.weights = {str(k): float(v) for k, v in state.get("weights", {}).items()}
        self.counts = {str(k): int(v) for k, v in state.get("counts", {}).items()}
        self.interaction_weights = {str(k): float(v) for k, v in state.get("interaction_weights", {}).items()}
        self.interaction_counts = {str(k): int(v) for k, v in state.get("interaction_counts", {}).items()}
        self.evaluated = set(state.get("evaluated", ()))
        self.proved_impossible = set(state.get("proved_impossible", ()))
        self.residue_counts = {str(k): int(v) for k, v in state.get("residue_counts", {}).items()}
        self.feature_counts = {str(k): int(v) for k, v in state.get("feature_counts", {}).items()}
        self.champions = {str(k): dict(v) for k, v in state.get("champions", {}).items()}
        self.experience_count = int(state.get("experience_count", 0))


__all__ = ["LearningConfig", "G4Learner"]
