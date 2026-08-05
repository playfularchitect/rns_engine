"""Immutable correctness and promotion law for G4 learning campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True, slots=True)
class JudgeLaw:
    require_compile: bool = True
    require_legal: bool = True
    require_exact: bool = True
    minimum_speedup: float = 1.002
    minimum_win_fraction: float = 20 / 31
    minimum_confidence_lower: float = 1.0
    paired_blocks: int = 31

    @property
    def digest(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def decide(self, observation: "ObservationLike") -> str:
        if self.require_compile and not observation.compile_ok:
            return "COMPILE_FAILURE"
        if self.require_legal and not observation.legal:
            return "ILLEGAL"
        if self.require_exact and not observation.exact:
            return "INEXACT"
        if observation.blocks != self.paired_blocks:
            return "JUDGE_CONTRACT_VIOLATION"
        if observation.wins < 0 or observation.wins > observation.blocks:
            return "JUDGE_CONTRACT_VIOLATION"
        if observation.speedup <= 0 or observation.confidence_lower <= 0:
            return "JUDGE_CONTRACT_VIOLATION"
        if observation.speedup <= self.minimum_speedup:
            return "BELOW_SPEED"
        if observation.wins / observation.blocks < self.minimum_win_fraction:
            return "LOW_WIN_RATE"
        if observation.confidence_lower <= self.minimum_confidence_lower:
            return "LOW_CONFIDENCE"
        return "PROMOTED"


class ObservationLike:
    compile_ok: bool
    legal: bool
    exact: bool
    speedup: float
    confidence_lower: float
    wins: int
    blocks: int


__all__ = ["JudgeLaw"]
