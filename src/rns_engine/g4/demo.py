"""Deterministic demonstration of mid-run G4 learning and transfer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .capsule import G4Capsule
from .learner import G4Learner
from .model import Candidate, Observation, SearchBoundary
from .supervisor import MidRunSupervisor


def demo_candidates() -> list[Candidate]:
    output: list[Candidate] = []
    index = 0
    for layout in ("row_major", "transpose_trick"):
        for staging in ("direct", "shared"):
            for reuse in ("none", "reuse_a", "reuse_b", "reuse_ab"):
                for split in ("split1", "split2", "split4"):
                    features = (layout, staging, reuse, split)
                    output.append(
                        Candidate.create(
                            f"C{index:03d}",
                            features=features,
                            parameters={"layout": layout, "staging": staging, "reuse": reuse, "split": split},
                            mutation_ops=("seed",),
                            description_cost=0.1 * len(features),
                            expected_work_cost=0.2 if staging == "shared" else 0.05,
                            expected_memory_cost=0.2 if split != "split1" else 0.0,
                        )
                    )
                    index += 1
    return output


def _noise(boundary: SearchBoundary, candidate: Candidate) -> float:
    payload = f"{boundary.fingerprint}:{candidate.fingerprint}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") / 2**32
    return (value - 0.5) * 0.012


def evaluator(boundary: SearchBoundary, candidate: Candidate) -> Observation:
    features = set(candidate.features)
    speedup = 0.93
    speedup += 0.065 if "row_major" in features else 0.0
    speedup += 0.045 if "direct" in features else -0.09
    speedup += 0.0 if "none" in features else 0.012
    if "reuse_ab" in features:
        speedup += 0.11
    if "reuse_a" in features or "reuse_b" in features:
        speedup -= 0.02
    deep_k = dict(boundary.constraints).get("k_band") == "deep"
    if "split2" in features:
        speedup += 0.075 if deep_k else -0.035
    if "split4" in features:
        speedup += 0.10 if deep_k else -0.07
    speedup += _noise(boundary, candidate)
    exact = not ("shared" in features and "split4" in features)
    wins = 27 if speedup > 1.01 else (21 if speedup > 1.002 else 14)
    confidence = speedup - 0.004
    return Observation(
        compile_ok=True,
        legal=True,
        exact=exact,
        speedup=speedup,
        confidence_lower=confidence,
        wins=wins,
        blocks=31,
        elapsed_seconds=0.01,
        actual_memory_cost=0.1 if "shared" in features else 0.0,
        actual_work_cost=0.1 if "shared" in features else 0.0,
    )


def run_demo(capsule_path: Path, *, budget: int = 20) -> dict[str, object]:
    capsule = G4Capsule(capsule_path)
    learner = G4Learner(capsule=capsule)
    supervisor = MidRunSupervisor(learner)
    candidates = demo_candidates()
    first = SearchBoundary.create(
        environment="synthetic_t4",
        domain="exact_gemm",
        operation="matmul",
        constraints={"k_band": "deep", "output": "row_major"},
    )
    first_summary = supervisor.run(first, candidates, evaluator, budget=budget)

    second = SearchBoundary.create(
        environment="synthetic_t4",
        domain="exact_gemm",
        operation="batched_matmul",
        constraints={"k_band": "deep", "output": "row_major"},
    )
    before = [item.candidate.candidate_id for item in learner.rank(second, candidates)[:5]]
    second_summary = supervisor.run(second, candidates, evaluator, budget=min(8, budget))

    return {
        "schema": "rns_engine.g4_midrun_demo.v1",
        "judge_digest": learner.judge.digest,
        "first_boundary": asdict(first_summary),
        "transfer_top5_before_second_run": before,
        "second_boundary": asdict(second_summary),
        "capsule": capsule.verify(),
        "experience_count": learner.experience_count,
        "note": "Synthetic deterministic proof of the learning loop; not a performance claim.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic G4 mid-run learning demonstration.")
    parser.add_argument("--capsule", type=Path, default=Path("G4_Capsule"))
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("g4_midrun_demo.json"))
    args = parser.parse_args(argv)
    report = run_demo(args.capsule, budget=args.budget)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
