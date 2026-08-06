"""Advanced G4 learning APIs. Ordinary numerical users do not need this layer."""

from .actg import actg_to_bytes, actg_to_object, bytes_to_actg, candidate_genome, object_to_actg
from .capsule import G4Capsule
from .law import JudgeLaw
from .learner import G4Learner, LearningConfig
from .model import Candidate, Experience, Observation, RankedCandidate, ResiduePattern, SearchBoundary
from .supervisor import MidRunSupervisor, RunSummary
from .evidence import BoundaryContract, PhysicalEvidence, aspect_band, ratio_band, size_band
from .residue import (
    ExactResidue,
    ResidueLedger,
    external_baseline_residue,
    extract_residue,
    physical_floor_residue,
)
from .patterns import ExactPattern, describe_signature, discover_exact_patterns
from .mdl import (
    CompleteCost,
    MergeDecision,
    PromotionDecision,
    SplitDecision,
    evaluate_merge,
    evaluate_promotion,
    evaluate_split,
)
from .proofs import ProofObject, ProofStore
from .predictive import PredictiveClass, PredictiveOutcome, discover_predictive_classes
from .mutations import MutationOperator, MutationProposal, MutationRegistry, propose_mutations
from .supersteps import (
    DeterministicEdge,
    Superstep,
    compile_superstep,
    deterministic_edge_is_valid,
)
from .certificates import CertificateRecord, ContinuationState, build_certificate
from .teacher import ResidueDrivenTeacher, TeachingResult

__all__ = [
    "bytes_to_actg",
    "actg_to_bytes",
    "object_to_actg",
    "actg_to_object",
    "candidate_genome",
    "G4Capsule",
    "JudgeLaw",
    "LearningConfig",
    "G4Learner",
    "SearchBoundary",
    "Candidate",
    "Observation",
    "Experience",
    "ResiduePattern",
    "RankedCandidate",
    "MidRunSupervisor",
    "RunSummary",
    "BoundaryContract",
    "PhysicalEvidence",
    "size_band",
    "ratio_band",
    "aspect_band",
    "ExactResidue",
    "ResidueLedger",
    "extract_residue",
    "external_baseline_residue",
    "physical_floor_residue",
    "ExactPattern",
    "discover_exact_patterns",
    "describe_signature",
    "CompleteCost",
    "PromotionDecision",
    "evaluate_promotion",
    "MergeDecision",
    "evaluate_merge",
    "SplitDecision",
    "evaluate_split",
    "ProofObject",
    "ProofStore",
    "PredictiveOutcome",
    "PredictiveClass",
    "discover_predictive_classes",
    "MutationProposal",
    "MutationRegistry",
    "MutationOperator",
    "propose_mutations",
    "DeterministicEdge",
    "Superstep",
    "deterministic_edge_is_valid",
    "compile_superstep",
    "ContinuationState",
    "CertificateRecord",
    "build_certificate",
    "TeachingResult",
    "ResidueDrivenTeacher",
]
