from importlib.metadata import PackageNotFoundError, version as _distribution_version

from ._core import (
    HAS_AVX2,
    M,
    M0,
    M1,
    M2,
    M3,
    encode,
    decode,
    op,
    add,
    sub,
    mul,
    div_,
    fma,
    affine_repeat,
    mul_u64,
    fma_u64,
    affine_repeat_u64,
    add_u64_io,
    add_u64_io_omp,
    add_u64_auto,
    sub_u64_io,
    sub_u64_io_omp,
    sub_u64_auto,
    mul_u64_io,
    mul_u64_io_omp,
    mul_u64_auto,
    fma_u64_io,
    fma_u64_io_omp,
    fma_u64_auto,
    affine_repeat_u64_io,
    affine_repeat_u64_io_omp,
    affine_repeat_u64_auto,
    omp_max_threads,
    omp_set_num_threads,
    omp_num_procs,
)
from .engine import EncodedArray, Session, SessionCache
from .signed import (
    HALF_M,
    SIGNED_MAX,
    SIGNED_MIN,
    UNIQUE_SIGNED_MAX,
    UNIQUE_SIGNED_MIN,
    SignedRangeCertificate,
    SignedSession,
    certify_signed_bound,
    certify_signed_dot_bound,
    decode_signed,
    encode_signed,
)
from .weighted import (
    WeightedInt32Result,
    accumulate_weighted_int32,
    certify_weighted_sum_bound,
)
from .capacity import (
    DigitPlaneGemmCapacityPlan,
    SignedCapacityPlan,
    plan_digit_plane_gemm_capacity,
    plan_signed_capacity,
    plan_weighted_sum_capacity,
)
from .coefficients import (
    GroupedCoefficientCapacityPlan,
    plan_grouped_coefficient_capacity,
)
from .rail_planning import (
    MersenneRailCandidate,
    MersenneRailSearchResult,
    MersenneRailSetPlan,
    search_mersenne_rail_sets,
    search_mersenne_rails_for_capacity,
)
from .wide import (
    BALANCED_EXTRA_EXPONENTS,
    BASE_MODULI,
    SMALLEST_PRODUCT_EXTRA_EXPONENTS,
    WideRNSConfig,
    WideWeightedResult,
    accumulate_weighted_int32_wide,
    mersenne_modulus,
    moduli_from_mersenne_exponents,
)
from .exact_gemm import (
    ExactSharedScaleGemmReceipt,
    SharedScaleMatrix,
    decompose_signed_radix,
    exact_integer_matmul,
    exact_shared_scale_gemm,
    grouped_plane_gemm,
    reconstruct_grouped_partials,
    reconstruct_signed_radix,
)
from .cuda_contract import (
    BackendVerification,
    CpuExactPipelineBackend,
    CpuGroupedPartialsBackend,
    CudaGemmFixture,
    ExactPipelineBackend,
    build_cuda_gemm_fixture,
    verify_backend,
)
from .lane_plan import (
    ParallelRailPrior,
    RailLearningLane,
    build_parallel_rail_prior,
)
from .pre_cuda import (
    PreCudaReadinessReport,
    build_default_pre_cuda_fixture,
    default_pre_cuda_matrices,
    run_pre_cuda_readiness,
)
from .g4_results import g4_results
from .g4_benchmark import g4_benchmark


try:
    __version__ = _distribution_version("rns_engine")
except PackageNotFoundError:
    # This only occurs when importing directly from an unpacked source tree
    # rather than from an installed wheel/editable build.
    __version__ = "0+unknown"

__all__ = [
    "HAS_AVX2",
    "M", "M0", "M1", "M2", "M3",
    "HALF_M", "SIGNED_MIN", "SIGNED_MAX",
    "UNIQUE_SIGNED_MIN", "UNIQUE_SIGNED_MAX",
    "encode", "decode", "encode_signed", "decode_signed", "op",
    "add", "sub", "mul", "div_", "fma",
    "affine_repeat",
    "mul_u64", "fma_u64", "affine_repeat_u64",
    "add_u64_io", "add_u64_io_omp", "add_u64_auto",
    "sub_u64_io", "sub_u64_io_omp", "sub_u64_auto",
    "mul_u64_io", "mul_u64_io_omp", "mul_u64_auto",
    "fma_u64_io", "fma_u64_io_omp", "fma_u64_auto",
    "affine_repeat_u64_io", "affine_repeat_u64_io_omp", "affine_repeat_u64_auto",
    "omp_max_threads", "omp_set_num_threads", "omp_num_procs",
    "EncodedArray", "SessionCache", "Session", "SignedSession",
    "SignedRangeCertificate", "certify_signed_bound", "certify_signed_dot_bound",
    "WeightedInt32Result", "accumulate_weighted_int32", "certify_weighted_sum_bound",
    "SignedCapacityPlan", "DigitPlaneGemmCapacityPlan",
    "GroupedCoefficientCapacityPlan",
    "plan_signed_capacity", "plan_weighted_sum_capacity",
    "plan_digit_plane_gemm_capacity", "plan_grouped_coefficient_capacity",
    "MersenneRailCandidate", "MersenneRailSetPlan", "MersenneRailSearchResult",
    "search_mersenne_rail_sets", "search_mersenne_rails_for_capacity",
    "BASE_MODULI", "SMALLEST_PRODUCT_EXTRA_EXPONENTS",
    "BALANCED_EXTRA_EXPONENTS", "WideRNSConfig", "WideWeightedResult",
    "mersenne_modulus", "moduli_from_mersenne_exponents",
    "accumulate_weighted_int32_wide",
    "SharedScaleMatrix", "ExactSharedScaleGemmReceipt",
    "exact_integer_matmul", "decompose_signed_radix",
    "reconstruct_signed_radix", "grouped_plane_gemm",
    "reconstruct_grouped_partials", "exact_shared_scale_gemm",
    "CudaGemmFixture", "BackendVerification", "ExactPipelineBackend",
    "CpuExactPipelineBackend", "CpuGroupedPartialsBackend",
    "build_cuda_gemm_fixture", "verify_backend",
    "RailLearningLane", "ParallelRailPrior", "build_parallel_rail_prior",
    "PreCudaReadinessReport", "default_pre_cuda_matrices",
    "build_default_pre_cuda_fixture", "run_pre_cuda_readiness",
    "g4_results", "g4_benchmark",
    "info",
]


def info():
    print(f"rns_engine v{__version__}")
    print(f"  Dynamic range : [0, {M:,})")
    print(f"  Signed view   : [{SIGNED_MIN:,}, {SIGNED_MAX:,}]")
    print(f"  Unique bound  : |x| < {HALF_M:,}")
    print(f"  Moduli        : {M0} x {M1} x {M2} x {M3}")
    print(f"  AVX2          : {'yes' if HAS_AVX2 else 'no'}")
    print("  Core APIs     : add/sub/mul/fma + raw/omp/auto scalar-broadcast family")
    print("  Bridge APIs   : weighted signed INT32 accumulation with range receipts")
    print("  Planning APIs : local/global capacity + exact Mersenne rail-set search")
    print("  Pre-CUDA APIs : seven-rail oracle + exact shared-scale GEMM fixtures")
    print("  G4 APIs       : frozen Series 1 evidence + Tesla T4 reproduction benchmark")
