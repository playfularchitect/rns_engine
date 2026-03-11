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

__version__ = "0.4.0rc1"

__all__ = [
    "HAS_AVX2",
    "M", "M0", "M1", "M2", "M3",
    "encode", "decode", "op",
    "add", "sub", "mul", "div_", "fma",
    "affine_repeat",
    "mul_u64", "fma_u64", "affine_repeat_u64",
    "add_u64_io", "add_u64_io_omp", "add_u64_auto",
    "sub_u64_io", "sub_u64_io_omp", "sub_u64_auto",
    "mul_u64_io", "mul_u64_io_omp", "mul_u64_auto",
    "fma_u64_io", "fma_u64_io_omp", "fma_u64_auto",
    "affine_repeat_u64_io", "affine_repeat_u64_io_omp", "affine_repeat_u64_auto",
    "omp_max_threads", "omp_set_num_threads", "omp_num_procs",
    "EncodedArray", "SessionCache", "Session",
    "info",
]

def info():
    print(f"rns_engine v{__version__}")
    print(f"  Dynamic range : [0, {M:,})")
    print(f"  Moduli        : {M0} x {M1} x {M2} x {M3}")
    print(f"  AVX2          : {'yes' if HAS_AVX2 else 'no'}")
    print("  Core APIs     : add/sub/mul/fma + raw/omp/auto scalar-broadcast family")
