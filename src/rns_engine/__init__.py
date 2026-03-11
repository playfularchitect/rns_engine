from ._core import HAS_AVX2
from ._core import M
from ._core import M0
from ._core import M1
from ._core import M2
from ._core import M3
from ._core import add
from ._core import add_u64_auto
from ._core import add_u64_io
from ._core import add_u64_io_omp
from ._core import affine_repeat
from ._core import affine_repeat_u64
from ._core import affine_repeat_u64_auto
from ._core import affine_repeat_u64_io
from ._core import affine_repeat_u64_io_omp
from ._core import decode
from ._core import div_ as div
from ._core import encode
from ._core import fma
from ._core import fma_u64
from ._core import fma_u64_auto
from ._core import fma_u64_io
from ._core import fma_u64_io_omp
from ._core import mul
from ._core import mul_u64
from ._core import mul_u64_auto
from ._core import mul_u64_io
from ._core import mul_u64_io_omp
from ._core import omp_max_threads
from ._core import omp_num_procs
from ._core import omp_set_num_threads
from ._core import sub
from ._core import sub_u64_auto
from ._core import sub_u64_io
from ._core import sub_u64_io_omp
from .engine import EncodedArray
from .engine import Session
from .engine import SessionCache

__version__ = "0.4.0rc1"

__all__ = [
    "__version__",
    "M",
    "M0",
    "M1",
    "M2",
    "M3",
    "HAS_AVX2",
    "encode",
    "decode",
    "add",
    "sub",
    "mul",
    "div",
    "fma",
    "mul_u64",
    "fma_u64",
    "affine_repeat",
    "affine_repeat_u64",
    "add_u64_io",
    "sub_u64_io",
    "mul_u64_io",
    "fma_u64_io",
    "affine_repeat_u64_io",
    "add_u64_io_omp",
    "sub_u64_io_omp",
    "mul_u64_io_omp",
    "fma_u64_io_omp",
    "affine_repeat_u64_io_omp",
    "add_u64_auto",
    "sub_u64_auto",
    "mul_u64_auto",
    "fma_u64_auto",
    "affine_repeat_u64_auto",
    "omp_max_threads",
    "omp_set_num_threads",
    "omp_num_procs",
    "EncodedArray",
    "SessionCache",
    "Session",
]
