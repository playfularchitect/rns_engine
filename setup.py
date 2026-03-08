"""
setup.py — build configuration for the rns_engine C++ extension.

Notes:
- The default build is portable and does not assume AVX2.
- AVX2 can be enabled explicitly on supported x86_64 machines by setting:
    RNS_ENGINE_ENABLE_AVX2=1
- NumPy headers are included explicitly because the extension uses NumPy arrays.
"""

import os
import platform
from setuptools import setup, Extension
import pybind11
import numpy as np


def env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_compile_args():
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Windows":
        args = ["/std:c++17", "/O2", "/DNDEBUG"]
        if machine in ("amd64", "x86_64") and env_flag("RNS_ENGINE_ENABLE_AVX2"):
            args.append("/arch:AVX2")
        return args

    args = ["-std=c++17", "-O3", "-DNDEBUG", "-Wno-unused-function"]

    # Keep published builds portable by default.
    # Opt into AVX2 explicitly for local/source builds on supported CPUs.
    if machine in ("x86_64", "amd64") and env_flag("RNS_ENGINE_ENABLE_AVX2"):
        args += ["-mavx2", "-funroll-loops"]

    return args


def get_link_args():
    return []


ext = Extension(
    "rns_engine._core",
    sources=["src/rns_engine/_core.cpp"],
    include_dirs=[pybind11.get_include(), np.get_include()],
    extra_compile_args=get_compile_args(),
    extra_link_args=get_link_args(),
    language="c++",
)

setup(ext_modules=[ext])
