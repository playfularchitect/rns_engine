"""
setup.py — builds the rns_engine C++ extension.

Detects AVX2 support and enables it when available.
Falls back to scalar on non-x86 or older hardware.
"""

import sys
import platform
from setuptools import setup, Extension
import pybind11


def get_compile_args():
    """Return compiler flags appropriate for the current platform."""
    system = platform.system()
    machine = platform.machine()

    common = ["-std=c++17", "-O3", "-DNDEBUG", "-Wno-unused-function"]

    if system in ("Linux", "Darwin") and machine in ("x86_64", "AMD64"):
        # Enable AVX2 on x86-64 Linux/Mac
        return common + ["-mavx2", "-march=native", "-funroll-loops"]

    elif system == "Windows":
        # MSVC flags
        return ["/std:c++17", "/O2", "/DNDEBUG", "/arch:AVX2"]

    else:
        # ARM, RISC-V, etc. — scalar fallback, still fast
        return common

    return common


def get_link_args():
    if platform.system() == "Windows":
        return []
    return []


ext = Extension(
    "rns_engine._core",
    sources=["src/rns_engine/_core.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=get_compile_args(),
    extra_link_args=get_link_args(),
    language="c++",
)

setup(ext_modules=[ext])
