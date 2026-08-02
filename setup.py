import os
import platform

import numpy as np
import pybind11
from setuptools import Extension, setup


def get_compile_args():
    system = platform.system()
    archflags = os.environ.get("ARCHFLAGS", "")
    if "arm64" in archflags:
        machine = "arm64"
    else:
        machine = platform.machine().lower()

    if system == "Windows":
        args = ["/std:c++17", "/O2", "/DNDEBUG"]
        if machine in ("x86_64", "amd64"):
            args += ["/arch:AVX2", "/D_FORCE_AVX2"]
        return args

    args = ["-std=c++17", "-O3", "-DNDEBUG"]

    if system == "Linux":
        args += ["-fopenmp"]

    # macOS intentionally uses the scalar fallback instead of OpenMP so the
    # arm64 and x86_64 wheels do not gain an external libomp dependency.
    if machine in ("x86_64", "amd64"):
        args += ["-mavx2", "-funroll-loops", "-DFORCE_AVX2"]

    return args


def get_link_args():
    if platform.system() == "Linux":
        return ["-fopenmp"]
    return []


def native_extension(name, source):
    return Extension(
        name,
        sources=[source],
        include_dirs=[pybind11.get_include(), np.get_include()],
        extra_compile_args=get_compile_args(),
        extra_link_args=get_link_args(),
        language="c++",
    )


ext = native_extension("rns_engine._core", "src/rns_engine/_core.cpp")
weighted_ext = native_extension(
    "rns_engine._weighted",
    "src/rns_engine/_weighted.cpp",
)

# Project metadata and package discovery live in pyproject.toml. Keeping them
# out of setup.py prevents the two build entry points from drifting apart.
setup(
    ext_modules=[ext, weighted_ext],
    include_package_data=True,
    zip_safe=False,
)
