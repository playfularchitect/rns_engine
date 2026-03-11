import os
import platform
from setuptools import setup, Extension, find_packages
import pybind11
import numpy as np


def get_macos_libomp():
    for prefix in ["/opt/homebrew", "/usr/local"]:
        inc = f"{prefix}/opt/libomp/include"
        lib = f"{prefix}/opt/libomp/lib"
        if os.path.isdir(inc) and os.path.isdir(lib):
            return inc, lib
    return None, None


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

    if system == "Darwin":
        omp_inc, _ = get_macos_libomp()
        if omp_inc:
            args += ["-Xpreprocessor", "-fopenmp", f"-I{omp_inc}"]

    if machine in ("x86_64", "amd64"):
        args += ["-mavx2", "-funroll-loops", "-DFORCE_AVX2"]

    return args


def get_link_args():
    system = platform.system()

    if system == "Linux":
        return ["-fopenmp"]

    if system == "Darwin":
        _, omp_lib = get_macos_libomp()
        if omp_lib:
            return [f"-L{omp_lib}", "-lomp"]

    return []


ext = Extension(
    "rns_engine._core",
    sources=["src/rns_engine/_core.cpp"],
    include_dirs=[pybind11.get_include(), np.get_include()],
    extra_compile_args=get_compile_args(),
    extra_link_args=get_link_args(),
    language="c++",
)

setup(
    name="rns_engine",
    version="0.4.0rc1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=[ext],
    include_package_data=True,
    zip_safe=False,
)
