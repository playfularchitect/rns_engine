import platform
from setuptools import setup, Extension, find_packages
import pybind11
import numpy as np


def get_compile_args():
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Windows":
        args = ["/std:c++17", "/O2", "/DNDEBUG"]
        if machine in ("x86_64", "amd64"):
            args += ["/D_FORCE_AVX2"]
        return args

    args = ["-std=c++17", "-O3", "-DNDEBUG"]

    # OpenMP on Unix-like platforms
    if system in ("Linux", "Darwin"):
        args += ["-fopenmp"]

    # AVX2 on x86_64 only
    if machine in ("x86_64", "amd64"):
        args += ["-mavx2", "-funroll-loops", "-DFORCE_AVX2"]

    return args


def get_link_args():
    system = platform.system()

    if system == "Windows":
        return []

    if system in ("Linux", "Darwin"):
        return ["-fopenmp"]

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
