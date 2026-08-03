from importlib.metadata import version
from pathlib import Path
import tomllib

import rns_engine


def _pyproject():
    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_runtime_version_matches_distribution_metadata():
    assert rns_engine.__version__ == version("rns_engine")
    assert rns_engine.__version__ != "0+unknown"


def test_python_314_is_declared_and_built():
    project = _pyproject()
    metadata = project["project"]
    cibuildwheel = project["tool"]["cibuildwheel"]

    assert metadata["requires-python"] == ">=3.10,<3.15"
    assert "Programming Language :: Python :: 3.14" in metadata["classifiers"]
    assert "cp314-*" in cibuildwheel["build"]


def test_python_314_uses_compatible_native_build_dependencies():
    project = _pyproject()
    build_requires = project["build-system"]["requires"]
    test_requires = project["project"]["optional-dependencies"]["test"]

    assert "pybind11>=3.0.0" in build_requires
    assert "numpy==2.4.5; python_version >= '3.14'" in build_requires
    assert "numpy==2.4.5; python_version >= '3.14'" in test_requires
