from importlib.metadata import version

import rns_engine


def test_runtime_version_matches_distribution_metadata():
    assert rns_engine.__version__ == version("rns_engine")
    assert rns_engine.__version__ != "0+unknown"
