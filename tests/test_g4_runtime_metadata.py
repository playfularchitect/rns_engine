import rns_engine.g4_runtime as runtime


def test_integer_replay_trust_alias_matches_certified_bundle():
    meta = runtime.runtime_metadata()
    assert meta["integer_replay_bundle_sha256"] == meta["certified_integer_replay_bundle_sha256"]


def test_public_runtime_compile_contract_matches_v7_t4_build():
    meta = runtime.runtime_metadata()
    assert meta["compile"]["arch"] == "sm_75"
    assert meta["compile"]["common"] == [
        "-O3",
        "-std=c++17",
        "-arch=sm_75",
        "-Xcompiler=-ffunction-sections",
        "-Xcompiler=-fdata-sections",
        "-Xlinker=--gc-sections",
    ]
    assert meta["compile"]["libraries"] == ["-lcublasLt", "-lcublas"]
    assert meta["compile"]["shared_extra"] == ["-shared", "-Xcompiler=-fPIC"]
