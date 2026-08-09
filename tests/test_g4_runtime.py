import hashlib

import rns_engine.g4_runtime as runtime


def test_certified_public_runtime_payload_and_members_are_integrity_checked():
    meta = runtime.runtime_metadata()
    root = runtime.extracted_runtime_root()
    assert meta["schema"] == "RNS-ENGINE-G4S1-PUBLIC-T4-SOURCE-RUNTIME-1"
    assert meta["supported_shapes"] == 1024
    assert meta["privacy_boundary"].startswith("public execution sources only")
    for name, expected in meta["members_sha256"].items():
        raw = (root / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected


def test_shape_and_certification_manifests_are_complete():
    shapes = runtime.shape_map()
    cert = runtime.certification_summary()
    assert len(shapes) == 1024
    assert cert["integer_benchmark"]["full"]["exact_replay_correct"] == 1024
    assert cert["g4_matmul"]["all_1024_random_full_signed_int8"] == "PASS"
    assert cert["g4_matmul"]["quick_24_extreme_values"] == "PASS"
    assert cert["g4_matmul"]["standard_128_sparse_extremes"] == "PASS"
