import hashlib

import rns_engine.g4_runtime as runtime


def test_every_declared_source_payload_chunk_has_expected_sha256():
    meta = runtime.runtime_metadata()
    for name in meta["payload_parts"]:
        raw = runtime._data_file(name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == meta["payload_part_sha256"][name]
