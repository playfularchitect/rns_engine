from __future__ import annotations

import base64
import lzma
import hashlib

import pytest

from rns_engine.g4_benchmark import _data_file, _runtime_metadata, g4_benchmark


def test_bad_mode_rejected_before_gpu_probe():
    with pytest.raises(ValueError):
        g4_benchmark("not-a-mode", display=False)


def test_packaged_replay_payload_decodes_to_validated_binary():
    meta = _runtime_metadata()
    parts = []
    for name in meta["payload_parts"]:
        part = _data_file(name).read_bytes()
        assert hashlib.sha256(part).hexdigest() == meta["payload_part_sha256"][name]
        parts.append(part)
    payload = b"".join(parts)

    assert hashlib.sha256(payload).hexdigest() == meta["payload_sha256"]

    binary = lzma.decompress(base64.b64decode(payload))
    assert hashlib.sha256(binary).hexdigest() == meta["binary_sha256"]
    assert meta["binary_sha256"] == "addce9d253b67f18944d558894e0d2e3273a9bd8de5c78f969bbc6bc5e763a66"
    assert meta["modes"] == {"quick": 24, "standard": 128, "full": 1024}
