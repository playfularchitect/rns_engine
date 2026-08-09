"""Integrity-checked loader for the certified public G4 Series 1 Tesla T4 runtime."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import lzma
import os
from importlib import resources
from pathlib import Path
import tarfile
import tempfile

from .g4_benchmark import _detect_t4

_META_FILE = "g4s1_public_t4_runtime.json"
_SCHEMA = "RNS-ENGINE-G4S1-PUBLIC-T4-RUNTIME-2"
_META_CACHE: dict | None = None
_EXTRACTED_CACHE: Path | None = None
_SHAPE_CACHE: dict | None = None
_CERT_CACHE: dict | None = None
_GPU_CACHE: dict | None = None


def _data_file(name: str):
    return resources.files(__package__).joinpath("data", name)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def runtime_metadata() -> dict:
    global _META_CACHE
    if _META_CACHE is not None:
        return _META_CACHE
    try:
        with _data_file(_META_FILE).open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "this rns_engine release does not include the certified G4 Series 1 public T4 runtime"
        ) from exc
    if meta.get("schema") != _SCHEMA:
        raise RuntimeError("unrecognized G4 Series 1 public T4 runtime metadata")
    if meta.get("hardware") != "Tesla T4" or meta.get("compute_capability") != "7.5":
        raise RuntimeError("G4 Series 1 public runtime metadata has an unexpected hardware contract")
    _META_CACHE = meta
    return meta


def _bundle_bytes(meta: dict) -> bytes:
    parts = []
    for name in meta["payload_parts"]:
        raw = _data_file(name).read_bytes()
        expected = meta["payload_part_sha256"][name]
        actual = _sha256(raw)
        if actual != expected:
            raise RuntimeError(
                f"G4 Series 1 public runtime payload part failed SHA-256: {name}; "
                f"expected {expected}, got {actual}"
            )
        parts.append(raw)
    payload = b"".join(parts)
    if _sha256(payload) != meta["payload_sha256"]:
        raise RuntimeError("G4 Series 1 public runtime failed reassembled payload SHA-256")
    return payload


def _decode_tar(meta: dict) -> bytes:
    payload = _bundle_bytes(meta)
    try:
        xz_bytes = base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise RuntimeError("G4 Series 1 public runtime payload could not be base64 decoded") from exc
    if _sha256(xz_bytes) != meta["xz_sha256"]:
        raise RuntimeError("G4 Series 1 public runtime compressed archive failed SHA-256")
    try:
        tar_bytes = lzma.decompress(xz_bytes)
    except Exception as exc:
        raise RuntimeError("G4 Series 1 public runtime compressed archive could not be decompressed") from exc
    if _sha256(tar_bytes) != meta["tar_sha256"]:
        raise RuntimeError("G4 Series 1 public runtime tar archive failed SHA-256")
    return tar_bytes


def extracted_runtime_root() -> Path:
    global _EXTRACTED_CACHE
    if _EXTRACTED_CACHE is not None:
        return _EXTRACTED_CACHE
    meta = runtime_metadata()
    root = Path(tempfile.gettempdir()) / "rns_engine_g4s1" / meta["tar_sha256"]
    expected = meta["members_sha256"]

    complete = root / ".complete"
    if complete.exists():
        try:
            if complete.read_text(encoding="ascii").strip() == meta["tar_sha256"]:
                valid = True
                for name, sha in expected.items():
                    path = root / name
                    if not path.is_file() or _sha256(path.read_bytes()) != sha:
                        valid = False
                        break
                if valid:
                    _EXTRACTED_CACHE = root
                    return root
        except OSError:
            pass

    tar_bytes = _decode_tar(meta)
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if set(names) != set(expected):
            missing = sorted(set(expected) - set(names))
            extra = sorted(set(names) - set(expected))
            raise RuntimeError(f"G4 runtime member manifest mismatch; missing={missing}, extra={extra}")
        for member in members:
            if not member.isfile() or member.issym() or member.islnk():
                raise RuntimeError(f"G4 runtime contains a non-regular member: {member.name}")
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise RuntimeError(f"G4 runtime contains an unsafe member path: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"G4 runtime member could not be read: {member.name}")
            raw = handle.read()
            if _sha256(raw) != expected[member.name]:
                raise RuntimeError(f"G4 runtime member failed SHA-256: {member.name}")
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + f".tmp-{os.getpid()}")
            temp.write_bytes(raw)
            temp.chmod(0o700 if member.name.startswith(("integer/", "matmul/")) else 0o600)
            os.replace(temp, target)
    complete.write_text(meta["tar_sha256"] + "\n", encoding="ascii")
    _EXTRACTED_CACHE = root
    return root


def runtime_member(name: str) -> Path:
    meta = runtime_metadata()
    if name not in meta["members_sha256"]:
        raise KeyError(name)
    return extracted_runtime_root() / name


def integer_binary(family: str) -> Path:
    meta = runtime_metadata()
    try:
        name = meta["integer_members"][family]
    except KeyError as exc:
        raise RuntimeError(f"G4 Series 1 integer runtime is missing family {family!r}") from exc
    return runtime_member(name)


def matmul_library(family: str) -> Path:
    meta = runtime_metadata()
    try:
        name = meta["matmul_members"][family]
    except KeyError as exc:
        raise RuntimeError(f"G4 Series 1 matmul runtime is missing family {family!r}") from exc
    return runtime_member(name)


def shape_map() -> dict[tuple[int, int, int], dict]:
    global _SHAPE_CACHE
    if _SHAPE_CACHE is not None:
        return _SHAPE_CACHE
    meta = runtime_metadata()
    payload = json.loads(runtime_member(meta["shape_manifest_member"]).read_text(encoding="utf-8"))
    if payload.get("schema") != "RNS-ENGINE-G4S1-MATMUL-SHAPES-1":
        raise RuntimeError("unrecognized G4 Series 1 shape manifest")
    result: dict[tuple[int, int, int], dict] = {}
    for row in payload["shapes"]:
        key = (int(row["m"]), int(row["n"]), int(row["k"]))
        if key in result:
            raise RuntimeError(f"duplicate G4 Series 1 shape in installed manifest: {key}")
        result[key] = row
    if len(result) != int(meta["supported_shapes"]):
        raise RuntimeError(
            f"G4 Series 1 shape manifest expected {meta['supported_shapes']} unique shapes, got {len(result)}"
        )
    _SHAPE_CACHE = result
    return result


def certification_summary() -> dict:
    global _CERT_CACHE
    if _CERT_CACHE is not None:
        return _CERT_CACHE
    meta = runtime_metadata()
    payload = json.loads(runtime_member(meta["certification_member"]).read_text(encoding="utf-8"))
    if payload.get("schema") != "RNS-ENGINE-G4S1-PUBLIC-T4-CERTIFICATION-1":
        raise RuntimeError("unrecognized G4 Series 1 certification summary")
    _CERT_CACHE = payload
    return payload


def ensure_t4() -> dict:
    global _GPU_CACHE
    if _GPU_CACHE is None:
        _GPU_CACHE = _detect_t4()
    return dict(_GPU_CACHE)


__all__ = [
    "runtime_metadata",
    "extracted_runtime_root",
    "runtime_member",
    "integer_binary",
    "matmul_library",
    "shape_map",
    "certification_summary",
    "ensure_t4",
]
