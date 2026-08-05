"""Opt-in atomic persistent storage for G4 learning state."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import zipfile


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class G4Capsule:
    """A caller-selected directory; no unrelated filesystem access occurs."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).expanduser().resolve()
        self.state_dir = self.root / "state"
        self.experience_dir = self.root / "experiences"
        self.index_path = self.root / "INDEX.json"

    def initialize(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.experience_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._atomic_json(
                self.index_path,
                {
                    "schema": "rns_engine.g4_capsule.v1",
                    "state": {},
                    "experiences": {},
                },
            )

    def _atomic_bytes(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _atomic_json(self, path: Path, value: Any) -> str:
        payload = _json_bytes(value)
        self._atomic_bytes(path, payload)
        return _sha256(payload)

    def _read_index(self) -> dict[str, Any]:
        self.initialize()
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write_index(self, index: Mapping[str, Any]) -> None:
        self._atomic_json(self.index_path, dict(index))

    def save_state(self, state: Mapping[str, Any], *, name: str = "learner") -> str:
        self.initialize()
        path = self.state_dir / f"{name}.json"
        digest = self._atomic_json(path, dict(state))
        index = self._read_index()
        index.setdefault("state", {})[name] = {
            "path": str(path.relative_to(self.root)),
            "sha256": digest,
        }
        self._write_index(index)
        return digest

    def load_state(self, *, name: str = "learner") -> dict[str, Any] | None:
        self.initialize()
        index = self._read_index()
        entry = index.get("state", {}).get(name)
        if entry is None:
            return None
        path = self.root / entry["path"]
        payload = path.read_bytes()
        observed = _sha256(payload)
        if observed != entry["sha256"]:
            raise RuntimeError(f"capsule state checksum mismatch for {name}")
        return json.loads(payload.decode("utf-8"))

    def store_experience(self, experience: Mapping[str, Any]) -> str:
        self.initialize()
        payload = _json_bytes(dict(experience))
        digest = _sha256(payload)
        path = self.experience_dir / f"{digest}.json"
        if not path.exists():
            self._atomic_bytes(path, payload)
        index = self._read_index()
        index.setdefault("experiences", {})[digest] = {
            "path": str(path.relative_to(self.root)),
            "sha256": digest,
        }
        self._write_index(index)
        return digest

    def verify(self) -> dict[str, Any]:
        index = self._read_index()
        checked = 0
        for group in ("state", "experiences"):
            for entry in index.get(group, {}).values():
                payload = (self.root / entry["path"]).read_bytes()
                if _sha256(payload) != entry["sha256"]:
                    raise RuntimeError(f"capsule checksum mismatch: {entry['path']}")
                checked += 1
        return {"ok": True, "checked": checked, "root": str(self.root)}

    def export(self, destination: str | os.PathLike[str]) -> Path:
        self.verify()
        destination_path = Path(destination).expanduser().resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(self.root))
        return destination_path


__all__ = ["G4Capsule"]
