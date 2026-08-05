"""Reversible ACTG encoding for G4 genomes and canonical search objects."""

from __future__ import annotations

import json
from typing import Any, Mapping

_TO_BASE = {0: "A", 1: "C", 2: "T", 3: "G"}
_FROM_BASE = {base: value for value, base in _TO_BASE.items()}


def bytes_to_actg(payload: bytes) -> str:
    """Encode bytes with the frozen A=00, C=01, T=10, G=11 mapping."""

    return "".join(_TO_BASE[(byte >> shift) & 0b11] for byte in payload for shift in (6, 4, 2, 0))


def actg_to_bytes(genome: str) -> bytes:
    """Decode a four-bases-per-byte ACTG string."""

    normalized = "".join(genome.split()).upper()
    if len(normalized) % 4:
        raise ValueError("ACTG byte genomes must contain a multiple of four bases")
    try:
        values = [_FROM_BASE[base] for base in normalized]
    except KeyError as exc:
        raise ValueError(f"invalid ACTG base {exc.args[0]!r}") from exc
    output = bytearray()
    for offset in range(0, len(values), 4):
        byte = 0
        for value in values[offset : offset + 4]:
            byte = (byte << 2) | value
        output.append(byte)
    return bytes(output)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def object_to_actg(value: Any) -> str:
    return bytes_to_actg(canonical_json(value).encode("utf-8"))


def actg_to_object(genome: str) -> Any:
    return json.loads(actg_to_bytes(genome).decode("utf-8"))


def candidate_genome(
    *,
    features: tuple[str, ...],
    parameters: Mapping[str, Any],
    mutation_ops: tuple[str, ...] = (),
) -> str:
    return object_to_actg(
        {
            "features": sorted(features),
            "parameters": dict(sorted(parameters.items())),
            "mutation_ops": list(mutation_ops),
        }
    )


__all__ = [
    "bytes_to_actg",
    "actg_to_bytes",
    "canonical_json",
    "object_to_actg",
    "actg_to_object",
    "candidate_genome",
]
