"""Public, frozen G4 Series 1 benchmark evidence."""
from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import json
from copy import deepcopy
from importlib import resources
from typing import TextIO
import sys

_SUMMARY = "g4s1_public_summary.json"
_INTEGER_ROWS = "g4s1_integer_fp16_input_results.csv"
_RATIONAL_ROWS = "g4s1_dynamic_exact_rational_results.csv"
_LEDGER_PACKAGING = "g4s1_ledger_packaging.json"

_BOOL_FIELDS = {"certified_exact_win", "actual_noninteger_inputs", "range_proved", "fp16_value_set_proved", "exact_replay_pre", "exact_replay_post"}
_INT_FIELDS = {"m", "n", "k", "exact_block_wins", "paired_blocks"}
_FLOAT_FIELDS = {"speedup_floating_over_exact", "bootstrap_low", "bootstrap_high", "exact_median_ms", "floating_median_ms", "speedup_fp16_over_exact", "fp16_median_ms"}

_ALIASES = {
    "all": "all",
    "integer": "integer_fp16_input_clean_sweep",
    "int": "integer_fp16_input_clean_sweep",
    "clean": "integer_fp16_input_clean_sweep",
    "rational": "dynamic_exact_rational",
    "exact-rational": "dynamic_exact_rational",
    "g416": "dynamic_exact_rational",
}


def _data_file(name: str):
    return resources.files(__package__).joinpath("data", name)


def _load_summary() -> dict:
    with _data_file(_SUMMARY).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_row(row: dict[str, str]) -> dict:
    typed = {}
    for key, value in row.items():
        if key in _BOOL_FIELDS:
            typed[key] = value == "True"
        elif key in _INT_FIELDS and value != "":
            typed[key] = int(value)
        elif key in _FLOAT_FIELDS and value != "":
            typed[key] = float(value)
        else:
            typed[key] = value
    return typed


def _load_rows(name: str) -> list[dict]:
    filename = _INTEGER_ROWS if name == "integer_fp16_input_clean_sweep" else _RATIONAL_ROWS
    with _data_file(_LEDGER_PACKAGING).open("r", encoding="utf-8") as handle:
        packaging = json.load(handle)
    if packaging.get("schema") != "RNS-ENGINE-G4S1-LEDGER-PACKAGING-1":
        raise RuntimeError("unrecognized G4 Series 1 ledger packaging schema")
    entry = packaging["ledgers"][filename]
    part_names = entry.get("packaged_parts")
    part_hashes = entry.get("packaged_part_sha256", {})
    if not isinstance(part_names, list) or not part_names:
        raise RuntimeError(f"packaged G4 ledger part manifest is missing: {filename}")
    parts = []
    for part_name in part_names:
        part = _data_file(part_name).read_bytes()
        if hashlib.sha256(part).hexdigest() != part_hashes.get(part_name):
            raise RuntimeError(f"packaged G4 ledger part failed integrity check: {part_name}")
        parts.append(part)
    payload = b"".join(parts)
    if hashlib.sha256(payload).hexdigest() != entry["packaged_sha256"]:
        raise RuntimeError(f"packaged G4 ledger failed integrity check after reassembly: {filename}")
    try:
        raw = gzip.decompress(base64.b64decode(payload))
    except Exception as exc:
        raise RuntimeError(f"packaged G4 ledger could not be decoded: {filename}") from exc
    if hashlib.sha256(raw).hexdigest() != entry["raw_sha256"]:
        raise RuntimeError(f"decoded G4 ledger failed frozen hash check: {filename}")
    text = raw.decode("utf-8")
    return [_coerce_row(row) for row in csv.DictReader(io.StringIO(text))]


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _print_integer(campaign: dict, stream: TextIO) -> None:
    print("[Exact integer / FP16-input cuBLASLt clean sweep]", file=stream)
    print(
        f"Exact wins: {campaign['exact_wins']} / {campaign['declared_shapes']} "
        f"({_pct(campaign['exact_win_rate'])})",
        file=stream,
    )
    print(f"Floating wins: {campaign['floating_wins']}", file=stream)
    print("Exact replay before + after timing: PASS on all 1,024 shapes", file=stream)
    print(f"Median speedup among exact wins: {campaign['exact_win_speedup_median']:.3f}x", file=stream)
    print(f"Best exact-win speedup: {campaign['best_exact_win_speedup']:.3f}x", file=stream)
    print(f"Baseline: {campaign['floating_baseline']}", file=stream)


def _print_rational(campaign: dict, stream: TextIO) -> None:
    print("[Dynamic exact rational / FP16 campaign]", file=stream)
    print(
        f"Certified exact wins: {campaign['certified_exact_wins']} / {campaign['target_shapes']} "
        f"({_pct(campaign['certified_exact_win_rate'])})",
        file=stream,
    )
    print(f"Remaining unresolved at archive freeze: {campaign['remaining_unresolved']}", file=stream)
    print("Certified winners use 31 paired timing blocks", file=stream)
    print("Certified winners: non-integer inputs PASS | range proof PASS | FP16 value-set proof PASS", file=stream)
    print(f"Median certified speedup: {campaign['certified_speedup_median']:.3f}x", file=stream)
    print(f"Best certified speedup: {campaign['best_certified_speedup']:.3f}x", file=stream)


def _shape_rows(campaign_name: str, shape_id: str) -> list[dict]:
    wanted = shape_id.upper()
    rows = _load_rows(campaign_name)
    return [row for row in rows if row.get("shape_id", "").upper() == wanted]


def g4_results(
    campaign: str = "all",
    *,
    shape_id: str | None = None,
    display: bool = True,
    stream: TextIO | None = None,
) -> dict:
    """Return and optionally print frozen G4 Series 1 public evidence.

    Parameters
    ----------
    campaign:
        ``"all"`` (default), ``"integer"`` or ``"rational"``.
    shape_id:
        Optional frozen suite ID such as ``"T4GL0021"``. When supplied, the
        matching sanitized row is attached under ``"shape_rows"``.
    display:
        Print a compact human-readable report. The structured evidence is
        returned regardless.
    stream:
        Optional text stream for printed output; defaults to stdout.

    Notes
    -----
    The 938/1024 figure is the exact-integer clean-sweep result. The 870/1024
    figure is the separately certified dynamic exact-rational result. They are
    deliberately reported as different campaigns.
    """
    try:
        selected = _ALIASES[campaign.strip().lower()]
    except (AttributeError, KeyError) as exc:
        choices = "all, integer, rational"
        raise ValueError(f"unknown G4 campaign {campaign!r}; choose one of: {choices}") from exc

    evidence = _load_summary()
    if evidence.get("schema") != "RNS-ENGINE-G4S1-PUBLIC-EVIDENCE-1":
        raise RuntimeError("unrecognized G4 Series 1 public-evidence schema")

    if selected == "all":
        result = deepcopy(evidence)
    else:
        result = {
            "schema": evidence["schema"],
            "series": evidence["series"],
            "freeze_status": evidence["freeze_status"],
            "privacy_boundary": deepcopy(evidence["privacy_boundary"]),
            "provenance": deepcopy(evidence["provenance"]),
            "claim": deepcopy(evidence["claims"][selected]),
            "campaign": deepcopy(evidence["campaigns"][selected]),
        }

    if shape_id is not None:
        names = (
            ["integer_fp16_input_clean_sweep", "dynamic_exact_rational"]
            if selected == "all"
            else [selected]
        )
        matches = []
        for name in names:
            for row in _shape_rows(name, shape_id):
                matches.append({"campaign": name, **row})
        if not matches:
            raise KeyError(f"shape_id {shape_id!r} was not found in the frozen G4 Series 1 public ledger")
        result["shape_rows"] = matches

    if display:
        out = stream if stream is not None else sys.stdout
        print("=== G4 SERIES 1 — FROZEN TESLA T4 RESULTS ===", file=out)
        print(file=out)
        if selected in ("all", "integer_fp16_input_clean_sweep"):
            _print_integer(evidence["campaigns"]["integer_fp16_input_clean_sweep"], out)
        if selected == "all":
            print(file=out)
        if selected in ("all", "dynamic_exact_rational"):
            _print_rational(evidence["campaigns"]["dynamic_exact_rational"], out)
        if selected == "all":
            print(file=out)
            print("Important: 91.60% integer coverage and 84.96% rational certification are separate claims.", file=out)
        if shape_id is not None:
            print(file=out)
            print(f"Shape {shape_id.upper()}:", file=out)
            for row in result["shape_rows"]:
                print(json.dumps(row, sort_keys=True), file=out)

    return result


__all__ = ["g4_results"]
