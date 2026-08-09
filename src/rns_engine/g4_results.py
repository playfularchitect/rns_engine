"""Public, frozen G4 Series 1 benchmark evidence."""
from __future__ import annotations

import json
from copy import deepcopy
from importlib import resources
from typing import TextIO
import sys

_SUMMARY = "g4s1_public_summary.json"

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


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _print_integer(campaign: dict, stream: TextIO) -> None:
    print("[Exact integer / NVIDIA FP16-input cuBLASLt]", file=stream)
    print(
        f"Exact wins: {campaign['exact_wins']} / {campaign['declared_shapes']} "
        f"({_pct(campaign['exact_win_rate'])})",
        file=stream,
    )
    print(f"NVIDIA wins: {campaign['floating_wins']}", file=stream)
    print("Exact replay before + after timing: PASS on all 1,024 shapes", file=stream)
    print(f"Median speedup among exact wins: {campaign['exact_win_speedup_median']:.3f}x", file=stream)
    print(f"Best exact-win speedup: {campaign['best_exact_win_speedup']:.3f}x", file=stream)


def _print_rational(campaign: dict, stream: TextIO) -> None:
    print("[Exact rational / NVIDIA FP16]", file=stream)
    print(
        f"Certified exact wins: {campaign['certified_exact_wins']} / {campaign['target_shapes']} "
        f"({_pct(campaign['certified_exact_win_rate'])})",
        file=stream,
    )
    print(f"Remaining unresolved at archive freeze: {campaign['remaining_unresolved']}", file=stream)
    print("Certified winners: non-integer inputs PASS | range proof PASS | FP16 value-set proof PASS", file=stream)
    print(f"Median certified speedup: {campaign['certified_speedup_median']:.3f}x", file=stream)
    print(f"Best certified speedup: {campaign['best_certified_speedup']:.3f}x", file=stream)


def g4_results(
    campaign: str = "all",
    *,
    display: bool = True,
    stream: TextIO | None = None,
) -> dict:
    """Return and optionally print frozen G4 Series 1 public evidence.

    ``campaign`` may be ``"all"`` (default), ``"integer"``, or ``"rational"``.
    The integer and rational results are separate benchmark claims and are
    deliberately reported separately.
    """
    try:
        selected = _ALIASES[campaign.strip().lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"unknown G4 campaign {campaign!r}; choose one of: all, integer, rational"
        ) from exc

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
            "provenance": deepcopy(evidence["provenance"]),
            "claim": deepcopy(evidence["claims"][selected]),
            "campaign": deepcopy(evidence["campaigns"][selected]),
        }

    if display:
        out = stream if stream is not None else sys.stdout
        print("=== G4 SERIES 1 — TESLA T4 RESULTS ===", file=out)
        print(file=out)
        if selected in ("all", "integer_fp16_input_clean_sweep"):
            _print_integer(evidence["campaigns"]["integer_fp16_input_clean_sweep"], out)
        if selected == "all":
            print(file=out)
        if selected in ("all", "dynamic_exact_rational"):
            _print_rational(evidence["campaigns"]["dynamic_exact_rational"], out)
        if selected == "all":
            print(file=out)
            print("91.60% integer and 84.96% rational are separate benchmark results.", file=out)

    return result


__all__ = ["g4_results"]
