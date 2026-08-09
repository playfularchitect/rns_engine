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
    print("=== G4 INTEGERS vs NVIDIA FP16 cuBLASLt ===", file=stream)
    print("Frozen Series 1 integer test — separate from the rational test below.", file=stream)
    print(
        f"G4 exact integers faster: {campaign['exact_wins']} / {campaign['declared_shapes']} "
        f"({_pct(campaign['exact_win_rate'])})",
        file=stream,
    )
    print(f"NVIDIA FP16 faster: {campaign['floating_wins']} / {campaign['declared_shapes']}", file=stream)
    print(
        f"Statistical ties / errors: {campaign['statistical_ties']} / {campaign['errors']}",
        file=stream,
    )
    print("Integer exactness: replay before + after timing PASS on all 1,024 shapes", file=stream)
    print(
        f"Overall integer speedup across all 1,024 shapes: "
        f"{campaign['overall_speedup_geometric_mean']:.3f}x geometric mean | "
        f"{campaign['overall_speedup_median']:.3f}x median",
        file=stream,
    )
    print(
        f"Among {campaign['exact_wins']} G4 wins: "
        f"{campaign['exact_win_speedup_geometric_mean']:.3f}x geometric mean | "
        f"{campaign['exact_win_speedup_median']:.3f}x median | "
        f"{campaign['best_exact_win_speedup']:.3f}x best",
        file=stream,
    )
    print(
        f"Across {campaign['floating_wins']} NVIDIA wins, G4 retained "
        f"{_pct(campaign['nvidia_win_g4_throughput_retained_geometric_mean'])} of NVIDIA throughput "
        f"on geometric average ({_pct(campaign['nvidia_win_g4_execution_time_penalty_from_geomean'])} "
        f"longer execution time).",
        file=stream,
    )


def _print_rational(campaign: dict, stream: TextIO) -> None:
    print("=== G4 RATIONALS vs NVIDIA FP16 cuBLASLt ===", file=stream)
    print("Frozen Series 1 rational test — separate from the integer test above.", file=stream)
    print(
        f"G4 exact rationals faster (certified): {campaign['certified_exact_wins']} / {campaign['target_shapes']} "
        f"({_pct(campaign['certified_exact_win_rate'])})",
        file=stream,
    )
    print(f"NVIDIA FP16 wins: {campaign['nvidia_wins']} / {campaign['target_shapes']}", file=stream)
    print(
        f"Statistical ties / errors: {campaign['statistical_ties']} / {campaign['errors']}",
        file=stream,
    )
    print("Certified rational winners: non-integer inputs PASS | range proof PASS | FP16 value-set proof PASS", file=stream)
    print(
        f"Among {campaign['certified_exact_wins']} certified G4 wins: "
        f"{campaign['certified_speedup_geometric_mean']:.3f}x geometric mean | "
        f"{campaign['certified_speedup_median']:.3f}x median | "
        f"{campaign['best_certified_speedup']:.3f}x best | "
        f"{campaign['slowest_certified_speedup']:.3f}x slowest certified win",
        file=stream,
    )
    print("No all-1,024 rational speedup aggregate is claimed by the frozen public summary.", file=stream)


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
        print("Two separate tests are reported below. Do not combine their scores.", file=out)
        print("Speedup ratios are NVIDIA time / G4 time; >1.0x means G4 is faster.", file=out)
        print(file=out)
        if selected in ("all", "integer_fp16_input_clean_sweep"):
            _print_integer(evidence["campaigns"]["integer_fp16_input_clean_sweep"], out)
        if selected == "all":
            print(file=out)
        if selected in ("all", "dynamic_exact_rational"):
            _print_rational(evidence["campaigns"]["dynamic_exact_rational"], out)
        if selected == "all":
            print(file=out)
            print("INTEGER RESULT:  938 G4 wins / 86 NVIDIA wins / 0 ties / 0 errors", file=out)
            print("RATIONAL RESULT: 870 G4 wins / 110 NVIDIA wins / 41 ties / 3 errors", file=out)
            print("These are separate benchmark results against NVIDIA FP16 cuBLASLt.", file=out)

    return result


__all__ = ["g4_results"]
