"""Public G4 Series 1 benchmark entry points."""
from __future__ import annotations

import sys
from typing import TextIO

from .g4_integer_benchmark import _g4_integer_benchmark, g4_integer_benchmark
from .g4_trust import g4_benchmark as _legacy_rational_benchmark, _sha256_json
from .g4_xops import build_xops_summary, format_xops, print_xops_key, print_xops_summary

_ALLOWED_MODES = {"quick", "standard", "full"}


class _WordingStream:
    """Update legacy rational-only wrapper wording without touching benchmark math."""

    def __init__(self, target: TextIO):
        self._target = target

    @staticmethod
    def _replace(text: str) -> str:
        return text.replace(
            "g4_benchmark() does not rerun the integer benchmark; the integer score is context only.",
            "g4_rational_benchmark() reruns the rational benchmark only; the integer score above is context.",
        )

    def write(self, text: str) -> int:
        return self._target.write(self._replace(text))

    def flush(self) -> None:
        self._target.flush()


def _box(title: str, lines: list[str], stream: TextIO, width: int = 112) -> None:
    inner = width - 4
    border = "+" + "-" * (width - 2) + "+"
    print(border, file=stream)
    print("| " + title[:inner].ljust(inner) + " |", file=stream)
    print(border, file=stream)
    for line in lines:
        print("| " + line[:inner].ljust(inner) + " |", file=stream)
    print(border, file=stream)


def _validate_mode(mode: str) -> str:
    try:
        selected = mode.strip().lower()
    except AttributeError as exc:
        raise ValueError("mode must be quick, standard, or full") from exc
    if selected not in _ALLOWED_MODES:
        raise ValueError(f"unknown G4 benchmark mode {mode!r}; choose quick, standard, or full")
    return selected


def _g4_rational_benchmark(
    mode: str = "quick",
    *,
    display: bool = True,
    stream: TextIO | None = None,
    show_xops_key: bool = True,
) -> dict:
    selected = _validate_mode(mode)
    out = stream if stream is not None else sys.stdout
    if display and show_xops_key:
        print_xops_key(out)
        print(file=out)

    result = _legacy_rational_benchmark(
        selected,
        display=display,
        stream=_WordingStream(out) if display else out,
    )
    xops = build_xops_summary(
        result["rows"],
        species="G4_RATIONALS",
        headline_time_key="fraction_end_to_end_median_ms",
        kernel_time_key="exact_median_ms",
        timing_boundary=(
            "exact rational end-to-end median: exact GEMM execution plus rational metadata/bookkeeping; "
            "2*M*N*K XOPs are credited while the bookkeeping remains inside timing"
        ),
    )
    result["xops"] = xops
    result["xops_receipt_sha256"] = xops["xops_receipt_sha256"]

    if display:
        print(file=out)
        print_xops_summary("XOPS / G4OPS - G4 RATIONALS", xops, out)
    return result


def g4_rational_benchmark(
    mode: str = "quick",
    *,
    display: bool = True,
    stream: TextIO | None = None,
) -> dict:
    """Rerun the frozen G4 Series 1 exact-rational benchmark on a Tesla T4."""
    return _g4_rational_benchmark(mode, display=display, stream=stream, show_xops_key=True)


def g4_benchmark(
    mode: str = "quick",
    *,
    display: bool = True,
    stream: TextIO | None = None,
) -> dict:
    """Run both public G4 Series 1 benchmarks: integers first, then rationals."""
    selected = _validate_mode(mode)
    out = stream if stream is not None else sys.stdout

    if display:
        _box(
            "G4 SERIES 1 - COMBINED PUBLIC TESLA T4 BENCHMARK",
            [
                "This call runs two separate benchmarks in order: G4 INTEGERS first, then G4 RATIONALS.",
                "Their scores remain separate; the combined call is a convenience runner, not a combined win percentage.",
                f"Mode: {selected}",
            ],
            out,
        )
        print(file=out)
        print_xops_key(out)
        print(file=out)
        print("=== PART 1 / 2: G4 INTEGERS ===", file=out, flush=True)

    integer = _g4_integer_benchmark(
        selected,
        display=display,
        stream=out,
        show_xops_key=False,
    )

    if display:
        print(file=out)
        print("=== PART 2 / 2: G4 RATIONALS ===", file=out, flush=True)

    rational = _g4_rational_benchmark(
        selected,
        display=display,
        stream=out,
        show_xops_key=False,
    )

    receipt_material = {
        "schema": "RNS-ENGINE-G4S1-COMBINED-BENCHMARK-1",
        "mode": selected,
        "integer_run_receipt_sha256": integer["trust_pack"]["run_receipt_sha256"],
        "integer_xops_receipt_sha256": integer["xops"]["xops_receipt_sha256"],
        "rational_run_receipt_sha256": rational["trust_pack"]["run_receipt_sha256"],
        "rational_xops_receipt_sha256": rational["xops"]["xops_receipt_sha256"],
    }
    combined_receipt = _sha256_json(receipt_material)
    result = {
        "schema": "RNS-ENGINE-G4S1-COMBINED-BENCHMARK-1",
        "series": 1,
        "mode": selected,
        "integer": integer,
        "rational": rational,
        "combined_receipt_sha256": combined_receipt,
    }

    if display:
        isum = integer["summary"]; rsum = rational["summary"]
        _box(
            "G4 SERIES 1 - COMBINED RUN SUMMARY (SCORES REMAIN SEPARATE)",
            [
                f"G4 INTEGERS:  {isum['live_exact_wins']} G4 wins | {isum['live_floating_wins']} NVIDIA wins | {isum['live_statistical_ties']} ties | {format_xops(integer['xops']['suite_g4ops_per_second'])} suite G4OPS",
                f"G4 RATIONALS: {rsum['live_exact_wins']} G4 wins | {rsum['live_floating_wins']} NVIDIA wins | {rsum['live_statistical_ties']} ties | {format_xops(rational['xops']['suite_g4ops_per_second'])} suite G4OPS",
                "No combined win percentage is reported because integer and rational are different benchmarks.",
                f"Combined receipt SHA-256: {combined_receipt}",
            ],
            out,
        )
    return result


__all__ = ["g4_benchmark", "g4_integer_benchmark", "g4_rational_benchmark"]
