"""Exact-operations-per-second accounting for public G4 GEMM benchmarks."""
from __future__ import annotations

import hashlib
import json
import statistics
from typing import Iterable, TextIO

_XOPS_SCHEMA = "RNS-ENGINE-XOPS-1"


def gemm_xop_count(m: int, n: int, k: int) -> int:
    """Return the conventional GEMM operation count, 2*M*N*K.

    XOPS intentionally uses the same operation-count convention as GEMM FLOPS so
    exact and floating-point throughput can be compared without changing the
    accounting rule.
    """
    m = int(m); n = int(n); k = int(k)
    if m < 0 or n < 0 or k < 0:
        raise ValueError("GEMM dimensions must be non-negative")
    return 2 * m * n * k


def xops_per_second(m: int, n: int, k: int, milliseconds: float) -> float:
    milliseconds = float(milliseconds)
    if milliseconds <= 0:
        raise ValueError("milliseconds must be > 0")
    return gemm_xop_count(m, n, k) * 1000.0 / milliseconds


def format_xops(rate: float) -> str:
    rate = float(rate)
    units = (
        (1e18, "EXOPS"),
        (1e15, "PXOPS"),
        (1e12, "TXOPS"),
        (1e9, "GXOPS"),
        (1e6, "MXOPS"),
        (1e3, "kXOPS"),
    )
    for scale, suffix in units:
        if abs(rate) >= scale:
            return f"{rate / scale:.3f} {suffix}"
    return f"{rate:.3f} XOPS"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def build_xops_summary(
    rows: Iterable[dict],
    *,
    species: str,
    headline_time_key: str,
    timing_boundary: str,
    kernel_time_key: str | None = None,
) -> dict:
    """Build an exact-throughput summary from per-shape benchmark rows.

    A shape receives XOP credit only when ``exact_replay_passed`` is true.
    Failed exactness therefore contributes zero credited XOPs. Public benchmark
    runners fail closed on such a row, but the accounting law is explicit here
    as well.
    """
    materialized = list(rows)
    per_shape = []
    credited_ops = 0
    total_headline_seconds = 0.0
    total_kernel_seconds = 0.0
    headline_rates = []
    kernel_rates = []
    exact_rows = 0

    for row in materialized:
        m = int(row["m"]); n = int(row["n"]); k = int(row["k"])
        ops = gemm_xop_count(m, n, k)
        exact = bool(row.get("exact_replay_passed"))
        headline_ms = float(row[headline_time_key])
        if headline_ms <= 0:
            raise RuntimeError(f"{row.get('shape_id', '<unknown>')} has non-positive exact timing")
        total_headline_seconds += headline_ms / 1000.0

        headline_rate = 0.0
        if exact:
            exact_rows += 1
            credited_ops += ops
            headline_rate = ops * 1000.0 / headline_ms
            headline_rates.append(headline_rate)

        kernel_rate = None
        if kernel_time_key is not None:
            kernel_ms = float(row[kernel_time_key])
            if kernel_ms <= 0:
                raise RuntimeError(f"{row.get('shape_id', '<unknown>')} has non-positive kernel timing")
            total_kernel_seconds += kernel_ms / 1000.0
            if exact:
                kernel_rate = ops * 1000.0 / kernel_ms
                kernel_rates.append(kernel_rate)

        per_shape.append(
            {
                "shape_id": row.get("shape_id"),
                "m": m,
                "n": n,
                "k": k,
                "xops_credited": ops if exact else 0,
                "g4ops_per_second": headline_rate,
                **({"kernel_g4ops_per_second": kernel_rate or 0.0} if kernel_time_key else {}),
            }
        )

    suite_rate = credited_ops / total_headline_seconds if total_headline_seconds > 0 else 0.0
    summary = {
        "schema": _XOPS_SCHEMA,
        "species": species,
        "definitions": {
            "XOP": "one mathematically exact arithmetic operation",
            "XOPS": "exact arithmetic operations per second",
            "G4OPS": "XOPS delivered by a G4 implementation",
        },
        "gemm_counting_rule": "2*M*N*K XOPs per GEMM, matching the conventional GEMM FLOPS count",
        "exactness_rule": "a shape that fails exactness receives 0 XOP credit",
        "headline_timing_boundary": timing_boundary,
        "rows_total": len(materialized),
        "rows_exact_eligible": exact_rows,
        "xops_credited": credited_ops,
        "suite_g4ops_per_second": suite_rate,
        "median_shape_g4ops_per_second": statistics.median(headline_rates) if headline_rates else 0.0,
        "peak_shape_g4ops_per_second": max(headline_rates, default=0.0),
        "per_shape": per_shape,
    }
    if kernel_time_key is not None:
        summary.update(
            {
                "kernel_timing_key": kernel_time_key,
                "suite_kernel_g4ops_per_second": credited_ops / total_kernel_seconds if total_kernel_seconds > 0 else 0.0,
                "median_shape_kernel_g4ops_per_second": statistics.median(kernel_rates) if kernel_rates else 0.0,
                "peak_shape_kernel_g4ops_per_second": max(kernel_rates, default=0.0),
            }
        )

    receipt_material = dict(summary)
    receipt_material.pop("per_shape")
    receipt_material["per_shape_sha256"] = hashlib.sha256(_canonical_json_bytes(per_shape)).hexdigest()
    summary["per_shape_sha256"] = receipt_material["per_shape_sha256"]
    summary["xops_receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(receipt_material)).hexdigest()
    return summary


def _box(title: str, lines: list[str], stream: TextIO, width: int = 112) -> None:
    inner = width - 4
    border = "+" + "-" * (width - 2) + "+"
    print(border, file=stream)
    print("| " + title[:inner].ljust(inner) + " |", file=stream)
    print(border, file=stream)
    for line in lines:
        print("| " + line[:inner].ljust(inner) + " |", file=stream)
    print(border, file=stream)


def print_xops_key(stream: TextIO) -> None:
    _box(
        "XOPS / G4OPS KEY",
        [
            "XOP   = one mathematically exact arithmetic operation.",
            "XOPS  = exact arithmetic operations per second.",
            "G4OPS = XOPS delivered by a G4 implementation.",
            "GEMM accounting: 2*M*N*K XOPs, matching the conventional GEMM FLOPS counting rule.",
            "Exactness rule: a shape that fails exactness earns 0 XOPS.",
        ],
        stream,
    )


def print_xops_summary(title: str, summary: dict, stream: TextIO) -> None:
    lines = [
        f"Suite G4OPS:                {format_xops(summary['suite_g4ops_per_second'])}",
        f"Median per-shape G4OPS:     {format_xops(summary['median_shape_g4ops_per_second'])}",
        f"Peak observed G4OPS:        {format_xops(summary['peak_shape_g4ops_per_second'])}",
        f"Exact rows earning XOPS:    {summary['rows_exact_eligible']} / {summary['rows_total']}",
        f"Timing boundary:             {summary['headline_timing_boundary']}",
    ]
    if "suite_kernel_g4ops_per_second" in summary:
        lines.insert(1, f"Suite kernel-only G4OPS:    {format_xops(summary['suite_kernel_g4ops_per_second'])}")
    lines += [
        f"Per-shape XOPS SHA-256:      {summary['per_shape_sha256']}",
        f"XOPS receipt SHA-256:        {summary['xops_receipt_sha256']}",
    ]
    _box(title, lines, stream)


__all__ = [
    "gemm_xop_count",
    "xops_per_second",
    "format_xops",
    "build_xops_summary",
    "print_xops_key",
    "print_xops_summary",
]
