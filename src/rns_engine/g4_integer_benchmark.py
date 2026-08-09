"""Public G4 Series 1 exact-integer replay benchmark on Tesla T4."""
from __future__ import annotations

import json
import queue
import random
import statistics
import subprocess
import sys
import threading
import time
from typing import TextIO

from .g4_benchmark import _BLOCKS, _BOOTSTRAP_REPETITIONS, _PROMOTION, _REQUIRED_WINS
from .g4_runtime import ensure_t4, integer_binary, runtime_metadata
from .g4_trust import (
    _build_cuda_summary,
    _current_cuda_version,
    _environment_name,
    _format_gpu_state,
    _gpu_state_snapshot,
    _package_version,
    _sha256_json,
    _utc_now,
)
from .g4_xops import build_xops_summary, print_xops_key, print_xops_summary

_ALLOWED_MODES = {"quick", "standard", "full"}
_FAMILIES = ("generic", "direct", "splitk", "fused", "strip", "row")
_BAR_WIDTH = 30
_SPINNER = "|/-\\"


def _box(title: str, lines: list[str], stream: TextIO, width: int = 112) -> None:
    inner = width - 4
    border = "+" + "-" * (width - 2) + "+"
    print(border, file=stream)
    print("| " + title[:inner].ljust(inner) + " |", file=stream)
    print(border, file=stream)
    for line in lines:
        print("| " + line[:inner].ljust(inner) + " |", file=stream)
    print(border, file=stream)


def _fmt_time(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--:--"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _progress_line(label: str, done: int, total: int, started: float, spin: int) -> str:
    elapsed = max(0.001, time.monotonic() - started)
    fraction = min(1.0, done / max(1, total))
    filled = int(_BAR_WIDTH * fraction)
    eta = None if done <= 0 else elapsed * (total - done) / done
    return (
        f"{_SPINNER[spin % len(_SPINNER)]} {label:<9} "
        f"[{'#' * filled}{'-' * (_BAR_WIDTH - filled)}] {done:4d}/{total} {fraction * 100:5.1f}%  "
        f"elapsed {_fmt_time(elapsed)}  ETA {_fmt_time(eta)}"
    )


def _redraw(stream: TextIO, line: str, previous_len: int) -> int:
    print("\r" + line.ljust(previous_len), end="", flush=True, file=stream)
    return max(previous_len, len(line))


def _reader_thread(stream, output_queue: queue.Queue) -> None:
    try:
        for line in stream:
            output_queue.put(line.rstrip("\n"))
    finally:
        output_queue.put(None)


def _stream_family(
    binary,
    mode: str,
    expected_total: int,
    done_before: int,
    started: float,
    *,
    display: bool,
    stream: TextIO,
) -> tuple[list[dict], list[dict]]:
    proc = subprocess.Popen(
        [str(binary), "--mode", mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    output_queue: queue.Queue = queue.Queue()
    threading.Thread(target=_reader_thread, args=(proc.stdout, output_queue), daemon=True).start()
    rows: list[dict] = []
    summaries: list[dict] = []
    unexpected: list[str] = []
    previous_len = 0
    spin = 0
    finished = False
    while not finished:
        try:
            item = output_queue.get(timeout=0.35)
        except queue.Empty:
            item = "__TICK__"
        if item is None:
            finished = True
        elif item == "__TICK__":
            pass
        elif item.startswith("RNS_G4S1_INTEGER_MEASUREMENT "):
            rows.append(json.loads(item.split(" ", 1)[1]))
        elif item.startswith("RNS_G4S1_INTEGER_SUMMARY "):
            summaries.append(json.loads(item.split(" ", 1)[1]))
        elif item:
            unexpected.append(item)
        if display:
            previous_len = _redraw(
                stream,
                _progress_line("GPU test", done_before + len(rows), expected_total, started, spin),
                previous_len,
            )
        spin += 1
    rc = proc.wait()
    if rc not in (0, 2):
        detail = " | ".join(unexpected[-8:]) if unexpected else "no additional error text"
        raise RuntimeError(f"{binary.name} failed with exit {rc}: {detail}")
    return rows, summaries


def _bootstrap_ci(values: list[float], seed: int) -> tuple[float, float]:
    if len(values) != _BLOCKS:
        raise RuntimeError(f"expected {_BLOCKS} paired ratios, got {len(values)}")
    rng = random.Random(seed)
    samples = []
    median = statistics.median
    for _ in range(_BOOTSTRAP_REPETITIONS):
        samples.append(median([values[rng.randrange(_BLOCKS)] for _ in range(_BLOCKS)]))
    samples.sort()
    return samples[int(0.025 * len(samples))], samples[min(len(samples) - 1, int(0.975 * len(samples)))]


def _certify(row: dict) -> dict:
    if row.get("status") == "ERROR" or row.get("exact_replay_passed") is not True:
        out = dict(row)
        out.update({"live_decision": "ERROR", "matches_frozen_decision": False})
        return out
    ratios = [float(value) for value in row["ratios"]]
    exact_times = [float(value) for value in row["exact_times_ms"]]
    fp_times = [float(value) for value in row["fp_times_ms"]]
    if len(ratios) != _BLOCKS or len(exact_times) != _BLOCKS or len(fp_times) != _BLOCKS:
        raise RuntimeError(f"{row.get('shape_id')} did not emit the frozen {_BLOCKS}-block protocol")
    center = statistics.median(ratios)
    wins = sum(value > 1.0 for value in ratios)
    low, high = _bootstrap_ci(ratios, int(row["bootstrap_seed"]))
    if center > _PROMOTION and wins >= _REQUIRED_WINS and low > 1.0:
        decision = "EXACT_WIN"
    elif center < 1.0 / _PROMOTION and wins <= _BLOCKS - _REQUIRED_WINS and high < 1.0:
        decision = "FLOATING_WIN"
    else:
        decision = "STATISTICAL_TIE"
    exact_median = statistics.median(exact_times)
    fp_median = statistics.median(fp_times)
    out = dict(row)
    out.update(
        {
            "live_decision": decision,
            "matches_frozen_decision": decision == row.get("frozen_decision"),
            "median_within_block_speedup": center,
            "bootstrap_low": low,
            "bootstrap_high": high,
            "exact_block_wins": wins,
            "paired_blocks": _BLOCKS,
            "exact_median_ms": exact_median,
            "fp16_median_ms": fp_median,
            "speedup_fp16_over_exact": fp_median / exact_median if exact_median > 0 else 0.0,
        }
    )
    return out


def _run_native(mode: str, expected: int, *, display: bool, stream: TextIO) -> tuple[list[dict], list[dict], float]:
    rows: list[dict] = []
    summaries: list[dict] = []
    started = time.monotonic()
    for family in _FAMILIES:
        family_rows, family_summaries = _stream_family(
            integer_binary(family),
            mode,
            expected,
            len(rows),
            started,
            display=display,
            stream=stream,
        )
        rows.extend(family_rows)
        summaries.extend(family_summaries)
    elapsed = time.monotonic() - started
    if display:
        _redraw(stream, _progress_line("GPU test", expected, expected, started, 0), 0)
        print(file=stream)
    if len(rows) != expected:
        raise RuntimeError(f"integer replay expected {expected} measurements, got {len(rows)}")
    errors = [row for row in rows if row.get("status") == "ERROR" or row.get("exact_replay_passed") is not True]
    if errors:
        raise RuntimeError(f"G4 integer replay failed exactness/runtime verification on {len(errors)} shape(s)")
    if sum(int(summary.get("runtime_errors", 0)) for summary in summaries) != 0:
        raise RuntimeError("G4 integer replay native summaries reported runtime errors")
    if sum(int(summary.get("exact_replay_failures", 0)) for summary in summaries) != 0:
        raise RuntimeError("G4 integer replay native summaries reported exactness failures")
    return rows, summaries, elapsed


def _verify_all(measurements: list[dict], *, display: bool, stream: TextIO) -> tuple[list[dict], float]:
    started = time.monotonic()
    previous_len = 0
    rows = []
    total = len(measurements)
    for index, row in enumerate(measurements, 1):
        rows.append(_certify(row))
        if display:
            previous_len = _redraw(stream, _progress_line("Verify", index, total, started, index), previous_len)
    elapsed = time.monotonic() - started
    if display:
        _redraw(stream, _progress_line("Verify", total, total, started, total + 1), previous_len)
        print(file=stream)
    if any(row["live_decision"] == "ERROR" for row in rows):
        raise RuntimeError("G4 integer replay failed closed during certification")
    return rows, elapsed


def _make_trust_pack(result: dict, *, started_utc: str, completed_utc: str, environment: str,
                     package_version: str, current_cuda: str, gpu_before: dict, gpu_after: dict) -> dict:
    meta = runtime_metadata()
    receipt = {
        "schema": "RNS-ENGINE-G4S1-INTEGER-TRUST-PACK-1",
        "series": 1,
        "campaign": "exact_integer_vs_fp16_input_cublaslt",
        "live_test": "G4_INTEGERS_vs_NVIDIA_FP16_cuBLASLt",
        "mode": result["mode"],
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "package": {"name": "rns_engine", "version": package_version, "release_tag": f"v{package_version}"},
        "environment": {
            "name": environment,
            "python": sys.version.split()[0],
            "current_cuda_reported_by_nvidia_smi": current_cuda,
        },
        "gpu": result["gpu"],
        "gpu_state_before": gpu_before,
        "gpu_state_after": gpu_after,
        "protocol": {
            "comparison": "G4 exact integer GEMM vs NVIDIA FP16-input cuBLASLt",
            "paired_timing_blocks_per_shape": _BLOCKS,
            "bootstrap_resamples_per_shape": _BOOTSTRAP_REPETITIONS,
            "bootstrap_confidence_interval": 0.95,
            "promotion_threshold": _PROMOTION,
            "required_exact_block_wins": _REQUIRED_WINS,
            "fail_closed_on_runtime_error": True,
            "fail_closed_on_exactness_failure": True,
        },
        "exactness": {
            "rows_exact": len(result["rows"]),
            "rows_total": len(result["rows"]),
            "runtime_errors": 0,
            "exact_replay_failures": 0,
        },
        "runtime": {
            "public_payload_sha256": meta["payload_sha256"],
            "public_tar_sha256": meta["tar_sha256"],
            "integer_replay_bundle_sha256": meta["integer_replay_bundle_sha256"],
            "source_archive_sha256": meta["source_archive_sha256"],
            "frozen_final_ledger_sha256": meta["frozen_final_ledger_sha256"],
            "build_nvcc": meta["build_nvcc"],
        },
        "summary": result["summary"],
        "xops": result["xops"],
        "timing_seconds": result["timing_seconds"],
        "result_rows_sha256": _sha256_json(result["rows"]),
    }
    receipt["run_receipt_sha256"] = _sha256_json(receipt)
    return receipt


def _print_trust_pack(trust: dict, stream: TextIO) -> None:
    gpu = trust["gpu"]; runtime = trust["runtime"]; env = trust["environment"]
    _box(
        "TRUST PACK - G4 INTEGER RUN RECEIPT",
        [
            "Live test: G4 INTEGERS vs NVIDIA FP16-input cuBLASLt",
            f"Environment: {env['name']} | rns_engine {trust['package']['version']} | release {trust['package']['release_tag']}",
            f"GPU: {gpu.get('name')} | compute capability {gpu.get('compute_capability')} | driver {gpu.get('driver_version')}",
            f"Replay build toolchain: {_build_cuda_summary(runtime['build_nvcc'])}",
            f"GPU state before: {_format_gpu_state(trust['gpu_state_before'])}",
            f"GPU state after:  {_format_gpu_state(trust['gpu_state_after'])}",
            "",
            f"Protocol: {_BLOCKS} paired timing blocks/shape | {_BOOTSTRAP_REPETITIONS:,} bootstrap resamples/shape | 95% CI",
            f"Integer exactness gate: {trust['exactness']['rows_exact']} / {trust['exactness']['rows_total']} exact",
            "",
            f"Integer replay bundle SHA-256: {runtime['integer_replay_bundle_sha256']}",
            f"Public runtime payload SHA-256: {runtime['public_payload_sha256']}",
            f"Source archive SHA-256:          {runtime['source_archive_sha256']}",
            f"All integer result rows SHA-256:{trust['result_rows_sha256']}",
            f"Run receipt SHA-256:             {trust['run_receipt_sha256']}",
            "",
            f"Started UTC: {trust['started_utc']} | Completed UTC: {trust['completed_utc']}",
        ],
        stream,
    )


def _g4_integer_benchmark(
    mode: str = "quick",
    *,
    display: bool = True,
    stream: TextIO | None = None,
    show_xops_key: bool = True,
) -> dict:
    try:
        selected = mode.strip().lower()
    except AttributeError as exc:
        raise ValueError("mode must be quick, standard, or full") from exc
    if selected not in _ALLOWED_MODES:
        raise ValueError(f"unknown G4 integer benchmark mode {mode!r}; choose quick, standard, or full")

    out = stream if stream is not None else sys.stdout
    meta = runtime_metadata()
    expected = int(meta["modes"][selected])
    started_utc = _utc_now()
    environment = _environment_name()
    package_version = _package_version()
    current_cuda = _current_cuda_version()
    gpu_before = _gpu_state_snapshot()
    gpu = ensure_t4()

    if display:
        if show_xops_key:
            print_xops_key(out)
            print(file=out)
        _box(
            "G4 INTEGERS - PUBLIC TESLA T4 REPRODUCTION",
            [
                f"GPU: {gpu['name']} | compute capability {gpu['compute_capability']} | driver {gpu['driver_version']}",
                "Exact integer matrix multiplication directly against NVIDIA FP16-input cuBLASLt",
                f"This run: {expected} matrix-multiplication shapes ({selected} mode)",
                "Frozen Series 1 integer result: 938 / 1024 faster than NVIDIA (91.60%)",
            ],
            out,
        )
        print(file=out)
        print("Running G4 INTEGERS. Progress updates in place.", file=out, flush=True)

    measurements, native_summaries, gpu_seconds = _run_native(selected, expected, display=display, stream=out)
    rows, verify_seconds = _verify_all(measurements, display=display, stream=out)

    live_exact = sum(row["live_decision"] == "EXACT_WIN" for row in rows)
    live_fp = sum(row["live_decision"] == "FLOATING_WIN" for row in rows)
    live_ties = sum(row["live_decision"] == "STATISTICAL_TIE" for row in rows)
    matches = sum(bool(row["matches_frozen_decision"]) for row in rows)
    win_speedups = [row["speedup_fp16_over_exact"] for row in rows if row["live_decision"] == "EXACT_WIN"]
    summary = {
        "shapes_run": expected,
        "runtime_errors": 0,
        "exact_replay_failures": 0,
        "live_exact_wins": live_exact,
        "live_floating_wins": live_fp,
        "live_statistical_ties": live_ties,
        "integer_frozen_decision_matches": matches,
        "median_speedup_among_exact_wins": statistics.median(win_speedups) if win_speedups else 0.0,
    }
    xops = build_xops_summary(
        rows,
        species="G4_INTEGERS",
        headline_time_key="exact_median_ms",
        timing_boundary="exact integer GEMM median execution time; exactness verification is outside timing",
    )
    result = {
        "schema": "RNS-ENGINE-G4S1-INTEGER-LIVE-BENCHMARK-1",
        "series": 1,
        "campaign": "exact_integer_vs_fp16_input_cublaslt",
        "live_test": "G4_INTEGERS_vs_NVIDIA_FP16_cuBLASLt",
        "mode": selected,
        "gpu": gpu,
        "runtime": {
            "public_payload_sha256": meta["payload_sha256"],
            "integer_replay_bundle_sha256": meta["integer_replay_bundle_sha256"],
            "source_archive_sha256": meta["source_archive_sha256"],
            "build_nvcc": meta["build_nvcc"],
        },
        "timing_seconds": {"gpu_test": gpu_seconds, "result_verification": verify_seconds, "total": gpu_seconds + verify_seconds},
        "summary": summary,
        "xops": xops,
        "native_summaries": native_summaries,
        "rows": rows,
    }
    gpu_after = _gpu_state_snapshot()
    completed_utc = _utc_now()
    trust = _make_trust_pack(
        result,
        started_utc=started_utc,
        completed_utc=completed_utc,
        environment=environment,
        package_version=package_version,
        current_cuda=current_cuda,
        gpu_before=gpu_before,
        gpu_after=gpu_after,
    )
    result["trust_pack"] = trust

    if display:
        print(file=out)
        _box(
            "G4 INTEGERS - EXACTNESS - FRESH RUN",
            [f"G4 integer calculations correct: {expected} / {expected}", "Result: PASS"],
            out,
        )
        print(file=out)
        _box(
            "G4 INTEGERS - SPEED vs NVIDIA FP16 - FRESH RUN",
            [
                f"G4 INTEGERS faster than NVIDIA FP16: {live_exact} / {expected} ({live_exact / expected * 100:.2f}%)",
                f"NVIDIA FP16 faster than G4 INTEGERS: {live_fp} / {expected} ({live_fp / expected * 100:.2f}%)",
                f"G4 INTEGERS / NVIDIA FP16 statistical ties: {live_ties} / {expected} ({live_ties / expected * 100:.2f}%)",
                f"Median speedup when G4 INTEGERS win: {summary['median_speedup_among_exact_wins']:.3f}x",
            ],
            out,
        )
        print(file=out)
        print_xops_summary("XOPS / G4OPS - G4 INTEGERS", xops, out)
        print(file=out)
        _box(
            "G4 INTEGERS - REPRODUCIBILITY",
            [
                f"INTEGER winner/tie classification matches original INTEGER benchmark: {matches} / {expected} ({matches / expected * 100:.2f}%)",
                f"Integer replay bundle SHA-256: {meta['integer_replay_bundle_sha256']}",
                f"Total elapsed: {_fmt_time(gpu_seconds + verify_seconds)}",
            ],
            out,
        )
        print(file=out)
        _print_trust_pack(trust, out)

    return result


def g4_integer_benchmark(mode: str = "quick", *, display: bool = True, stream: TextIO | None = None) -> dict:
    """Rerun the frozen G4 Series 1 exact-integer benchmark on a Tesla T4."""
    return _g4_integer_benchmark(mode, display=display, stream=stream, show_xops_key=True)


__all__ = ["g4_integer_benchmark"]
