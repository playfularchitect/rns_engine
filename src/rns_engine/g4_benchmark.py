"""Public G4 Series 1 Tesla T4 reproduction benchmark.

The benchmark replays the frozen Series 1 exact-rational implementations against
NVIDIA's FP16 cuBLASLt baseline on a Tesla T4. The packaged replay payload
decodes byte-for-byte to the stripped binary that passed the 1,024-shape public
validation run. This module only handles integrity checks, display, timing
collection, and the frozen statistical certification rule.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import queue
import random
from importlib import resources
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import TextIO

from .g4_results import g4_results

_RUNTIME_META = "g4s1_t4_runtime.json"
_ALLOWED_MODES = {"quick", "standard", "full"}
_PROMOTION = 1.002
_REQUIRED_WINS = 20
_BOOTSTRAP_REPETITIONS = 20_000
_BLOCKS = 31
_BAR_WIDTH = 30
_SPINNER = "|/-\\"


def _data_file(name: str):
    return resources.files(__package__).joinpath("data", name)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _runtime_metadata() -> dict:
    try:
        with _data_file(_RUNTIME_META).open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "G4 Series 1 T4 runtime is not installed. Install a release that includes "
            "the validated T4 replay runtime."
        ) from exc
    if meta.get("schema") != "RNS-ENGINE-G4S1-T4-RUNTIME-3":
        raise RuntimeError("unrecognized G4 Series 1 T4 runtime metadata schema")
    if meta.get("privacy_scan_passed") is not True:
        raise RuntimeError("G4 Series 1 T4 runtime metadata failed its release gate")
    if meta.get("payload_format") != "base64+gzip":
        raise RuntimeError("unsupported G4 Series 1 replay payload format")
    return meta


def _load_runtime_bytes(meta: dict) -> bytes:
    payload_name = meta["payload_name"]
    try:
        payload_text = _data_file(payload_name).read_text(encoding="ascii")
    except FileNotFoundError as exc:
        raise RuntimeError(f"G4 replay payload is missing: {payload_name}") from exc
    payload_bytes = payload_text.encode("ascii")
    actual_payload_sha = _sha256_bytes(payload_bytes)
    expected_payload_sha = meta["payload_sha256"]
    if actual_payload_sha != expected_payload_sha:
        raise RuntimeError(
            "G4 replay payload integrity check failed: "
            f"expected {expected_payload_sha}, got {actual_payload_sha}"
        )
    try:
        compressed = base64.b64decode(payload_text, validate=False)
        binary = gzip.decompress(compressed)
    except Exception as exc:
        raise RuntimeError("G4 replay payload could not be decoded") from exc
    actual_binary_sha = _sha256_bytes(binary)
    expected_binary_sha = meta["binary_sha256"]
    if actual_binary_sha != expected_binary_sha:
        raise RuntimeError(
            "G4 replay binary integrity check failed after decoding: "
            f"expected {expected_binary_sha}, got {actual_binary_sha}"
        )
    return binary


def _detect_t4() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(command, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "g4_benchmark() requires an NVIDIA Tesla T4. In Colab, select a T4 GPU and run again."
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            "g4_benchmark() requires an NVIDIA Tesla T4; nvidia-smi failed: "
            + (proc.stderr.strip() or f"exit {proc.returncode}")
        )
    rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            "g4_benchmark() currently requires exactly one visible Tesla T4 GPU. "
            f"nvidia-smi returned {len(rows)} GPU rows."
        )
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) < 3:
        raise RuntimeError(f"could not parse nvidia-smi output: {rows[0]!r}")
    name, compute_cap, driver = parts[:3]
    if "T4" not in name or compute_cap != "7.5":
        raise RuntimeError(
            "G4 Series 1 is frozen to Tesla T4 / compute capability 7.5. "
            f"Detected {name!r} / cc {compute_cap}."
        )
    return {"name": name, "compute_capability": compute_cap, "driver_version": driver}


def _bootstrap_median_ci(values: list[float], seed: int) -> tuple[float, float]:
    if len(values) != _BLOCKS:
        raise RuntimeError(f"expected {_BLOCKS} paired ratios, got {len(values)}")
    rng = random.Random(seed)
    count = len(values)
    samples = []
    append = samples.append
    median = statistics.median
    for _ in range(_BOOTSTRAP_REPETITIONS):
        append(median([values[rng.randrange(count)] for _ in range(count)]))
    samples.sort()
    return (
        samples[int(0.025 * len(samples))],
        samples[min(len(samples) - 1, int(0.975 * len(samples)))],
    )


def _certify_measurement(row: dict) -> dict:
    if row.get("status") == "ERROR":
        out = dict(row)
        out.update({"live_decision": "ERROR", "matches_frozen_decision": False})
        return out
    if row.get("exact_replay_passed") is not True:
        out = dict(row)
        out.update({"live_decision": "EXACT_REPLAY_FAILED", "matches_frozen_decision": False})
        return out

    ratios = [float(value) for value in row.get("ratios", [])]
    exact_times = [float(value) for value in row.get("exact_times_ms", [])]
    fp_times = [float(value) for value in row.get("fp_times_ms", [])]
    if len(ratios) != _BLOCKS or len(exact_times) != _BLOCKS or len(fp_times) != _BLOCKS:
        raise RuntimeError(
            f"{row.get('shape_id', '<unknown>')} did not emit the frozen {_BLOCKS}-block protocol"
        )

    center = statistics.median(ratios)
    exact_median = statistics.median(exact_times)
    fp_median = statistics.median(fp_times)
    wins = sum(value > 1.0 for value in ratios)
    low, high = _bootstrap_median_ci(ratios, int(row["bootstrap_seed"]))

    if center > _PROMOTION and wins >= _REQUIRED_WINS and low > 1.0:
        decision = "EXACT_WIN"
    elif center < 1.0 / _PROMOTION and wins <= len(ratios) - _REQUIRED_WINS and high < 1.0:
        decision = "FLOATING_WIN"
    else:
        decision = "STATISTICAL_TIE"

    metadata_ns = float(row["metadata_ns"])
    end_to_end_exact_ms = exact_median + metadata_ns / 1_000_000.0
    end_to_end_speedup = fp_median / end_to_end_exact_ms if end_to_end_exact_ms > 0 else 0.0
    if decision == "EXACT_WIN" and end_to_end_speedup <= _PROMOTION:
        decision = "STATISTICAL_TIE"

    out = dict(row)
    out.update(
        {
            "exact_median_ms": exact_median,
            "fp16_median_ms": fp_median,
            "speedup_fp16_over_exact": fp_median / exact_median if exact_median > 0 else 0.0,
            "median_within_block_speedup": center,
            "bootstrap_low": low,
            "bootstrap_high": high,
            "exact_block_wins": wins,
            "paired_blocks": len(ratios),
            "fraction_end_to_end_median_ms": end_to_end_exact_ms,
            "fraction_end_to_end_speedup": end_to_end_speedup,
            "live_decision": decision,
            "matches_frozen_decision": decision == row.get("frozen_decision"),
        }
    )
    return out


def _fmt_time(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--:--"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _progress_line(label: str, done: int, total: int, started: float, spin: int) -> str:
    elapsed = max(0.001, time.monotonic() - started)
    fraction = min(1.0, done / max(1, total))
    filled = int(_BAR_WIDTH * fraction)
    bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
    eta = None if done <= 0 else elapsed * (total - done) / done
    return (
        f"{_SPINNER[spin % len(_SPINNER)]} {label:<9} "
        f"[{bar}] {done:4d}/{total} {fraction * 100:5.1f}%  "
        f"elapsed {_fmt_time(elapsed)}  ETA {_fmt_time(eta)}"
    )


def _redraw(stream: TextIO, line: str, previous_len: int) -> int:
    padded = line.ljust(previous_len)
    print("\r" + padded, end="", flush=True, file=stream)
    return max(previous_len, len(line))


def _box(title: str, lines: list[str], stream: TextIO, width: int = 92) -> None:
    inner = width - 4
    border = "+" + "-" * (width - 2) + "+"
    print(border, file=stream)
    print("| " + title[:inner].ljust(inner) + " |", file=stream)
    print(border, file=stream)
    for line in lines:
        print("| " + line[:inner].ljust(inner) + " |", file=stream)
    print(border, file=stream)


def _reader_thread(stream, output_queue: queue.Queue) -> None:
    try:
        for line in stream:
            output_queue.put(line.rstrip("\n"))
    finally:
        output_queue.put(None)


def _run_native(
    binary: bytes,
    mode: str,
    expected_shapes: int,
    *,
    display: bool,
    stream: TextIO,
) -> tuple[list[dict], dict, float]:
    measurements: list[dict] = []
    native_summary: dict | None = None
    unexpected: list[str] = []

    with tempfile.TemporaryDirectory(prefix="rns-g4s1-") as td:
        executable = Path(td) / "g4s1_t4_replay"
        executable.write_bytes(binary)
        executable.chmod(0o700)
        proc = subprocess.Popen(
            [str(executable), "--mode", mode],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        output_queue: queue.Queue = queue.Queue()
        thread = threading.Thread(
            target=_reader_thread,
            args=(proc.stdout, output_queue),
            daemon=True,
        )
        thread.start()

        started = time.monotonic()
        previous_len = 0
        spin = 0
        stream_done = False
        while not stream_done:
            try:
                item = output_queue.get(timeout=0.35)
            except queue.Empty:
                item = "__TICK__"

            if item is None:
                stream_done = True
            elif item == "__TICK__":
                pass
            elif item.startswith("RNS_G4S1_MEASUREMENT "):
                measurements.append(json.loads(item[len("RNS_G4S1_MEASUREMENT "):]))
            elif item.startswith("RNS_G4S1_SUMMARY "):
                native_summary = json.loads(item[len("RNS_G4S1_SUMMARY "):])
            elif item:
                unexpected.append(item)

            if display:
                previous_len = _redraw(
                    stream,
                    _progress_line("GPU test", len(measurements), expected_shapes, started, spin),
                    previous_len,
                )
            spin += 1

        returncode = proc.wait()
        elapsed = time.monotonic() - started
        if display:
            _redraw(
                stream,
                _progress_line("GPU test", len(measurements), expected_shapes, started, spin),
                previous_len,
            )
            print(file=stream)

    if returncode != 0:
        detail = "\n".join(unexpected[-12:]) if unexpected else "no additional error text"
        raise RuntimeError(f"G4 T4 replay failed with exit {returncode}.\n{detail}")
    if native_summary is None:
        raise RuntimeError("G4 T4 replay runtime did not emit a summary")
    if len(measurements) != expected_shapes or int(native_summary.get("shapes_run", -1)) != expected_shapes:
        raise RuntimeError(
            f"G4 T4 replay shape-count mismatch: expected {expected_shapes}, "
            f"native={native_summary.get('shapes_run')}, rows={len(measurements)}"
        )
    if int(native_summary.get("runtime_errors", 0)) != 0:
        raise RuntimeError(f"G4 T4 replay reported runtime errors: {native_summary}")
    if int(native_summary.get("exact_replay_failures", 0)) != 0:
        raise RuntimeError(f"G4 T4 replay failed exactness verification: {native_summary}")
    return measurements, native_summary, elapsed


def _certify_all(
    measurements: list[dict],
    *,
    display: bool,
    stream: TextIO,
) -> tuple[list[dict], float]:
    started = time.monotonic()
    previous_len = 0
    rows: list[dict] = []
    total = len(measurements)
    for index, measurement in enumerate(measurements, start=1):
        rows.append(_certify_measurement(measurement))
        if display:
            previous_len = _redraw(
                stream,
                _progress_line("Verify", index, total, started, index),
                previous_len,
            )
    elapsed = time.monotonic() - started
    if display:
        _redraw(
            stream,
            _progress_line("Verify", total, total, started, total + 1),
            previous_len,
        )
        print(file=stream)
    return rows, elapsed


def g4_benchmark(
    mode: str = "quick",
    *,
    display: bool = True,
    stream: TextIO | None = None,
) -> dict:
    """Rerun the frozen G4 Series 1 exact-rational benchmark on a Tesla T4.

    ``quick`` runs 24 shapes, ``standard`` runs 128, and ``full`` runs all
    1,024 shapes. The returned dictionary contains every per-shape measurement;
    the default display stays compact and updates two progress/ETA lines in place.
    """
    try:
        selected = mode.strip().lower()
    except AttributeError as exc:
        raise ValueError("mode must be quick, standard, or full") from exc
    if selected not in _ALLOWED_MODES:
        raise ValueError(f"unknown G4 benchmark mode {mode!r}; choose quick, standard, or full")

    out = stream if stream is not None else sys.stdout
    meta = _runtime_metadata()
    expected_shapes = int(meta["modes"][selected])
    gpu = _detect_t4()
    binary = _load_runtime_bytes(meta)
    frozen = g4_results(display=False)
    integer_claim = frozen["claims"]["integer_fp16_input_clean_sweep"]
    rational_claim = frozen["claims"]["dynamic_exact_rational"]

    if display:
        print(file=out)
        _box(
            "G4 SERIES 1 - PUBLIC TESLA T4 REPRODUCTION",
            [
                f"GPU: {gpu['name']} | compute capability {gpu['compute_capability']} | driver {gpu['driver_version']}",
                "Exact rational matrix multiplication directly against NVIDIA FP16 cuBLASLt",
                f"This run: {expected_shapes} matrix-multiplication shapes ({selected} mode)",
            ],
            out,
        )
        print(file=out)
        _box(
            "ORIGINAL FROZEN RESULTS - TWO SEPARATE TESTS",
            [
                f"Exact integers faster than NVIDIA FP16:  {integer_claim['numerator']} / {integer_claim['denominator']}  ({integer_claim['rate'] * 100:.2f}%)",
                f"Exact rationals faster than NVIDIA FP16: {rational_claim['numerator']} / {rational_claim['denominator']}  ({rational_claim['rate'] * 100:.2f}%)",
            ],
            out,
        )
        print(file=out)
        print("Running the benchmark. The lines below update in place.", file=out)

    measurements, native_summary, gpu_seconds = _run_native(
        binary,
        selected,
        expected_shapes,
        display=display,
        stream=out,
    )
    rows, verify_seconds = _certify_all(measurements, display=display, stream=out)

    live_exact = sum(row["live_decision"] == "EXACT_WIN" for row in rows)
    live_fp = sum(row["live_decision"] == "FLOATING_WIN" for row in rows)
    live_ties = sum(row["live_decision"] == "STATISTICAL_TIE" for row in rows)
    matches = sum(bool(row["matches_frozen_decision"]) for row in rows)
    win_speedups = [
        row["speedup_fp16_over_exact"] for row in rows if row["live_decision"] == "EXACT_WIN"
    ]
    end_to_end_speedups = [
        row["fraction_end_to_end_speedup"] for row in rows if row["live_decision"] == "EXACT_WIN"
    ]
    median_speedup = statistics.median(win_speedups) if win_speedups else 0.0
    median_e2e_speedup = statistics.median(end_to_end_speedups) if end_to_end_speedups else 0.0

    summary = dict(native_summary)
    summary.update(
        {
            "live_exact_wins": live_exact,
            "live_floating_wins": live_fp,
            "live_statistical_ties": live_ties,
            "frozen_decision_matches": matches,
            "median_speedup_among_exact_wins": median_speedup,
            "median_end_to_end_speedup_among_exact_wins": median_e2e_speedup,
        }
    )

    result = {
        "schema": "RNS-ENGINE-G4S1-LIVE-BENCHMARK-2",
        "series": 1,
        "campaign": "dynamic_exact_rational_vs_fp16",
        "mode": selected,
        "gpu": gpu,
        "runtime": {
            "binary_sha256": meta["binary_sha256"],
            "payload_sha256": meta["payload_sha256"],
            "source_archive_sha256": meta["source_archive_sha256"],
            "build_nvcc": meta["build_nvcc"],
            "config_provenance": meta.get("config_provenance", {}),
        },
        "timing_seconds": {
            "gpu_test": gpu_seconds,
            "result_verification": verify_seconds,
            "total": gpu_seconds + verify_seconds,
        },
        "summary": summary,
        "rows": rows,
    }

    if display:
        print(file=out)
        _box(
            "EXACTNESS - FRESH RUN",
            [
                f"Exact rational calculations correct: {expected_shapes} / {expected_shapes}",
                "Result: PASS",
            ],
            out,
        )
        print(file=out)
        _box(
            "SPEED - FRESH RUN DIRECTLY AGAINST NVIDIA FP16",
            [
                f"Exact rational faster: {live_exact} / {expected_shapes}  ({live_exact / expected_shapes * 100:.2f}%)",
                f"NVIDIA FP16 faster:    {live_fp} / {expected_shapes}  ({live_fp / expected_shapes * 100:.2f}%)",
                f"Too close to call:     {live_ties} / {expected_shapes}  ({live_ties / expected_shapes * 100:.2f}%)",
                f"Median speedup when exact wins: {median_speedup:.3f}x",
                f"Median speedup including rational bookkeeping: {median_e2e_speedup:.3f}x",
            ],
            out,
        )
        print(file=out)
        _box(
            "REPRODUCIBILITY",
            [
                f"Same winner/tie result as original: {matches} / {expected_shapes}  ({matches / expected_shapes * 100:.2f}%)",
                f"Validated replay binary SHA-256: {meta['binary_sha256']}",
                f"Total elapsed: {_fmt_time(gpu_seconds + verify_seconds)}",
            ],
            out,
        )

    return result


__all__ = ["g4_benchmark"]
