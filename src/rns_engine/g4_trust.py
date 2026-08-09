"""Human-readable, cryptographically committed trust receipt for G4 replay runs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as distribution_version
from typing import TextIO

from .g4_benchmark import (
    _BLOCKS,
    _BOOTSTRAP_REPETITIONS,
    _PROMOTION,
    _REQUIRED_WINS,
    _box,
    _runtime_metadata,
    g4_benchmark as _core_g4_benchmark,
)

_TRUST_SCHEMA = "RNS-ENGINE-G4S1-TRUST-PACK-1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _package_version() -> str:
    try:
        return distribution_version("rns_engine")
    except PackageNotFoundError:
        return "0+unknown"


def _environment_name() -> str:
    if (
        "google.colab" in sys.modules
        or os.environ.get("COLAB_RELEASE_TAG")
        or os.environ.get("COLAB_BACKEND_VERSION")
        or os.environ.get("COLAB_GPU")
    ):
        return "Google Colab"
    return platform.platform()


def _current_cuda_version() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unavailable"
    if proc.returncode != 0:
        return "unavailable"
    match = re.search(r"CUDA Version:\s*([0-9.]+)", proc.stdout)
    return match.group(1) if match else "unavailable"


def _gpu_state_snapshot() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=pstate,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}
    if proc.returncode != 0:
        return {
            "available": False,
            "error": proc.stderr.strip() or f"nvidia-smi exit {proc.returncode}",
        }
    rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        return {"available": False, "error": f"expected one GPU row, got {len(rows)}"}
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 5:
        return {"available": False, "error": f"unexpected GPU state row: {rows[0]}"}
    return {
        "available": True,
        "pstate": parts[0],
        "temperature_c": parts[1],
        "power_w": parts[2],
        "sm_clock_mhz": parts[3],
        "memory_clock_mhz": parts[4],
    }


def _format_gpu_state(snapshot: dict) -> str:
    if snapshot.get("available") is not True:
        return "unavailable"
    return (
        f"{snapshot['pstate']} | {snapshot['temperature_c']} C | {snapshot['power_w']} W | "
        f"SM {snapshot['sm_clock_mhz']} MHz | MEM {snapshot['memory_clock_mhz']} MHz"
    )


def _build_cuda_summary(build_nvcc: str) -> str:
    match = re.search(r"release\s+([0-9.]+),\s+V([0-9.]+)", build_nvcc)
    if match:
        return f"CUDA {match.group(1)} / nvcc V{match.group(2)}"
    for line in reversed(build_nvcc.splitlines()):
        if line.strip():
            return line.strip()
    return "unavailable"


def _make_trust_pack(
    result: dict,
    *,
    started_utc: str,
    completed_utc: str,
    environment: str,
    package_version: str,
    current_cuda: str,
    gpu_before: dict,
    gpu_after: dict,
) -> dict:
    meta = _runtime_metadata()
    rows = result["rows"]
    rows_sha256 = _sha256_json(rows)
    exact_rows = sum(bool(row.get("exact_replay_passed")) for row in rows)

    receipt = {
        "schema": _TRUST_SCHEMA,
        "series": result.get("series"),
        "campaign": result.get("campaign"),
        "mode": result.get("mode"),
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "package": {
            "name": "rns_engine",
            "version": package_version,
            "release_tag": f"v{package_version}",
        },
        "environment": {
            "name": environment,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "current_cuda_reported_by_nvidia_smi": current_cuda,
        },
        "gpu": result.get("gpu", {}),
        "gpu_state_before": gpu_before,
        "gpu_state_after": gpu_after,
        "protocol": {
            "comparison": "exact rational GEMM vs NVIDIA FP16 cuBLASLt",
            "paired_timing_blocks_per_shape": _BLOCKS,
            "bootstrap_resamples_per_shape": _BOOTSTRAP_REPETITIONS,
            "bootstrap_confidence_interval": 0.95,
            "promotion_threshold": _PROMOTION,
            "required_exact_block_wins": _REQUIRED_WINS,
            "fail_closed_on_runtime_error": True,
            "fail_closed_on_exactness_failure": True,
        },
        "exactness": {
            "rows_exact": exact_rows,
            "rows_total": len(rows),
            "runtime_errors": int(result.get("summary", {}).get("runtime_errors", 0)),
            "exact_replay_failures": int(
                result.get("summary", {}).get("exact_replay_failures", 0)
            ),
        },
        "runtime": {
            "binary_sha256": result["runtime"]["binary_sha256"],
            "payload_sha256": result["runtime"]["payload_sha256"],
            "source_archive_sha256": result["runtime"]["source_archive_sha256"],
            "public_runtime_source_sha256": meta.get("public_runtime_source_sha256"),
            "build_nvcc": result["runtime"]["build_nvcc"],
        },
        "timing_seconds": result.get("timing_seconds", {}),
        "summary": result.get("summary", {}),
        "result_rows_sha256": rows_sha256,
    }
    receipt_sha256 = _sha256_json(receipt)
    return {**receipt, "run_receipt_sha256": receipt_sha256}


def _print_trust_pack(trust: dict, stream: TextIO) -> None:
    gpu = trust["gpu"]
    package = trust["package"]
    exactness = trust["exactness"]
    runtime = trust["runtime"]
    protocol = trust["protocol"]
    environment = trust["environment"]

    _box(
        "TRUST PACK - CRYPTOGRAPHIC RUN RECEIPT",
        [
            f"Environment: {environment['name']} | rns_engine {package['version']} | release {package['release_tag']}",
            f"Python: {environment['python']} | CUDA reported by NVIDIA: {environment['current_cuda_reported_by_nvidia_smi']}",
            f"GPU: {gpu.get('name', 'unknown')} | compute capability {gpu.get('compute_capability', 'unknown')} | driver {gpu.get('driver_version', 'unknown')}",
            f"Replay build toolchain: {_build_cuda_summary(runtime['build_nvcc'])}",
            f"GPU state before: {_format_gpu_state(trust['gpu_state_before'])}",
            f"GPU state after:  {_format_gpu_state(trust['gpu_state_after'])}",
            "",
            f"Protocol: {protocol['paired_timing_blocks_per_shape']} paired timing blocks/shape | {protocol['bootstrap_resamples_per_shape']:,} bootstrap resamples/shape | 95% CI",
            f"Exactness gate: {exactness['rows_exact']} / {exactness['rows_total']} exact | runtime errors {exactness['runtime_errors']} | exactness failures {exactness['exact_replay_failures']}",
            "",
            f"Replay binary SHA-256:   {runtime['binary_sha256']}",
            f"Replay payload SHA-256:  {runtime['payload_sha256']}",
            f"Source archive SHA-256:   {runtime['source_archive_sha256']}",
            f"All result rows SHA-256:  {trust['result_rows_sha256']}",
            f"Run receipt SHA-256:      {trust['run_receipt_sha256']}",
            "",
            f"Started UTC: {trust['started_utc']} | Completed UTC: {trust['completed_utc']}",
        ],
        stream,
        width=112,
    )


def g4_benchmark(
    mode: str = "quick",
    *,
    display: bool = True,
    stream: TextIO | None = None,
) -> dict:
    """Run the public G4 benchmark and append a cryptographic trust receipt."""
    out = stream if stream is not None else sys.stdout
    started_utc = _utc_now()
    environment = _environment_name()
    package_version = _package_version()
    current_cuda = _current_cuda_version()
    gpu_before = _gpu_state_snapshot()

    result = _core_g4_benchmark(mode, display=display, stream=out)

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
        _print_trust_pack(trust, out)

    return result


__all__ = ["g4_benchmark"]
