from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .capacity import plan_digit_plane_gemm_capacity
from .cuda_contract import (
    CpuExactPipelineBackend,
    CudaGemmFixture,
    build_cuda_gemm_fixture,
    verify_backend,
)
from .exact_gemm import SharedScaleMatrix, exact_shared_scale_gemm
from .rail_planning import search_mersenne_rails_for_capacity
from .wide import WideRNSConfig


@dataclass(frozen=True, slots=True)
class PreCudaReadinessReport:
    ready: bool
    checks: dict[str, bool]
    target_required_modulus_bits: int
    target_additional_bits: int
    minimum_extra_rails_under_31_bits: int
    profiles: dict[str, dict[str, Any]]
    fixture_name: str

    def require_ready(self) -> PreCudaReadinessReport:
        if not self.ready:
            failed = [name for name, passed in self.checks.items() if not passed]
            raise AssertionError(f"pre-CUDA readiness failed: {failed}")
        return self

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(self.to_json() + "\n", encoding="utf-8")
        return destination


def _target_capacity():
    return plan_digit_plane_gemm_capacity(
        inner_dimension=5120,
        radix=128,
        left_digit_abs_bounds=[127] * 8,
        right_digit_abs_bounds=[127] * 8,
    ).capacity


def _profile_summary(config: WideRNSConfig, target: Any) -> dict[str, Any]:
    return {
        "moduli": list(config.moduli),
        "rail_count": config.rail_count,
        "product_bits": config.product_bits,
        "closes_target": config.product >= target.minimum_required_modulus,
    }


def default_pre_cuda_matrices() -> tuple[SharedScaleMatrix, SharedScaleMatrix]:
    left = SharedScaleMatrix(
        np.asarray(
            [
                [2**42 + 17, -(2**35) + 5, 123456789],
                [-999999999, 2**40 - 3, 77],
            ],
            dtype=object,
        ),
        scale=6,
    )
    right = SharedScaleMatrix(
        np.asarray(
            [
                [17, -(2**31) + 1],
                [2**33 + 9, 41],
                [-12345, 2**29 - 7],
            ],
            dtype=object,
        ),
        scale=35,
    )
    return left, right


def build_default_pre_cuda_fixture(
    *,
    config: WideRNSConfig | None = None,
) -> CudaGemmFixture:
    left, right = default_pre_cuda_matrices()
    return build_cuda_gemm_fixture(
        "pre-cuda-eight-plane-witness",
        left,
        right,
        config=config or WideRNSConfig.balanced_seven_rail(),
        radix=128,
        left_plane_count=8,
        right_plane_count=8,
    )


def run_pre_cuda_readiness(*, seed: int = 20260802) -> PreCudaReadinessReport:
    """Run every deterministic CPU-side gate required before renting a T4."""

    target = _target_capacity()
    search = search_mersenne_rails_for_capacity(
        target,
        max_exponent=31,
        max_rails=4,
    )
    smallest = WideRNSConfig.smallest_product_seven_rail()
    balanced = WideRNSConfig.balanced_seven_rail()

    checks: dict[str, bool] = {
        "target_requires_126_bits": target.required_modulus_bits == 126,
        "target_deficit_is_71_bits": target.additional_bits_required == 71,
        "three_extra_rails_are_minimum": search.minimum_rail_count == 3,
        "smallest_profile_closes_target": smallest.product >= target.minimum_required_modulus,
        "balanced_profile_closes_target": balanced.product >= target.minimum_required_modulus,
    }

    rng = np.random.default_rng(seed)
    for config in (smallest, balanced):
        values = np.asarray(
            [
                0,
                1,
                -1,
                config.signed_max,
                config.signed_min,
                int(rng.integers(-(1 << 50), 1 << 50)),
            ],
            dtype=object,
        )
        round_trip = config.decode(config.encode(values), signed=True)
        checks[f"{config.name}:signed_round_trip"] = np.array_equal(values, round_trip)

        left_values = np.asarray([123456789012345, -777777777777], dtype=object)
        right_values = np.asarray([-333333333333, 222222222222], dtype=object)
        added = config.decode(
            config.add(config.encode(left_values), config.encode(right_values)),
            signed=True,
        )
        multiplied = config.decode(
            config.mul(config.encode(left_values), config.encode(right_values)),
            signed=True,
        )
        expected_add = left_values + right_values
        expected_mul = left_values * right_values
        checks[f"{config.name}:rail_add"] = np.array_equal(added, expected_add)
        checks[f"{config.name}:rail_mul"] = np.array_equal(multiplied, expected_mul)

    left, right = default_pre_cuda_matrices()
    receipt = exact_shared_scale_gemm(
        left,
        right,
        config=balanced,
        radix=128,
        left_plane_count=8,
        right_plane_count=8,
        require_unique=True,
    )
    checks["shared_scale_exact_match"] = receipt.exact_match
    checks["shared_scale_output_scale"] = receipt.output_scale == 210
    checks["grouped_coefficients_fit_int32"] = receipt.local_capacity.safe
    checks["wide_result_is_unique"] = receipt.unique

    fixture = build_default_pre_cuda_fixture(config=balanced)
    fixture_round_trip = CudaGemmFixture.from_json(fixture.to_json())
    checks["fixture_json_round_trip"] = fixture_round_trip == fixture
    verification = verify_backend(fixture_round_trip, CpuExactPipelineBackend())
    checks["backend_contract_reference_passes"] = verification.passed

    profiles = {
        smallest.name: _profile_summary(smallest, target),
        balanced.name: _profile_summary(balanced, target),
    }
    report = PreCudaReadinessReport(
        ready=all(checks.values()),
        checks=checks,
        target_required_modulus_bits=target.required_modulus_bits,
        target_additional_bits=target.additional_bits_required,
        minimum_extra_rails_under_31_bits=search.minimum_rail_count or 0,
        profiles=profiles,
        fixture_name=fixture.name,
    )
    return report

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional JSON path for the readiness receipt",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="optional JSON path for the frozen CUDA fixture",
    )
    args = parser.parse_args(argv)

    report = run_pre_cuda_readiness()
    report.require_ready()
    print(report.to_json())
    if args.report is not None:
        report.write_json(args.report)
    if args.fixture is not None:
        build_default_pre_cuda_fixture().write_json(args.fixture)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
