from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .exact_gemm import (
    ExactSharedScaleGemmReceipt,
    SharedScaleMatrix,
    exact_shared_scale_gemm,
    grouped_plane_gemm,
)
from .wide import WideRNSConfig


SCHEMA_VERSION = 1


def _array_to_lists(array: np.ndarray) -> Any:
    return np.asarray(array, dtype=object).tolist()


def _int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


@dataclass(frozen=True, slots=True)
class CudaGemmFixture:
    """Portable exact fixture consumed by the future CUDA backend."""

    name: str
    radix: int
    moduli: tuple[int, ...]
    left_scale: int
    right_scale: int
    left_numerators: Any
    right_numerators: Any
    left_planes: Any
    right_planes: Any
    expected_grouped_partials: Any
    weights: tuple[int, ...]
    expected_rails: Any
    expected_numerator: Any
    output_scale: int
    max_abs_bound: int
    schema_version: int = SCHEMA_VERSION

    @property
    def config(self) -> WideRNSConfig:
        return WideRNSConfig(self.moduli, f"fixture:{self.name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "radix": self.radix,
            "moduli": list(self.moduli),
            "left_scale": self.left_scale,
            "right_scale": self.right_scale,
            "left_numerators": self.left_numerators,
            "right_numerators": self.right_numerators,
            "left_planes": self.left_planes,
            "right_planes": self.right_planes,
            "expected_grouped_partials": self.expected_grouped_partials,
            "weights": list(self.weights),
            "expected_rails": self.expected_rails,
            "expected_numerator": self.expected_numerator,
            "output_scale": self.output_scale,
            "max_abs_bound": self.max_abs_bound,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(self.to_json() + "\n", encoding="utf-8")
        return destination

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CudaGemmFixture:
        version = int(payload.get("schema_version", -1))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported CUDA fixture schema {version}; expected {SCHEMA_VERSION}"
            )
        return cls(
            schema_version=version,
            name=str(payload["name"]),
            radix=int(payload["radix"]),
            moduli=_int_tuple(payload["moduli"]),
            left_scale=int(payload["left_scale"]),
            right_scale=int(payload["right_scale"]),
            left_numerators=payload["left_numerators"],
            right_numerators=payload["right_numerators"],
            left_planes=payload["left_planes"],
            right_planes=payload["right_planes"],
            expected_grouped_partials=payload["expected_grouped_partials"],
            weights=_int_tuple(payload["weights"]),
            expected_rails=payload["expected_rails"],
            expected_numerator=payload["expected_numerator"],
            output_scale=int(payload["output_scale"]),
            max_abs_bound=int(payload["max_abs_bound"]),
        )

    @classmethod
    def from_json(cls, text: str) -> CudaGemmFixture:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise TypeError("fixture JSON root must be an object")
        return cls.from_dict(payload)

    @classmethod
    def read_json(cls, path: str | Path) -> CudaGemmFixture:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


class ExactPipelineBackend(Protocol):
    def grouped_partials(
        self,
        left_planes: np.ndarray,
        right_planes: np.ndarray,
    ) -> np.ndarray: ...

    def weighted_rails(
        self,
        grouped_partials: np.ndarray,
        weights: tuple[int, ...],
        moduli: tuple[int, ...],
    ) -> tuple[np.ndarray, ...]: ...


class CpuExactPipelineBackend:
    name = "cpu-exact-reference"

    def grouped_partials(
        self,
        left_planes: np.ndarray,
        right_planes: np.ndarray,
    ) -> np.ndarray:
        return grouped_plane_gemm(left_planes, right_planes, require_int32=True)

    def weighted_rails(
        self,
        grouped_partials: np.ndarray,
        weights: tuple[int, ...],
        moduli: tuple[int, ...],
    ) -> tuple[np.ndarray, ...]:
        from .wide import accumulate_weighted_int32_wide

        return accumulate_weighted_int32_wide(
            np.asarray(grouped_partials, dtype=np.int32),
            weights,
            WideRNSConfig(moduli, "cpu-backend"),
            require_unique=False,
        ).rails


# Compatibility alias for the initial contract name.
CpuGroupedPartialsBackend = CpuExactPipelineBackend


@dataclass(frozen=True, slots=True)
class BackendVerification:
    backend_name: str
    grouped_partials_match: bool
    rails_match: bool
    numerator_match: bool

    @property
    def passed(self) -> bool:
        return self.grouped_partials_match and self.rails_match and self.numerator_match

    def require_passed(self) -> BackendVerification:
        if not self.passed:
            raise AssertionError(
                f"backend {self.backend_name!r} failed fixture verification: "
                f"partials={self.grouped_partials_match}, rails={self.rails_match}, "
                f"numerator={self.numerator_match}"
            )
        return self


def build_cuda_gemm_fixture(
    name: str,
    left: SharedScaleMatrix,
    right: SharedScaleMatrix,
    *,
    config: WideRNSConfig,
    radix: int = 128,
    left_plane_count: int | None = None,
    right_plane_count: int | None = None,
) -> CudaGemmFixture:
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    receipt = exact_shared_scale_gemm(
        left,
        right,
        config=config,
        radix=radix,
        left_plane_count=left_plane_count,
        right_plane_count=right_plane_count,
        require_unique=True,
    )
    return fixture_from_receipt(name, receipt)


def fixture_from_receipt(
    name: str,
    receipt: ExactSharedScaleGemmReceipt,
) -> CudaGemmFixture:
    return CudaGemmFixture(
        name=name,
        radix=receipt.radix,
        moduli=receipt.config.moduli,
        left_scale=receipt.left.scale,
        right_scale=receipt.right.scale,
        left_numerators=_array_to_lists(receipt.left.numerators),
        right_numerators=_array_to_lists(receipt.right.numerators),
        left_planes=_array_to_lists(receipt.left_planes),
        right_planes=_array_to_lists(receipt.right_planes),
        expected_grouped_partials=_array_to_lists(receipt.grouped_partials),
        weights=receipt.weighted_result.weights,
        expected_rails=[_array_to_lists(rail) for rail in receipt.weighted_result.rails],
        expected_numerator=_array_to_lists(receipt.reconstructed_numerator),
        output_scale=receipt.output_scale,
        max_abs_bound=receipt.weighted_result.max_abs_bound,
    )


def verify_backend(
    fixture: CudaGemmFixture,
    backend: ExactPipelineBackend,
) -> BackendVerification:
    left_planes = np.asarray(fixture.left_planes, dtype=np.int8)
    right_planes = np.asarray(fixture.right_planes, dtype=np.int8)
    candidate = np.asarray(
        backend.grouped_partials(left_planes, right_planes),
    )
    expected = np.asarray(fixture.expected_grouped_partials, dtype=np.int32)
    partials_match = candidate.dtype == np.int32 and np.array_equal(candidate, expected)

    config = fixture.config
    if candidate.dtype != np.int32:
        rails_match = False
        numerator_match = False
    else:
        candidate_rails = tuple(
            np.asarray(rail)
            for rail in backend.weighted_rails(
                candidate,
                fixture.weights,
                fixture.moduli,
            )
        )
        expected_rails = tuple(np.asarray(rail) for rail in fixture.expected_rails)
        rails_match = len(candidate_rails) == len(expected_rails) and all(
            np.array_equal(actual, expected)
            for actual, expected in zip(candidate_rails, expected_rails)
        )
        if rails_match:
            decoded = config.decode(candidate_rails, signed=True)
            numerator_match = config.uniquely_represents_bound(
                fixture.max_abs_bound
            ) and np.array_equal(
                decoded,
                np.asarray(fixture.expected_numerator, dtype=object),
            )
        else:
            numerator_match = False

    verification = BackendVerification(
        backend_name=getattr(backend, "name", type(backend).__name__),
        grouped_partials_match=partials_match,
        rails_match=rails_match,
        numerator_match=numerator_match,
    )
    return verification
