import numpy as np

import rns_engine as rns


def _fixture():
    left = rns.SharedScaleMatrix(
        np.asarray([[2**38 + 3, -17], [29, 2**34 - 5]], dtype=object),
        scale=6,
    )
    right = rns.SharedScaleMatrix(
        np.asarray([[31, -(2**30) + 9], [2**33 + 1, 43]], dtype=object),
        scale=35,
    )
    return rns.build_cuda_gemm_fixture(
        "unit-fixture",
        left,
        right,
        config=rns.WideRNSConfig.balanced_seven_rail(),
        left_plane_count=8,
        right_plane_count=8,
    )


def test_cuda_fixture_json_round_trip_and_cpu_backend_verification(tmp_path):
    fixture = _fixture()
    path = fixture.write_json(tmp_path / "fixture.json")
    loaded = rns.CudaGemmFixture.read_json(path)

    assert loaded == fixture
    verification = rns.verify_backend(loaded, rns.CpuExactPipelineBackend())
    assert verification.passed
    assert verification.require_passed() is verification


def test_cuda_fixture_detects_wrong_backend_output():
    fixture = _fixture()

    class WrongBackend:
        name = "wrong"

        def grouped_partials(self, left_planes, right_planes):
            expected = np.asarray(fixture.expected_grouped_partials, dtype=np.int32)
            return np.zeros_like(expected)

        def weighted_rails(self, grouped_partials, weights, moduli):
            config = rns.WideRNSConfig(moduli)
            return rns.accumulate_weighted_int32_wide(
                np.asarray(grouped_partials, dtype=np.int32),
                weights,
                config,
            ).rails

    verification = rns.verify_backend(fixture, WrongBackend())
    assert not verification.passed
    assert not verification.grouped_partials_match
