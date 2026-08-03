import json

import rns_engine as rns


def test_pre_cuda_readiness_report_is_fully_green_and_serializable(tmp_path):
    report = rns.run_pre_cuda_readiness()

    assert report.ready
    assert report.require_ready() is report
    assert report.target_required_modulus_bits == 126
    assert report.target_additional_bits == 71
    assert report.minimum_extra_rails_under_31_bits == 3
    assert all(report.checks.values())

    payload = json.loads(report.to_json())
    assert payload["ready"] is True
    path = report.write_json(tmp_path / "readiness.json")
    assert json.loads(path.read_text())["fixture_name"] == report.fixture_name


def test_parallel_lane_prior_preserves_shared_capacity_and_distinct_objectives():
    capacity = rns.plan_digit_plane_gemm_capacity(
        5120,
        128,
        [127] * 8,
        [127] * 8,
    ).capacity
    prior = rns.build_parallel_rail_prior(capacity)

    assert prior.required_additional_product == capacity.minimum_additional_product_factor
    assert [lane.objective for lane in prior.lanes] == ["smallest_product", "balanced"]
    assert prior.lanes[0].plan.exponents == (11, 29, 31)
    assert prior.lanes[1].plan.exponents == (23, 24, 25)
    assert all(lane.plan.sufficient for lane in prior.lanes)


def test_default_fixture_helper_is_frozen_and_verifiable(tmp_path):
    fixture = rns.build_default_pre_cuda_fixture()
    path = fixture.write_json(tmp_path / "cuda_fixture.json")
    loaded = rns.CudaGemmFixture.read_json(path)

    assert loaded.name == "pre-cuda-eight-plane-witness"
    assert loaded.output_scale == 210
    assert rns.verify_backend(loaded, rns.CpuExactPipelineBackend()).passed
