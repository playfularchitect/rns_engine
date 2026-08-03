import json
from pathlib import Path


def test_cuda_notebook_has_one_explicit_backend_integration_point():
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "colab" / "T4_EXACT_GEMM_BRINGUP.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert notebook["nbformat"] == 4
    assert "CUDA_BACKEND_INTEGRATION_POINT" in source
    assert "TODO_CUDA_GROUPED_PARTIALS" in source
    assert "TODO_CUDA_WEIGHTED_RAILS" in source
    assert "verification.require_passed()" in source
    assert "report.require_ready()" in source


def test_source_manifest_contains_reproducibility_assets():
    root = Path(__file__).resolve().parents[1]
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")

    assert "CAPACITY_PLANNING.md" in manifest
    assert "PRE_CUDA_READINESS.md" in manifest
    assert "CUDA_PARALLEL_LANES.md" in manifest
    assert "recursive-include benchmarks *.py" in manifest
    assert "recursive-include colab *.ipynb" in manifest
