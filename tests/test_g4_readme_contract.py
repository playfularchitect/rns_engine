from pathlib import Path


def test_readme_defines_xops_and_four_g4_apis():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "**XOP** — one mathematically exact arithmetic operation." in text
    assert "**XOPS** — exact arithmetic operations per second." in text
    assert "**G4OPS** — XOPS delivered by a G4 implementation." in text
    assert "g4_integer_benchmark" in text
    assert "g4_rational_benchmark" in text
    assert "g4_matmul" in text
