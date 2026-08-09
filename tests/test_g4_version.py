from pathlib import Path


def test_release_version_is_0_12_0():
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.12.0"' in text
