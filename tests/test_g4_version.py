from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_0_12_0():
    text = (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.12.0"' in text
