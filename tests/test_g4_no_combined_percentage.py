from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_readme_does_not_define_a_combined_integer_rational_win_rate():
    text = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "does **not** combine them into one win percentage" in text
