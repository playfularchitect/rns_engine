from pathlib import Path


def test_readme_does_not_define_a_combined_integer_rational_win_rate():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "does **not** combine them into one win percentage" in text
