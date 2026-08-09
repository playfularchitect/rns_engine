from io import StringIO

from rns_engine.g4_xops import print_xops_key


def test_benchmark_xops_key_uses_canonical_user_approved_wording():
    out = StringIO()
    print_xops_key(out)
    text = out.getvalue()
    assert "XOP   = one mathematically exact arithmetic operation." in text
    assert "XOPS  = exact arithmetic operations per second." in text
    assert "G4OPS = XOPS delivered by a G4 implementation." in text
