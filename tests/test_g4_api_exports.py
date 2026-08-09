import rns_engine as rns


def test_four_public_g4_entry_points_are_exported():
    assert callable(rns.g4_benchmark)
    assert callable(rns.g4_integer_benchmark)
    assert callable(rns.g4_rational_benchmark)
    assert callable(rns.g4_matmul)
