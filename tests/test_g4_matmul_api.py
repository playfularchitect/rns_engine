import importlib

import numpy as np
import pytest

from rns_engine.exact_gemm import SharedScaleMatrix

gm = importlib.import_module("rns_engine.g4_matmul")


def test_int8_range_rejected_before_native():
    with pytest.raises(OverflowError, match="signed INT8"):
        gm._integer_matrix([[128]], name="left")


def test_integer_result_shape_and_dtype(monkeypatch):
    monkeypatch.setattr(gm, "shape_map", lambda: {(2, 2, 2): {"family": "generic"}})
    monkeypatch.setattr(gm, "_run_native", lambda a, b, family: np.array([[19, 22], [43, 50]], dtype=np.int32))
    got = gm.g4_matmul(
        np.array([[1, 2], [3, 4]], dtype=np.int8),
        np.array([[5, 6], [7, 8]], dtype=np.int8),
    )
    assert got.dtype == np.int32
    assert got.tolist() == [[19, 22], [43, 50]]


def test_rational_metadata_is_exact(monkeypatch):
    monkeypatch.setattr(gm, "shape_map", lambda: {(1, 1, 1): {"family": "generic"}})
    monkeypatch.setattr(gm, "_run_native", lambda a, b, family: np.array([[6]], dtype=np.int32))
    left = SharedScaleMatrix(np.array([[2]], dtype=object), 3)
    right = SharedScaleMatrix(np.array([[3]], dtype=object), 5)
    got = gm.g4_matmul(left, right)
    assert isinstance(got, SharedScaleMatrix)
    assert int(got.numerators[0, 0]) == 6
    assert got.scale == 15


def test_unsupported_shape_fails_closed(monkeypatch):
    monkeypatch.setattr(gm, "shape_map", lambda: {})
    with pytest.raises(NotImplementedError, match="does not have a certified fast implementation"):
        gm.g4_matmul(np.ones((2, 2), dtype=np.int8), np.ones((2, 2), dtype=np.int8))
