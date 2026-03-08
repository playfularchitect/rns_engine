import numpy as np
import rns_engine as rns


def test_session_roundtrip_matches_core():
    s = rns.Session(cache_capacity=8)
    x = np.array([0, 1, 2, 123456789, rns.M - 1], dtype=np.uint64)
    enc = s.encode(x)
    got = s.decode(enc)
    assert np.array_equal(got, x)


def test_session_cache_hits_on_reencode():
    s = rns.Session(cache_capacity=8)
    x = np.arange(16, dtype=np.uint64)
    _ = s.encode(x)
    info0 = dict(s.cache_info())
    _ = s.encode(x)
    info1 = dict(s.cache_info())
    assert info1["hits"] == info0["hits"] + 1


def test_encoded_ops_match_core_pipeline():
    s = rns.Session(cache_capacity=8)
    a = np.array([5, 7, 11, 13], dtype=np.uint64)
    b = np.array([2, 3, 4, 5], dtype=np.uint64)
    ea = s.encode(a)
    eb = s.encode(b)
    got = s.decode(s.mul(s.add(ea, eb), eb))
    exp = rns.decode(*rns.mul(*rns.add(*rns.encode(a), *rns.encode(b)), *rns.encode(b)))
    assert np.array_equal(got, exp)


def test_one_shot_affine_matches_python():
    s = rns.Session(cache_capacity=8)
    x = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
    got = s.one_shot_affine(x, multiplier=1_000_003, addend=7)
    exp = ((x.astype(object) * 1_000_003) + 7) % rns.M
    assert np.array_equal(got, np.asarray(exp, dtype=np.uint64))


def test_hot_loop_affine_matches_python():
    s = rns.Session(cache_capacity=8)
    x = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
    got = s.hot_loop_affine(x, multiplier=1_000_003, addend=7, iterations=5)
    cur = x.astype(object)
    for _ in range(5):
        cur = (cur * 1_000_003 + 7) % rns.M
    exp = np.asarray(cur, dtype=np.uint64)
    assert np.array_equal(got, exp)
