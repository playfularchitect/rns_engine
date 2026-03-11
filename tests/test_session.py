import numpy as np
import pytest
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
    assert info1["size"] == info0["size"]


def test_session_cache_zero_capacity_never_stores():
    s = rns.Session(cache_capacity=0)
    x = np.arange(8, dtype=np.uint64)

    _ = s.encode(x)
    info0 = dict(s.cache_info())
    _ = s.encode(x)
    info1 = dict(s.cache_info())

    assert info0["size"] == 0
    assert info1["size"] == 0
    assert info1["hits"] == 0
    assert info1["misses"] == info0["misses"] + 1


def test_session_clear_cache_resets_stats():
    s = rns.Session(cache_capacity=4)
    x = np.arange(8, dtype=np.uint64)

    _ = s.encode(x)
    _ = s.encode(x)
    assert s.cache_info()["hits"] >= 1

    s.clear_cache()
    info = dict(s.cache_info())

    assert info == {"capacity": 4, "size": 0, "hits": 0, "misses": 0, "evictions": 0}


def test_encoded_ops_match_core_pipeline():
    s = rns.Session(cache_capacity=8)
    a = np.array([5, 7, 11, 13], dtype=np.uint64)
    b = np.array([2, 3, 4, 5], dtype=np.uint64)

    ea = s.encode(a)
    eb = s.encode(b)

    got = s.decode(s.mul(s.add(ea, eb), eb))
    exp = rns.decode(*rns.mul(*rns.add(*rns.encode(a), *rns.encode(b)), *rns.encode(b)))

    assert np.array_equal(got, exp)


def test_session_div_matches_core():
    s = rns.Session(cache_capacity=8)
    a = np.array([5, 7, 11, 13], dtype=np.uint64)
    b = np.array([3, 5, 7, 9], dtype=np.uint64)  # odd, nonzero mod 127/8191 here

    ea = s.encode(a)
    eb = s.encode(b)

    got = s.decode(s.div(ea, eb))
    exp = rns.decode(*rns.div_(*rns.encode(a), *rns.encode(b)))

    assert np.array_equal(got, exp)


def test_session_fma_matches_core():
    s = rns.Session(cache_capacity=8)
    a = np.array([1, 2, 3, 4], dtype=np.uint64)
    b = np.array([5, 6, 7, 8], dtype=np.uint64)
    c = np.array([9, 10, 11, 12], dtype=np.uint64)

    ea = s.encode(a)
    eb = s.encode(b)
    ec = s.encode(c)

    got = s.decode(s.fma(ea, eb, ec))
    exp = rns.decode(*rns.fma(*rns.encode(a), *rns.encode(b), *rns.encode(c)))

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


def test_hot_loop_affine_zero_iterations_is_identity():
    s = rns.Session(cache_capacity=8)
    x = np.array([1, 2, 3, 4, 5], dtype=np.uint64)

    got = s.hot_loop_affine(x, multiplier=123, addend=456, iterations=0)

    assert np.array_equal(got, x % rns.M)


def test_hot_loop_affine_rejects_negative_iterations():
    s = rns.Session(cache_capacity=8)
    x = np.array([1, 2, 3], dtype=np.uint64)

    with pytest.raises(ValueError):
        s.hot_loop_affine(x, multiplier=3, addend=1, iterations=-1)


def test_encoded_array_is_read_only():
    s = rns.Session(cache_capacity=8)
    x = np.array([1, 2, 3], dtype=np.uint64)

    enc = s.encode(x)

    assert enc.r0.flags.writeable is False
    assert enc.r1.flags.writeable is False
    assert enc.r2.flags.writeable is False
    assert enc.r3.flags.writeable is False

    with pytest.raises(ValueError):
        enc.r0[0] = 99


def test_cached_encode_returns_same_object_for_same_key():
    s = rns.Session(cache_capacity=8)
    x = np.arange(10, dtype=np.uint64)

    e0 = s.encode(x)
    e1 = s.encode(x)

    assert e0 is e1


def test_cache_tag_separates_entries():
    s = rns.Session(cache_capacity=8)
    x = np.arange(6, dtype=np.uint64)

    e0 = s.encode(x, tag="a")
    e1 = s.encode(x, tag="b")

    assert e0 is not e1
    assert s.cache_info()["size"] == 2


def test_cache_eviction_occurs():
    s = rns.Session(cache_capacity=2)

    _ = s.encode(np.array([1], dtype=np.uint64), tag="x1")
    _ = s.encode(np.array([2], dtype=np.uint64), tag="x2")
    info0 = dict(s.cache_info())

    _ = s.encode(np.array([3], dtype=np.uint64), tag="x3")
    info1 = dict(s.cache_info())

    assert info0["size"] == 2
    assert info1["size"] == 2
    assert info1["evictions"] == info0["evictions"] + 1


def test_service_identity_roundtrip():
    s = rns.Session(cache_capacity=8)
    x = np.array([10, 20, 30], dtype=np.uint64)

    got = s.service(x)

    assert np.array_equal(got, x)


def test_encode_rejects_non_1d_input():
    s = rns.Session(cache_capacity=8)
    x = np.array([[1, 2], [3, 4]], dtype=np.uint64)

    with pytest.raises(ValueError):
        s.encode(x)
