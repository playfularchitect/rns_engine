"""
tests/test_rns.py — correctness tests for rns_engine.

Run with:  pytest tests/ -v
"""

import pytest
import numpy as np
import rns_engine as rns


M = rns.M


# ── helpers ───────────────────────────────────────────────────────────────
def make(n, seed=42, odd_b=False):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, M, size=n, dtype=np.uint64)
    if odd_b:
        b = rng.integers(0, M // 2, size=n, dtype=np.uint64) * 2 + 1
    else:
        b = rng.integers(0, M, size=n, dtype=np.uint64)
    return a, b


def oracle(a_np, b_np, op):
    """Exact Python arbitrary-precision oracle."""
    n = len(a_np)
    result = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        ai, bi = int(a_np[i]), int(b_np[i])
        if   op == "add": result[i] = (ai + bi) % M
        elif op == "sub": result[i] = (ai - bi) % M
        elif op == "mul": result[i] = (ai * bi) % M
    return result


# ── basic sanity ──────────────────────────────────────────────────────────
def test_constants():
    assert rns.M  == 127 * 8191 * 65536
    assert rns.M0 == 127
    assert rns.M1 == 8191
    assert rns.M2 == 65536


def test_info_runs():
    rns.info()   # just check it doesn't crash


# ── encode / decode ───────────────────────────────────────────────────────
def test_roundtrip_small():
    vals = np.array([0, 1, 126, 127, 8190, 8191, 65535, M-1], dtype=np.uint64)
    assert np.array_equal(vals, rns.decode(*rns.encode(vals)))


def test_roundtrip_random():
    a, _ = make(10_000)
    assert np.array_equal(a, rns.decode(*rns.encode(a)))


def test_encode_reduces_mod_M():
    # values >= M should be reduced
    vals = np.array([M, M+1, M*2], dtype=np.uint64)
    decoded = rns.decode(*rns.encode(vals))
    assert np.array_equal(decoded, np.array([0, 1, 0], dtype=np.uint64))


# ── addition ──────────────────────────────────────────────────────────────
def test_add_correctness():
    a, b = make(5_000)
    got = rns.decode(*rns.add(*rns.encode(a), *rns.encode(b)))
    exp = oracle(a, b, "add")
    assert np.array_equal(got, exp)


def test_add_zero():
    a, _ = make(100)
    z = np.zeros(100, dtype=np.uint64)
    got = rns.decode(*rns.add(*rns.encode(a), *rns.encode(z)))
    assert np.array_equal(got, a)


def test_add_wraps():
    a = np.array([M - 1], dtype=np.uint64)
    b = np.array([1],     dtype=np.uint64)
    got = rns.decode(*rns.add(*rns.encode(a), *rns.encode(b)))
    assert got[0] == 0


# ── subtraction ───────────────────────────────────────────────────────────
def test_sub_correctness():
    a, b = make(5_000)
    got = rns.decode(*rns.sub(*rns.encode(a), *rns.encode(b)))
    exp = oracle(a, b, "sub")
    assert np.array_equal(got, exp)


def test_sub_self_is_zero():
    a, _ = make(100)
    got = rns.decode(*rns.sub(*rns.encode(a), *rns.encode(a)))
    assert np.all(got == 0)


def test_sub_wraps():
    a = np.array([0], dtype=np.uint64)
    b = np.array([1], dtype=np.uint64)
    got = rns.decode(*rns.sub(*rns.encode(a), *rns.encode(b)))
    assert got[0] == M - 1


# ── multiplication ────────────────────────────────────────────────────────
def test_mul_correctness():
    a, b = make(5_000)
    got = rns.decode(*rns.mul(*rns.encode(a), *rns.encode(b)))
    exp = oracle(a, b, "mul")
    assert np.array_equal(got, exp)


def test_mul_by_zero():
    a, _ = make(100)
    z = np.zeros(100, dtype=np.uint64)
    got = rns.decode(*rns.mul(*rns.encode(a), *rns.encode(z)))
    assert np.all(got == 0)


def test_mul_by_one():
    a, _ = make(100)
    one = np.ones(100, dtype=np.uint64)
    got = rns.decode(*rns.mul(*rns.encode(a), *rns.encode(one)))
    assert np.array_equal(got, a)


# ── division ──────────────────────────────────────────────────────────────
def test_div_correctness():
    # Use odd b values that are invertible on all rails
    a, b = make(1_000, odd_b=True)
    # also ensure nonzero mod 127 and 8191
    b = np.where(b % 127  == 0, b + 1, b)
    b = np.where(b % 8191 == 0, b + 2, b)
    b = b % M
    got  = rns.decode(*rns.div_(*rns.encode(a), *rns.encode(b)))
    # oracle: per-integer Python division in the field
    from math import gcd
    def mi(a, m):
        a = a % m
        if a == 0: return 0
        def eg(a, b):
            if not a: return b, 0, 1
            g, x, y = eg(b % a, a)
            return g, y - (b // a) * x, x
        g, x, _ = eg(a, m)
        return x % m if g == 1 else 0
    exp = np.array([
        int(rns.decode(
            np.array([int(a[i]) % 127  * mi(int(b[i]) % 127,  127)  % 127],  dtype=np.uint16),
            np.array([int(a[i]) % 8191 * mi(int(b[i]) % 8191, 8191) % 8191], dtype=np.uint32),
            np.array([int(a[i]) % 65536* mi(int(b[i]) % 65536,65536)% 65536],dtype=np.uint16),
        )[0]) for i in range(len(a))
    ], dtype=np.uint64)
    assert np.array_equal(got, exp)


# ── algebraic identities ──────────────────────────────────────────────────
def test_identity_sub_add(n=500):
    """a - b + b == a"""
    a, b = make(n)
    ea, eb = rns.encode(a), rns.encode(b)
    s1 = rns.sub(*ea, *eb)
    s2 = rns.add(*s1,  *eb)
    assert np.array_equal(rns.decode(*s2), a)


def test_identity_mul_div(n=200):
    """a * b / b == a  (b invertible)"""
    a, b = make(n, odd_b=True)
    b = np.where(b % 127  == 0, b + 1, b)
    b = np.where(b % 8191 == 0, b + 2, b)
    b = b % M
    ea, eb = rns.encode(a), rns.encode(b)
    s1 = rns.mul( *ea, *eb)
    s2 = rns.div_(*s1, *eb)
    assert np.array_equal(rns.decode(*s2), a)


def test_identity_distributive(n=500):
    """(a + b) * c == a*c + b*c"""
    rng = np.random.default_rng(99)
    a = rng.integers(0, M, size=n, dtype=np.uint64)
    b = rng.integers(0, M, size=n, dtype=np.uint64)
    c = rng.integers(0, M, size=n, dtype=np.uint64)
    ea, eb, ec = rns.encode(a), rns.encode(b), rns.encode(c)
    lhs = rns.decode(*rns.mul(*rns.add(*ea, *eb), *ec))
    rhs = rns.decode(*rns.add(*rns.mul(*ea, *ec), *rns.mul(*eb, *ec)))
    assert np.array_equal(lhs, rhs)


def test_identity_additive_inverse(n=500):
    """a + (-a) == 0"""
    a, _ = make(n)
    ea = rns.encode(a)
    ez = rns.encode(np.zeros(n, dtype=np.uint64))
    neg_a = rns.sub(*ez, *ea)
    result = rns.decode(*rns.add(*ea, *neg_a))
    assert np.all(result == 0)


def test_chain_expression(n=1000):
    """((a + b) * c - d) * e  matches Python exact arithmetic"""
    rng = np.random.default_rng(7)
    a,b,c,d,e = [rng.integers(0, 1000, size=n, dtype=np.uint64) for _ in range(5)]
    py_exp = ((a.astype(object)+b)*c-d)*e % M

    s1 = rns.add(*rns.encode(a), *rns.encode(b))
    s2 = rns.mul(*s1,             *rns.encode(c))
    s3 = rns.sub(*s2,             *rns.encode(d))
    s4 = rns.mul(*s3,             *rns.encode(e))
    result = rns.decode(*s4)

    assert np.array_equal(result, py_exp.astype(np.uint64))


# ── op() generic interface ────────────────────────────────────────────────
def test_op_matches_named_functions():
    a, b = make(100)
    ea, eb = rns.encode(a), rns.encode(b)
    for code, fn in [(0, rns.add), (1, rns.mul), (2, rns.sub)]:
        via_op   = rns.decode(*rns.op(*ea, *eb, code))
        via_name = rns.decode(*fn(*ea, *eb))
        assert np.array_equal(via_op, via_name)


def test_op_invalid_opcode():
    a, _ = make(10)
    ea = rns.encode(a)
    with pytest.raises(Exception):
        rns.op(*ea, *ea, 99)
