"""
tests/test_rns.py — correctness tests for rns_engine.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pytest
import rns_engine as rns


M = int(rns.M)


# ── helpers ───────────────────────────────────────────────────────────────
def make(n: int, seed: int = 42, odd_b: bool = False):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, M, size=n, dtype=np.uint64)
    if odd_b:
        b = rng.integers(0, M // 2, size=n, dtype=np.uint64) * 2 + 1
    else:
        b = rng.integers(0, M, size=n, dtype=np.uint64)
    return a, b


def make_invertible_divisor(n: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)

    b = rng.integers(0, M // 2, size=n, dtype=np.uint64) * 2 + 1
    b = b % np.uint64(M)

    bad = (
        (b % np.uint64(127) == 0)
        | (b % np.uint64(8191) == 0)
        | (b % np.uint64(524287) == 0)
        | ((b & np.uint64(1)) == 0)
    )

    while np.any(bad):
        b = np.where(bad, (b + np.uint64(2)) % np.uint64(M), b)
        bad = (
            (b % np.uint64(127) == 0)
            | (b % np.uint64(8191) == 0)
            | (b % np.uint64(524287) == 0)
            | ((b & np.uint64(1)) == 0)
        )

    return b.astype(np.uint64)


def oracle(a_np: np.ndarray, b_np: np.ndarray, op: str) -> np.ndarray:
    """Exact Python oracle modulo M."""
    n = len(a_np)
    result = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        ai, bi = int(a_np[i]), int(b_np[i])
        if op == "add":
            result[i] = (ai + bi) % M
        elif op == "sub":
            result[i] = (ai - bi) % M
        elif op == "mul":
            result[i] = (ai * bi) % M
        else:
            raise ValueError(f"unknown op: {op}")
    return result


def oracle_fma(a_np: np.ndarray, b_np: np.ndarray, c_np: np.ndarray) -> np.ndarray:
    n = len(a_np)
    result = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        ai, bi, ci = int(a_np[i]), int(b_np[i]), int(c_np[i])
        result[i] = (ai * bi + ci) % M
    return result


def modinv(a: int, m: int) -> int:
    a = a % m
    if a == 0:
        raise ValueError("non-invertible")

    def eg(x: int, y: int):
        if x == 0:
            return y, 0, 1
        g, s, t = eg(y % x, x)
        return g, t - (y // x) * s, s

    g, x, _ = eg(a, m)
    if g != 1:
        raise ValueError("non-invertible")
    return x % m


def division_oracle(a_np: np.ndarray, b_np: np.ndarray) -> np.ndarray:
    """Exact rail-wise division oracle using Python modular inverses."""
    n = len(a_np)
    out = np.zeros(n, dtype=np.uint64)

    for i in range(n):
        ai = int(a_np[i])
        bi = int(b_np[i])

        r0 = (ai % 127) * modinv(bi % 127, 127) % 127
        r1 = (ai % 8191) * modinv(bi % 8191, 8191) % 8191
        r2 = (ai % 65536) * modinv(bi % 65536, 65536) % 65536
        r3 = (ai % 524287) * modinv(bi % 524287, 524287) % 524287

        out[i] = int(
            rns.decode(
                np.array([r0], dtype=np.uint16),
                np.array([r1], dtype=np.uint16),
                np.array([r2], dtype=np.uint16),
                np.array([r3], dtype=np.uint32),
            )[0]
        )

    return out


# ── basic sanity ──────────────────────────────────────────────────────────
def test_constants():
    assert rns.M == 127 * 8191 * 65536 * 524287
    assert rns.M0 == 127
    assert rns.M1 == 8191
    assert rns.M2 == 65536
    assert rns.M3 == 524287


def test_info_runs():
    rns.info()


# ── encode / decode ───────────────────────────────────────────────────────
def test_roundtrip_small():
    vals = np.array([0, 1, 126, 127, 8190, 8191, 65535, 524286, M - 1], dtype=np.uint64)
    assert np.array_equal(vals, rns.decode(*rns.encode(vals)))


def test_roundtrip_random():
    a, _ = make(10_000)
    assert np.array_equal(a, rns.decode(*rns.encode(a)))


def test_encode_reduces_mod_M():
    vals = np.array([M, M + 1, M * 2], dtype=np.uint64)
    decoded = rns.decode(*rns.encode(vals))
    assert np.array_equal(decoded, np.array([0, 1, 0], dtype=np.uint64))


def test_encode_accepts_noncontiguous_input():
    base = np.arange(40, dtype=np.uint64)
    vals = base[::2]
    got = rns.decode(*rns.encode(vals))
    assert np.array_equal(got, vals % M)


def test_decode_accepts_noncontiguous_rails():
    vals = np.arange(30, dtype=np.uint64)
    e = rns.encode(vals)
    sliced = tuple(rail[::2] for rail in e)
    got = rns.decode(*sliced)
    assert np.array_equal(got, vals[::2] % M)


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
    b = np.array([1], dtype=np.uint64)
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


# ── fma ───────────────────────────────────────────────────────────────────
def test_fma_correctness():
    rng = np.random.default_rng(202)
    a = rng.integers(0, M, size=5_000, dtype=np.uint64)
    b = rng.integers(0, M, size=5_000, dtype=np.uint64)
    c = rng.integers(0, M, size=5_000, dtype=np.uint64)

    got = rns.decode(*rns.fma(*rns.encode(a), *rns.encode(b), *rns.encode(c)))
    exp = oracle_fma(a, b, c)
    assert np.array_equal(got, exp)


def test_fma_matches_mul_then_add():
    rng = np.random.default_rng(303)
    a = rng.integers(0, M, size=500, dtype=np.uint64)
    b = rng.integers(0, M, size=500, dtype=np.uint64)
    c = rng.integers(0, M, size=500, dtype=np.uint64)

    ea, eb, ec = rns.encode(a), rns.encode(b), rns.encode(c)
    via_fma = rns.decode(*rns.fma(*ea, *eb, *ec))
    via_two_step = rns.decode(*rns.add(*rns.mul(*ea, *eb), *ec))
    assert np.array_equal(via_fma, via_two_step)


# ── division ──────────────────────────────────────────────────────────────
def test_div_correctness():
    a, _ = make(1_000, seed=777)
    b = make_invertible_divisor(1_000, seed=888)

    got = rns.decode(*rns.div_(*rns.encode(a), *rns.encode(b)))
    exp = division_oracle(a, b)
    assert np.array_equal(got, exp)


def test_div_rejects_noninvertible_mod_127():
    a = np.array([5], dtype=np.uint64)
    b = np.array([127], dtype=np.uint64)
    with pytest.raises(ValueError):
        rns.div_(*rns.encode(a), *rns.encode(b))


def test_div_rejects_noninvertible_mod_8191():
    a = np.array([5], dtype=np.uint64)
    b = np.array([8191], dtype=np.uint64)
    with pytest.raises(ValueError):
        rns.div_(*rns.encode(a), *rns.encode(b))


def test_div_rejects_even_divisor_mod_65536():
    a = np.array([5], dtype=np.uint64)
    b = np.array([2], dtype=np.uint64)
    with pytest.raises(ValueError):
        rns.div_(*rns.encode(a), *rns.encode(b))


def test_div_rejects_noninvertible_mod_524287():
    a = np.array([5], dtype=np.uint64)
    b = np.array([524287], dtype=np.uint64)
    with pytest.raises(ValueError):
        rns.div_(*rns.encode(a), *rns.encode(b))


# ── algebraic identities ──────────────────────────────────────────────────
def test_identity_sub_add():
    n = 500
    a, b = make(n)
    ea, eb = rns.encode(a), rns.encode(b)
    s1 = rns.sub(*ea, *eb)
    s2 = rns.add(*s1, *eb)
    assert np.array_equal(rns.decode(*s2), a)


def test_identity_mul_div():
    n = 200
    a, _ = make(n, seed=11)
    b = make_invertible_divisor(n, seed=12)
    ea, eb = rns.encode(a), rns.encode(b)
    s1 = rns.mul(*ea, *eb)
    s2 = rns.div_(*s1, *eb)
    assert np.array_equal(rns.decode(*s2), a)


def test_identity_distributive():
    n = 500
    rng = np.random.default_rng(99)
    a = rng.integers(0, M, size=n, dtype=np.uint64)
    b = rng.integers(0, M, size=n, dtype=np.uint64)
    c = rng.integers(0, M, size=n, dtype=np.uint64)

    ea, eb, ec = rns.encode(a), rns.encode(b), rns.encode(c)
    lhs = rns.decode(*rns.mul(*rns.add(*ea, *eb), *ec))
    rhs = rns.decode(*rns.add(*rns.mul(*ea, *ec), *rns.mul(*eb, *ec)))
    assert np.array_equal(lhs, rhs)


def test_identity_additive_inverse():
    n = 500
    a, _ = make(n)
    ea = rns.encode(a)
    ez = rns.encode(np.zeros(n, dtype=np.uint64))
    neg_a = rns.sub(*ez, *ea)
    result = rns.decode(*rns.add(*ea, *neg_a))
    assert np.all(result == 0)


def test_chain_expression():
    n = 1_000
    rng = np.random.default_rng(7)
    a, b, c, d, e = [rng.integers(0, 1000, size=n, dtype=np.uint64) for _ in range(5)]
    py_exp = ((a.astype(object) + b) * c - d) * e % M

    s1 = rns.add(*rns.encode(a), *rns.encode(b))
    s2 = rns.mul(*s1, *rns.encode(c))
    s3 = rns.sub(*s2, *rns.encode(d))
    s4 = rns.mul(*s3, *rns.encode(e))
    result = rns.decode(*s4)

    assert np.array_equal(result, py_exp.astype(np.uint64))


# ── validation / generic interface ────────────────────────────────────────
def test_op_matches_named_functions():
    a, b = make(100)
    ea, eb = rns.encode(a), rns.encode(b)
    for code, fn in [(0, rns.add), (1, rns.mul), (2, rns.sub)]:
        via_op = rns.decode(*rns.op(*ea, *eb, code))
        via_name = rns.decode(*fn(*ea, *eb))
        assert np.array_equal(via_op, via_name)


def test_op_invalid_opcode():
    a, _ = make(10)
    ea = rns.encode(a)
    with pytest.raises(ValueError):
        rns.op(*ea, *ea, 99)


def test_op_rejects_length_mismatch():
    a = np.array([1, 2, 3], dtype=np.uint64)
    b = np.array([4, 5], dtype=np.uint64)

    ea = rns.encode(a)
    eb = rns.encode(b)

    with pytest.raises(ValueError):
        rns.add(*ea, *eb)


def test_fma_rejects_length_mismatch():
    a = np.array([1, 2, 3], dtype=np.uint64)
    b = np.array([4, 5, 6], dtype=np.uint64)
    c = np.array([7, 8], dtype=np.uint64)

    ea = rns.encode(a)
    eb = rns.encode(b)
    ec = rns.encode(c)

    with pytest.raises(ValueError):
        rns.fma(*ea, *eb, *ec)
