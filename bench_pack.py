
import sys, time, hashlib, statistics
import numpy as np

sys.path.insert(0, "/content/rns_core_test")
import _core as rns4

def sha16(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]

def med(xs):
    return statistics.median(xs)

def mean(xs):
    return sum(xs) / len(xs)

M = int(rns4.M)
ADD = 7
SUB = 7
MULT = 1_000_003
ITERATIONS = 1000
RUNS = 5
MASTER_SEED = 42
SIZES = [16_000, 64_000, 1_000_000]

print("=" * 80)
print("UNIFIED BENCHMARK PACK")
print("=" * 80)
print("module            :", rns4.__file__)
print("AVX2              :", rns4.HAS_AVX2)
print("M                 :", M)
print("omp_num_procs     :", rns4.omp_num_procs() if hasattr(rns4, "omp_num_procs") else "missing")
print("omp_max_threads   :", rns4.omp_max_threads() if hasattr(rns4, "omp_max_threads") else "missing")
print("has add_u64_auto  :", hasattr(rns4, "add_u64_auto"))
print("has sub_u64_auto  :", hasattr(rns4, "sub_u64_auto"))
print("has mul_u64_auto  :", hasattr(rns4, "mul_u64_auto"))
print("has fma_u64_auto  :", hasattr(rns4, "fma_u64_auto"))
print("has affine_repeat_u64_auto :", hasattr(rns4, "affine_repeat_u64_auto"))
print("=" * 80)
print(f"RUNS={RUNS}  SIZES={SIZES}  ITERATIONS={ITERATIONS}")
print("NOTE: repeat numbers are effective logical ops/s because affine steps are collapsed.")
print("=" * 80)

# ----------------------------------------------------------------------
# correctness sanity
# ----------------------------------------------------------------------
x = np.array([0, 1, 2, 3, 123456789, 9876543210123456 % M, M - 1], dtype=np.uint64)

ref_add = ((x.astype(object) + ADD) % M).astype(np.uint64)
ref_sub = ((x.astype(object) - SUB) % M).astype(np.uint64)
ref_mul = ((x.astype(object) * MULT) % M).astype(np.uint64)
ref_fma = ((x.astype(object) * MULT + ADD) % M).astype(np.uint64)
ref_rep = x.astype(object)
for _ in range(ITERATIONS):
    ref_rep = (ref_rep * MULT + ADD) % M
ref_rep = np.asarray(ref_rep, dtype=np.uint64)

rns4.omp_set_num_threads(2)

print("CORRECTNESS")
print("  add auto ok :", np.array_equal(rns4.add_u64_auto(x, ADD), ref_add))
print("  sub auto ok :", np.array_equal(rns4.sub_u64_auto(x, SUB), ref_sub))
print("  mul auto ok :", np.array_equal(rns4.mul_u64_auto(x, MULT), ref_mul))
print("  fma auto ok :", np.array_equal(rns4.fma_u64_auto(x, MULT, ADD), ref_fma))
print("  rep auto ok :", np.array_equal(rns4.affine_repeat_u64_auto(x, MULT, ADD, ITERATIONS), ref_rep))
print("=" * 80)

# ----------------------------------------------------------------------
# benchmark helpers
# ----------------------------------------------------------------------
def bench_add(x):
    out = {}

    t0 = time.perf_counter()
    y_io = rns4.add_u64_io(x, ADD)
    t1 = time.perf_counter()
    out["io_t"] = t1 - t0

    rns4.omp_set_num_threads(2)
    t2 = time.perf_counter()
    y_omp = rns4.add_u64_io_omp(x, ADD)
    t3 = time.perf_counter()
    out["omp_t"] = t3 - t2

    t4 = time.perf_counter()
    y_auto = rns4.add_u64_auto(x, ADD)
    t5 = time.perf_counter()
    out["auto_t"] = t5 - t4

    assert np.array_equal(y_io, y_omp)
    assert np.array_equal(y_io, y_auto)
    out["hash"] = sha16(y_auto)
    return out

def bench_sub(x):
    out = {}

    t0 = time.perf_counter()
    y_io = rns4.sub_u64_io(x, SUB)
    t1 = time.perf_counter()
    out["io_t"] = t1 - t0

    rns4.omp_set_num_threads(2)
    t2 = time.perf_counter()
    y_omp = rns4.sub_u64_io_omp(x, SUB)
    t3 = time.perf_counter()
    out["omp_t"] = t3 - t2

    t4 = time.perf_counter()
    y_auto = rns4.sub_u64_auto(x, SUB)
    t5 = time.perf_counter()
    out["auto_t"] = t5 - t4

    assert np.array_equal(y_io, y_omp)
    assert np.array_equal(y_io, y_auto)
    out["hash"] = sha16(y_auto)
    return out

def bench_mul(x):
    out = {}

    t0 = time.perf_counter()
    y_io = rns4.mul_u64_io(x, MULT)
    t1 = time.perf_counter()
    out["io_t"] = t1 - t0

    rns4.omp_set_num_threads(2)
    t2 = time.perf_counter()
    y_omp = rns4.mul_u64_io_omp(x, MULT)
    t3 = time.perf_counter()
    out["omp_t"] = t3 - t2

    t4 = time.perf_counter()
    y_auto = rns4.mul_u64_auto(x, MULT)
    t5 = time.perf_counter()
    out["auto_t"] = t5 - t4

    assert np.array_equal(y_io, y_omp)
    assert np.array_equal(y_io, y_auto)
    out["hash"] = sha16(y_auto)
    return out

def bench_fma(x):
    out = {}

    t0 = time.perf_counter()
    y_io = rns4.fma_u64_io(x, MULT, ADD)
    t1 = time.perf_counter()
    out["io_t"] = t1 - t0

    rns4.omp_set_num_threads(2)
    t2 = time.perf_counter()
    y_omp = rns4.fma_u64_io_omp(x, MULT, ADD)
    t3 = time.perf_counter()
    out["omp_t"] = t3 - t2

    t4 = time.perf_counter()
    y_auto = rns4.fma_u64_auto(x, MULT, ADD)
    t5 = time.perf_counter()
    out["auto_t"] = t5 - t4

    assert np.array_equal(y_io, y_omp)
    assert np.array_equal(y_io, y_auto)
    out["hash"] = sha16(y_auto)
    return out

def bench_rep(x):
    out = {}

    t0 = time.perf_counter()
    y_io = rns4.affine_repeat_u64_io(x, MULT, ADD, ITERATIONS)
    t1 = time.perf_counter()
    out["io_t"] = t1 - t0

    rns4.omp_set_num_threads(2)
    t2 = time.perf_counter()
    y_omp = rns4.affine_repeat_u64_io_omp(x, MULT, ADD, ITERATIONS)
    t3 = time.perf_counter()
    out["omp_t"] = t3 - t2

    t4 = time.perf_counter()
    y_auto = rns4.affine_repeat_u64_auto(x, MULT, ADD, ITERATIONS)
    t5 = time.perf_counter()
    out["auto_t"] = t5 - t4

    assert np.array_equal(y_io, y_omp)
    assert np.array_equal(y_io, y_auto)
    out["hash"] = sha16(y_auto)
    return out

OPS = [
    ("add", bench_add, False),
    ("sub", bench_sub, False),
    ("mul", bench_mul, False),
    ("fma", bench_fma, False),
    ("rep", bench_rep, True),
]

rng_master = np.random.default_rng(MASTER_SEED)
pack_summary = {}

for opname, fn, is_repeat in OPS:
    pack_summary[opname] = {}
    print("\n" + "=" * 80)
    print(f"{opname.upper()} FAMILY")
    print("=" * 80)

    for N in SIZES:
        io_vals = []
        omp_vals = []
        auto_vals = []

        warm_rng = np.random.default_rng(12345 + N + len(opname))
        warm_x = warm_rng.integers(0, M, size=min(N, 250_000), dtype=np.uint64)

        # warmup
        _ = fn(warm_x)

        print(f"N={N:,}")
        for run in range(1, RUNS + 1):
            seed = int(rng_master.integers(0, 2**31))
            rng = np.random.default_rng(seed)
            x = rng.integers(0, M, size=N, dtype=np.uint64)

            out = fn(x)

            scale = N * ITERATIONS / 1e6 if is_repeat else N / 1e6
            unit = "M ops/s" if is_repeat else "M vals/s"

            io_rate   = scale / out["io_t"]
            omp_rate  = scale / out["omp_t"]
            auto_rate = scale / out["auto_t"]

            io_vals.append(io_rate)
            omp_vals.append(omp_rate)
            auto_vals.append(auto_rate)

            print(f"  RUN {run} seed={seed}")
            print(f"    io   : {out['io_t']*1000:.2f} ms  ({io_rate:.1f} {unit})")
            print(f"    omp2 : {out['omp_t']*1000:.2f} ms  ({omp_rate:.1f} {unit})")
            print(f"    auto : {out['auto_t']*1000:.2f} ms  ({auto_rate:.1f} {unit})")
            print(f"    hash : {out['hash']}")

        pack_summary[opname][N] = {
            "io_med": med(io_vals),
            "omp_med": med(omp_vals),
            "auto_med": med(auto_vals),
            "unit": unit,
        }

        print("  SUMMARY")
        print(f"    io    median : {med(io_vals):8.1f} {unit}")
        print(f"    omp2  median : {med(omp_vals):8.1f} {unit}")
        print(f"    auto  median : {med(auto_vals):8.1f} {unit}")
        print(f"    auto / omp2  : {med(auto_vals)/med(omp_vals):.2f}x")
        print()

print("=" * 80)
print("PACK SUMMARY")
print("=" * 80)
for opname in ["add", "sub", "mul", "fma", "rep"]:
    print(f"{opname.upper()}:")
    for N in SIZES:
        row = pack_summary[opname][N]
        print(
            f"  N={N:>9,}  io={row['io_med']:>8.1f}  "
            f"omp2={row['omp_med']:>8.1f}  auto={row['auto_med']:>8.1f}  "
            f"[{row['unit']}]"
        )
    print()
