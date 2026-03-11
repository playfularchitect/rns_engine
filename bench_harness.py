
import sys, time, hashlib, statistics
import numpy as np

sys.path.insert(0, "/content/rns_core_test")
import _core as rns4

def sha16(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]

def mean(xs):
    return sum(xs) / len(xs)

def med(xs):
    return statistics.median(xs)

M = int(rns4.M)
MULT = 1_000_003
ADD = 7
N = 1_000_000
ITERATIONS = 1000
RUNS = 5
MASTER_SEED = 42

print("=" * 72)
print("RNS ENGINE REPRODUCIBLE BENCH HARNESS")
print("=" * 72)
print("module :", rns4.__file__)
print("AVX2   :", rns4.HAS_AVX2)
print("M      :", M)
print("has fma                 :", hasattr(rns4, "fma"))
print("has affine_repeat       :", hasattr(rns4, "affine_repeat"))
print("has fma_u64_io          :", hasattr(rns4, "fma_u64_io"))
print("has affine_repeat_u64_io:", hasattr(rns4, "affine_repeat_u64_io"))
print("has fma_u64_io_omp      :", hasattr(rns4, "fma_u64_io_omp"))
print("has affine_repeat_u64_io_omp :", hasattr(rns4, "affine_repeat_u64_io_omp"))
print("has omp_set_num_threads :", hasattr(rns4, "omp_set_num_threads"))
print("omp_num_procs           :", rns4.omp_num_procs() if hasattr(rns4, "omp_num_procs") else "missing")
print("omp_max_threads         :", rns4.omp_max_threads() if hasattr(rns4, "omp_max_threads") else "missing")
print("=" * 72)
print(f"N={N:,}  ITERATIONS={ITERATIONS:,}  RUNS={RUNS}  MASTER_SEED={MASTER_SEED}")
print("NOTE: repeat_* numbers are effective logical ops/s because affine steps are collapsed.")
print("=" * 72)

# correctness
x = np.array([0, 1, 2, 3, 123456789, 9876543210123456 % M, M - 1], dtype=np.uint64)
enc = rns4.encode(x)
dec = rns4.decode(*enc)

ref_fma = ((x.astype(object) * MULT + ADD) % M).astype(np.uint64)
ref_rep = x.astype(object)
for _ in range(ITERATIONS):
    ref_rep = (ref_rep * MULT + ADD) % M
ref_rep = np.asarray(ref_rep, dtype=np.uint64)

print("\nCORRECTNESS")
print("roundtrip ok :", np.array_equal(x, dec))

small_mult = rns4.encode(np.full(len(x), MULT, dtype=np.uint64))
small_add  = rns4.encode(np.full(len(x), ADD, dtype=np.uint64))

sep_fma = rns4.decode(*rns4.fma(*enc, *small_mult, *small_add))
print("separate fma ok:", np.array_equal(sep_fma, ref_fma))

fused_fma = rns4.fma_u64_io(x, MULT, ADD)
print("fused fma ok   :", np.array_equal(fused_fma, ref_fma))

fused_rep = rns4.affine_repeat_u64_io(x, MULT, ADD, ITERATIONS)
print("fused rep ok   :", np.array_equal(fused_rep, ref_rep))

if hasattr(rns4, "omp_set_num_threads"):
    rns4.omp_set_num_threads(1)
    omp1_fma = rns4.fma_u64_io_omp(x, MULT, ADD)
    omp1_rep = rns4.affine_repeat_u64_io_omp(x, MULT, ADD, ITERATIONS)
    print("omp(1) fma ok  :", np.array_equal(omp1_fma, ref_fma))
    print("omp(1) rep ok  :", np.array_equal(omp1_rep, ref_rep))

    rns4.omp_set_num_threads(2)
    omp2_fma = rns4.fma_u64_io_omp(x, MULT, ADD)
    omp2_rep = rns4.affine_repeat_u64_io_omp(x, MULT, ADD, ITERATIONS)
    print("omp(2) fma ok  :", np.array_equal(omp2_fma, ref_fma))
    print("omp(2) rep ok  :", np.array_equal(omp2_rep, ref_rep))

# warmup
rng = np.random.default_rng(12345)
warm_x = rng.integers(0, M, size=250_000, dtype=np.uint64)
warm_mult_arr = np.full(len(warm_x), MULT, dtype=np.uint64)
warm_add_arr  = np.full(len(warm_x), ADD, dtype=np.uint64)
warm_em = rns4.encode(warm_mult_arr)
warm_ea = rns4.encode(warm_add_arr)

_ = rns4.decode(*rns4.fma(*rns4.encode(warm_x), *warm_em, *warm_ea))
_ = rns4.fma_u64_io(warm_x, MULT, ADD)
_ = rns4.affine_repeat_u64_io(warm_x, MULT, ADD, ITERATIONS)

if hasattr(rns4, "omp_set_num_threads"):
    for t in (1, 2):
        rns4.omp_set_num_threads(t)
        _ = rns4.fma_u64_io_omp(warm_x, MULT, ADD)
        _ = rns4.affine_repeat_u64_io_omp(warm_x, MULT, ADD, ITERATIONS)

def bench_separate(x):
    em = rns4.encode(np.full(len(x), MULT, dtype=np.uint64))
    ea = rns4.encode(np.full(len(x), ADD,  dtype=np.uint64))

    t0 = time.perf_counter()
    out_fma = rns4.decode(*rns4.fma(*rns4.encode(x), *em, *ea))
    t_fma = time.perf_counter() - t0

    t1 = time.perf_counter()
    out_rep = rns4.decode(*rns4.affine_repeat(*rns4.encode(x), *em, *ea, ITERATIONS))
    t_rep = time.perf_counter() - t1
    return out_fma, t_fma, out_rep, t_rep

def bench_fused(x):
    t0 = time.perf_counter()
    out_fma = rns4.fma_u64_io(x, MULT, ADD)
    t_fma = time.perf_counter() - t0

    t1 = time.perf_counter()
    out_rep = rns4.affine_repeat_u64_io(x, MULT, ADD, ITERATIONS)
    t_rep = time.perf_counter() - t1
    return out_fma, t_fma, out_rep, t_rep

def bench_omp(x, threads):
    rns4.omp_set_num_threads(threads)

    t0 = time.perf_counter()
    out_fma = rns4.fma_u64_io_omp(x, MULT, ADD)
    t_fma = time.perf_counter() - t0

    t1 = time.perf_counter()
    out_rep = rns4.affine_repeat_u64_io_omp(x, MULT, ADD, ITERATIONS)
    t_rep = time.perf_counter() - t1
    return out_fma, t_fma, out_rep, t_rep

results = {
    "sep_fma": [], "sep_rep": [],
    "fused_fma": [], "fused_rep": [],
    "omp1_fma": [], "omp1_rep": [],
    "omp2_fma": [], "omp2_rep": [],
}

rng_master = np.random.default_rng(MASTER_SEED)

print("\n" + "=" * 72)
print("RUNS")
print("=" * 72)

for run in range(1, RUNS + 1):
    seed = int(rng_master.integers(0, 2**31))
    rng = np.random.default_rng(seed)
    x = rng.integers(0, M, size=N, dtype=np.uint64)

    sep_fma_out, sep_fma_t, sep_rep_out, sep_rep_t = bench_separate(x)
    fused_fma_out, fused_fma_t, fused_rep_out, fused_rep_t = bench_fused(x)

    assert np.array_equal(sep_fma_out, fused_fma_out)
    assert np.array_equal(sep_rep_out, fused_rep_out)

    results["sep_fma"].append(N / sep_fma_t / 1e6)
    results["sep_rep"].append(N * ITERATIONS / sep_rep_t / 1e6)
    results["fused_fma"].append(N / fused_fma_t / 1e6)
    results["fused_rep"].append(N * ITERATIONS / fused_rep_t / 1e6)

    print(f"RUN {run}  seed={seed}")
    print(f"  sep fma      : {sep_fma_t*1000:.2f} ms  ({results['sep_fma'][-1]:.1f} M vals/s)")
    print(f"  fused fma    : {fused_fma_t*1000:.2f} ms  ({results['fused_fma'][-1]:.1f} M vals/s)")
    print(f"  sep rep 1000 : {sep_rep_t*1000:.2f} ms  ({results['sep_rep'][-1]:.1f} M ops/s)")
    print(f"  fused rep1000: {fused_rep_t*1000:.2f} ms  ({results['fused_rep'][-1]:.1f} M ops/s)")
    print(f"  hashes       : fma={sha16(fused_fma_out)} rep={sha16(fused_rep_out)}")

    if hasattr(rns4, "omp_set_num_threads"):
        omp1_fma_out, omp1_fma_t, omp1_rep_out, omp1_rep_t = bench_omp(x, 1)
        omp2_fma_out, omp2_fma_t, omp2_rep_out, omp2_rep_t = bench_omp(x, 2)

        assert np.array_equal(fused_fma_out, omp1_fma_out)
        assert np.array_equal(fused_rep_out, omp1_rep_out)
        assert np.array_equal(fused_fma_out, omp2_fma_out)
        assert np.array_equal(fused_rep_out, omp2_rep_out)

        results["omp1_fma"].append(N / omp1_fma_t / 1e6)
        results["omp1_rep"].append(N * ITERATIONS / omp1_rep_t / 1e6)
        results["omp2_fma"].append(N / omp2_fma_t / 1e6)
        results["omp2_rep"].append(N * ITERATIONS / omp2_rep_t / 1e6)

        print(f"  omp1 fma     : {omp1_fma_t*1000:.2f} ms  ({results['omp1_fma'][-1]:.1f} M vals/s)")
        print(f"  omp1 rep1000 : {omp1_rep_t*1000:.2f} ms  ({results['omp1_rep'][-1]:.1f} M ops/s)")
        print(f"  omp2 fma     : {omp2_fma_t*1000:.2f} ms  ({results['omp2_fma'][-1]:.1f} M vals/s)")
        print(f"  omp2 rep1000 : {omp2_rep_t*1000:.2f} ms  ({results['omp2_rep'][-1]:.1f} M ops/s)")
    print()

print("=" * 72)
print("SUMMARY")
print("=" * 72)

for key in [
    "sep_fma", "fused_fma", "omp1_fma", "omp2_fma",
    "sep_rep", "fused_rep", "omp1_rep", "omp2_rep"
]:
    if results[key]:
        unit = "M vals/s" if "fma" in key else "M ops/s"
        print(f"{key:<10} mean={mean(results[key]):8.1f}  median={med(results[key]):8.1f}  [{unit}]")

print("\nSPEEDUPS (using medians)")
if results["fused_fma"]:
    print(f"fused_fma / sep_fma  : {med(results['fused_fma']) / med(results['sep_fma']):.2f}x")
if results["omp1_fma"]:
    print(f"omp1_fma / fused_fma : {med(results['omp1_fma']) / med(results['fused_fma']):.2f}x")
if results["omp2_fma"]:
    print(f"omp2_fma / fused_fma : {med(results['omp2_fma']) / med(results['fused_fma']):.2f}x")
    print(f"omp2_fma / omp1_fma  : {med(results['omp2_fma']) / med(results['omp1_fma']):.2f}x")

if results["fused_rep"]:
    print(f"fused_rep / sep_rep  : {med(results['fused_rep']) / med(results['sep_rep']):.2f}x")
if results["omp1_rep"]:
    print(f"omp1_rep / fused_rep : {med(results['omp1_rep']) / med(results['fused_rep']):.2f}x")
if results["omp2_rep"]:
    print(f"omp2_rep / fused_rep : {med(results['omp2_rep']) / med(results['fused_rep']):.2f}x")
    print(f"omp2_rep / omp1_rep  : {med(results['omp2_rep']) / med(results['omp1_rep']):.2f}x")
