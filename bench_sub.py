
import sys, time, hashlib, statistics
import numpy as np

sys.path.insert(0, "/content/rns_core_test")
import _core as rns4

def sha16(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]

def mean(xs): return sum(xs) / len(xs)
def med(xs): return statistics.median(xs)

M = int(rns4.M)
SUB = 7
RUNS = 5
MASTER_SEED = 42
sizes = [1_000, 4_000, 16_000, 64_000, 256_000, 1_000_000]

print("=" * 72)
print("SUB_U64 CANONICAL HARNESS")
print("=" * 72)
print("module :", rns4.__file__)
print("AVX2   :", rns4.HAS_AVX2)
print("M      :", M)
print("has sub_u64_io     :", hasattr(rns4, "sub_u64_io"))
print("has sub_u64_io_omp :", hasattr(rns4, "sub_u64_io_omp"))
print("omp_num_procs      :", rns4.omp_num_procs() if hasattr(rns4, "omp_num_procs") else "missing")
print("omp_max_threads    :", rns4.omp_max_threads() if hasattr(rns4, "omp_max_threads") else "missing")
print("=" * 72)

x = np.array([0, 1, 2, 3, 123456789, 9876543210123456 % M, M - 1], dtype=np.uint64)
ref = ((x.astype(object) - SUB) % M).astype(np.uint64)

rns4.omp_set_num_threads(2)

y_io = rns4.sub_u64_io(x, SUB)
rns4.omp_set_num_threads(1)
y_omp1 = rns4.sub_u64_io_omp(x, SUB)
rns4.omp_set_num_threads(2)
y_omp2 = rns4.sub_u64_io_omp(x, SUB)

print("CORRECTNESS")
print("io ok   :", np.array_equal(y_io, ref))
print("omp1 ok :", np.array_equal(y_omp1, ref))
print("omp2 ok :", np.array_equal(y_omp2, ref))
print("all eq  :", np.array_equal(y_io, y_omp1) and np.array_equal(y_io, y_omp2))
print()

rng_master = np.random.default_rng(MASTER_SEED)

for N in sizes:
    io_m = []
    omp1_m = []
    omp2_m = []

    warm_rng = np.random.default_rng(12345 + N)
    warm_x = warm_rng.integers(0, M, size=min(N, 250_000), dtype=np.uint64)
    _ = rns4.sub_u64_io(warm_x, SUB)
    rns4.omp_set_num_threads(1)
    _ = rns4.sub_u64_io_omp(warm_x, SUB)
    rns4.omp_set_num_threads(2)
    _ = rns4.sub_u64_io_omp(warm_x, SUB)

    print("=" * 72)
    print(f"N={N:,}")
    print("=" * 72)

    for run in range(1, RUNS + 1):
        seed = int(rng_master.integers(0, 2**31))
        rng = np.random.default_rng(seed)
        x = rng.integers(0, M, size=N, dtype=np.uint64)

        t0 = time.perf_counter()
        y_io = rns4.sub_u64_io(x, SUB)
        t1 = time.perf_counter()

        rns4.omp_set_num_threads(1)
        t2 = time.perf_counter()
        y_omp1 = rns4.sub_u64_io_omp(x, SUB)
        t3 = time.perf_counter()

        rns4.omp_set_num_threads(2)
        t4 = time.perf_counter()
        y_omp2 = rns4.sub_u64_io_omp(x, SUB)
        t5 = time.perf_counter()

        assert np.array_equal(y_io, y_omp1)
        assert np.array_equal(y_io, y_omp2)

        mio   = N / (t1 - t0) / 1e6
        momp1 = N / (t3 - t2) / 1e6
        momp2 = N / (t5 - t4) / 1e6

        io_m.append(mio)
        omp1_m.append(momp1)
        omp2_m.append(momp2)

        print(f"RUN {run} seed={seed}")
        print(f"  sub io   : {(t1-t0)*1000:.2f} ms  ({mio:.1f} M vals/s)")
        print(f"  sub omp1 : {(t3-t2)*1000:.2f} ms  ({momp1:.1f} M vals/s)")
        print(f"  sub omp2 : {(t5-t4)*1000:.2f} ms  ({momp2:.1f} M vals/s)")
        print(f"  hash     : {sha16(y_omp2)}")
        print()

    print("SUMMARY")
    print(f"  io    mean={mean(io_m):8.1f} median={med(io_m):8.1f}")
    print(f"  omp1  mean={mean(omp1_m):8.1f} median={med(omp1_m):8.1f}")
    print(f"  omp2  mean={mean(omp2_m):8.1f} median={med(omp2_m):8.1f}")
    print(f"  omp2 / io   : {med(omp2_m)/med(io_m):.2f}x")
    print(f"  omp2 / omp1 : {med(omp2_m)/med(omp1_m):.2f}x")
    print()
