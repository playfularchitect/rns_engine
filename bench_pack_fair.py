
import sys, subprocess, textwrap

ops = ["add", "sub", "mul", "fma", "rep"]
sizes = [16000, 64000, 1000000]

driver = r'''
import sys, time, statistics, hashlib
import numpy as np
sys.path.insert(0, "/content/rns_core_test")
import _core as rns4

OP = sys.argv[1]
BACKEND = sys.argv[2]
N = int(sys.argv[3])
RUNS = 5
MASTER_SEED = 42

M = int(rns4.M)
ADD = 7
SUB = 7
MULT = 1_000_003
ITERATIONS = 1000

def med(xs): return statistics.median(xs)
def sha16(arr): return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]

rng_master = np.random.default_rng(MASTER_SEED)

def run_backend(x):
    if OP == "add":
        if BACKEND == "io":
            t0 = time.perf_counter(); y = rns4.add_u64_io(x, ADD); t1 = time.perf_counter()
        elif BACKEND == "omp2":
            rns4.omp_set_num_threads(2)
            t0 = time.perf_counter(); y = rns4.add_u64_io_omp(x, ADD); t1 = time.perf_counter()
        else:
            rns4.omp_set_num_threads(2)
            t0 = time.perf_counter(); y = rns4.add_u64_auto(x, ADD); t1 = time.perf_counter()
        scale = N / 1e6
        unit = "M vals/s"

    elif OP == "sub":
        if BACKEND == "io":
            t0 = time.perf_counter(); y = rns4.sub_u64_io(x, SUB); t1 = time.perf_counter()
        elif BACKEND == "omp2":
            rns4.omp_set_num_threads(2)
            t0 = time.perf_counter(); y = rns4.sub_u64_io_omp(x, SUB); t1 = time.perf_counter()
        else:
            rns4.omp_set_num_threads(2)
            t0 = time.perf_counter(); y = rns4.sub_u64_auto(x, SUB); t1 = time.perf_counter()
        scale = N / 1e6
        unit = "M vals/s"

    elif OP == "mul":
        if BACKEND == "io":
            t0 = time.perf_counter(); y = rns4.mul_u64_io(x, MULT); t1 = time.perf_counter()
        elif BACKEND == "omp2":
            rns4.omp_set_num_threads(2)
            t0 = time.perf_counter(); y = rns4.mul_u64_io_omp(x, MULT); t1 = time.perf_counter()
        else:
            rns4.omp_set_num_threads(2)
            t0 = time.perf_counter(); y = rns4.mul_u64_auto(x, MULT); t1 = time.perf_counter()
        scale = N / 1e6
        unit = "M vals/s"

    elif OP == "fma":
        if BACKEND == "io":
            t0 = time.perf_counter(); y = rns4.fma_u64_io(x, MULT, ADD); t1 = time.perf_counter()
        elif BACKEND == "omp2":
            rns4.omp_set_num_threads(2)
            t0 = time.perf_counter(); y = rns4.fma_u64_io_omp(x, MULT, ADD); t1 = time.perf_counter()
        else:
            rns4.omp_set_num_threads(2)
            t0 = time.perf_counter(); y = rns4.fma_u64_auto(x, MULT, ADD); t1 = time.perf_counter()
        scale = N / 1e6
        unit = "M vals/s"

    elif OP == "rep":
        if BACKEND == "io":
            t0 = time.perf_counter(); y = rns4.affine_repeat_u64_io(x, MULT, ADD, ITERATIONS); t1 = time.perf_counter()
        elif BACKEND == "omp2":
            rns4.omp_set_num_threads(2)
            t0 = time.perf_counter(); y = rns4.affine_repeat_u64_io_omp(x, MULT, ADD, ITERATIONS); t1 = time.perf_counter()
        else:
            rns4.omp_set_num_threads(2)
            t0 = time.perf_counter(); y = rns4.affine_repeat_u64_auto(x, MULT, ADD, ITERATIONS); t1 = time.perf_counter()
        scale = N * ITERATIONS / 1e6
        unit = "M ops/s"

    else:
        raise RuntimeError(OP)

    return (t1 - t0), y, scale, unit

# warmup
warm_rng = np.random.default_rng(12345 + N + len(OP) + len(BACKEND))
warm_x = warm_rng.integers(0, M, size=min(N, 250000), dtype=np.uint64)
_ = run_backend(warm_x)

rates = []
last_hash = None
for _ in range(RUNS):
    seed = int(rng_master.integers(0, 2**31))
    rng = np.random.default_rng(seed)
    x = rng.integers(0, M, size=N, dtype=np.uint64)
    dt, y, scale, unit = run_backend(x)
    rates.append(scale / dt)
    last_hash = sha16(y)

print(f"{OP} {BACKEND} N={N} median={med(rates):.1f} unit={unit} hash={last_hash}")
'''

print("=" * 88)
print("FAIR UNIFIED PACK")
print("=" * 88)

for op in ops:
    print(f"\n{op.upper()}:")
    for N in sizes:
        rows = {}
        for backend in ["io", "omp2", "auto"]:
            p = subprocess.run(
                ["python", "-c", driver, op, backend, str(N)],
                text=True,
                capture_output=True,
            )
            if p.returncode != 0:
                print("FAILED:", op, backend, N)
                print(p.stdout)
                print(p.stderr)
                raise SystemExit(1)
            rows[backend] = p.stdout.strip()
            print(" ", rows[backend])
