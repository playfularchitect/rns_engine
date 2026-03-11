/*
 * rns_engine/_core.cpp  --  v0.4.0rc1
 *
 * 4-rail RNS exact integer arithmetic.
 * Best-known speed baseline + native affine_repeat().
 *
 * Key idea:
 *   - keep the faster v0.4.1-style core
 *   - add affine_repeat(...) so repeated x = x*m + k runs fully inside C++
 *   - avoid Python loop overhead for long pipelines
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdint.h>
#include <stdexcept>
#include <string>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads() { return 1; }
static inline void omp_set_num_threads(int) {}
static inline int omp_get_num_procs() { return 1; }
#endif

namespace py = pybind11;

using arr16 = py::array_t<uint16_t, py::array::c_style | py::array::forcecast>;
using arr32 = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;
using arr64 = py::array_t<uint64_t, py::array::c_style | py::array::forcecast>;

// -- Constants --------------------------------------------------------------
static constexpr uint32_t M0      = 127;
static constexpr uint32_t M1      = 8191;
static constexpr uint32_t M2      = 65536;
static constexpr uint32_t M3      = 524287;      // 2^19 - 1
static constexpr uint32_t M3_MASK = 0x7FFFFu;
static constexpr uint64_t BM      = (uint64_t)M0 * M1 * M2 * M3;
static constexpr uint32_t INV01   = 129;
static constexpr uint32_t INV012  = 24705;
static constexpr uint32_t INV0123 = 285373;

// -- Validation helpers -----------------------------------------------------
static int64_t require_1d_len(const py::array& a, const char* name) {
    if (a.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1D array");
    }
    return (int64_t)a.shape(0);
}

static void require_same_len(const py::array& a, const char* aname,
                             const py::array& b, const char* bname) {
    int64_t na = require_1d_len(a, aname);
    int64_t nb = require_1d_len(b, bname);
    if (na != nb) {
        throw std::invalid_argument(
            std::string("array length mismatch: ") + aname + " has length " +
            std::to_string(na) + ", " + bname + " has length " +
            std::to_string(nb));
    }
}

// -- Scalar helpers ---------------------------------------------------------
static inline uint16_t r127s(uint32_t x) {
    x = (x & 0x7F) + (x >> 7);
    x = (x & 0x7F) + (x >> 7);
    return x >= 127 ? x - 127 : (uint16_t)x;
}

static inline uint16_t r8191s(uint32_t x) {
    x = (x & 0x1FFF) + (x >> 13);
    x = (x & 0x1FFF) + (x >> 13);
    return x >= 8191 ? x - 8191 : (uint16_t)x;
}

static inline uint32_t add524287s(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    return s >= M3 ? (s - M3) : s;
}

static inline uint32_t sub524287s(uint32_t a, uint32_t b) {
    return a >= b ? (a - b) : (a + M3 - b);
}

static inline uint32_t mul524287s(uint32_t a, uint32_t b) {
    uint64_t x = (uint64_t)a * b;
    x = (x & M3_MASK) + (x >> 19);
    x = (x & M3_MASK) + (x >> 19);
    uint32_t r = (uint32_t)x;
    return r >= M3 ? (r - M3) : r;
}

static inline uint32_t fma524287s(uint32_t a, uint32_t b, uint32_t c) {
    uint64_t x = (uint64_t)a * b + c;
    x = (x & M3_MASK) + (x >> 19);
    x = (x & M3_MASK) + (x >> 19);
    x = (x & M3_MASK) + (x >> 19);
    uint32_t r = (uint32_t)x;
    return r >= M3 ? (r - M3) : r;
}

static inline uint32_t mod524287_u64(uint64_t x) {
    x = (x & M3_MASK) + (x >> 19);
    x = (x & M3_MASK) + (x >> 19);
    x = (x & M3_MASK) + (x >> 19);
    uint32_t r = (uint32_t)x;
    return r >= M3 ? (r - M3) : r;
}

static int64_t egcd(int64_t a, int64_t b, int64_t* x, int64_t* y) {
    if (!a) {
        *x = 0;
        *y = 1;
        return b;
    }
    int64_t x1, y1;
    int64_t g = egcd(b % a, a, &x1, &y1);
    *x = y1 - (b / a) * x1;
    *y = x1;
    return g;
}

static uint32_t inv_s(int64_t a, int64_t m) {
    a = ((a % m) + m) % m;
    if (!a) return 0;
    int64_t x, y;
    if (egcd(a, m, &x, &y) != 1) return 0;
    return (uint32_t)(((x % m) + m) % m);
}

static inline uint64_t garner4(uint16_t r0, uint16_t r1, uint16_t r2, uint32_t r3) {
    uint32_t t0 = r0;

    uint32_t t1 = (uint32_t)(
        ((int64_t)r1 - (int64_t)(t0 % M1) + M1) % M1
        * (uint64_t)INV01 % M1);

    uint64_t base01 = t0 + (uint64_t)t1 * M0;

    uint32_t base01_mod_m2 = (uint32_t)base01 & 0xFFFFu;
    uint32_t d2 = (uint32_t)(((uint64_t)r2 + M2 - base01_mod_m2) & 0xFFFFu);
    uint32_t t2 = (uint32_t)(((uint64_t)d2 * INV012) & 0xFFFFu);

    uint64_t base012 = base01 + (uint64_t)t2 * (uint64_t)M0 * M1;

    uint32_t d3 = sub524287s(r3, mod524287_u64(base012));
    uint32_t t3 = mul524287s(d3, INV0123);

    return base012 + (uint64_t)t3 * (uint64_t)M0 * M1 * M2;
}


static void require_divisors_invertible(const arr16& b0,
                                        const arr16& b1,
                                        const arr16& b2,
                                        const arr32& b3) {
    require_same_len(b0, "b0", b1, "b1");
    require_same_len(b0, "b0", b2, "b2");
    require_same_len(b0, "b0", b3, "b3");

    auto rb0 = b0.unchecked<1>();
    auto rb1 = b1.unchecked<1>();
    auto rb2 = b2.unchecked<1>();
    auto rb3 = b3.unchecked<1>();
    int64_t n = (int64_t)b0.shape(0);

    for (int64_t i = 0; i < n; ++i) {
        if (r127s(rb0(i)) == 0) {
            throw std::invalid_argument(
                "division error: b0 contains a non-invertible residue mod 127 at index " +
                std::to_string(i));
        }
        if (r8191s(rb1(i)) == 0) {
            throw std::invalid_argument(
                "division error: b1 contains a non-invertible residue mod 8191 at index " +
                std::to_string(i));
        }
        if ((rb2(i) & 1u) == 0u) {
            throw std::invalid_argument(
                "division error: b2 contains a non-invertible residue mod 65536 at index " +
                std::to_string(i));
        }
        if (rb3(i) % M3 == 0u) {
            throw std::invalid_argument(
                "division error: b3 contains a non-invertible residue mod 524287 at index " +
                std::to_string(i));
        }
    }
}

// -- Scalar kernel ----------------------------------------------------------
static void kernel_scalar(
    const uint16_t* a0, const uint16_t* a1, const uint16_t* a2, const uint32_t* a3,
    const uint16_t* b0, const uint16_t* b1, const uint16_t* b2, const uint32_t* b3,
    uint16_t* r0, uint16_t* r1, uint16_t* r2, uint32_t* r3,
    int64_t n, int op,
    const uint16_t* c0 = nullptr,
    const uint16_t* c1 = nullptr,
    const uint16_t* c2 = nullptr,
    const uint32_t* c3 = nullptr) {

    for (int64_t i = 0; i < n; i++) {
        switch (op) {
            case 0:
                r0[i] = r127s(a0[i] + b0[i]);
                r1[i] = r8191s(a1[i] + b1[i]);
                r2[i] = (uint16_t)((a2[i] + b2[i]) & 0xFFFF);
                r3[i] = add524287s(a3[i], b3[i]);
                break;

            case 1:
                r0[i] = r127s((uint32_t)a0[i] * b0[i]);
                r1[i] = r8191s((uint32_t)a1[i] * b1[i]);
                r2[i] = (uint16_t)((uint32_t)a2[i] * b2[i]);
                r3[i] = mul524287s(a3[i], b3[i]);
                break;

            case 2:
                r0[i] = r127s(M0 + a0[i] - r127s(b0[i]));
                r1[i] = r8191s(M1 + a1[i] - r8191s(b1[i]));
                r2[i] = (uint16_t)((M2 + a2[i] - (b2[i] % M2)) & 0xFFFF);
                r3[i] = sub524287s(a3[i], b3[i] % M3);
                break;

            case 3:
                r0[i] = r127s((uint32_t)a0[i] * inv_s(b0[i], M0));
                r1[i] = r8191s((uint32_t)a1[i] * inv_s(b1[i], M1));
                r2[i] = (uint16_t)(((uint32_t)a2[i] * inv_s(b2[i], M2)) & 0xFFFF);
                r3[i] = mul524287s(a3[i], inv_s(b3[i], M3));
                break;

            case 4:
                r0[i] = r127s((uint32_t)a0[i] * b0[i] + c0[i]);
                r1[i] = r8191s((uint32_t)a1[i] * b1[i] + c1[i]);
                r2[i] = (uint16_t)((uint32_t)a2[i] * b2[i] + c2[i]);
                r3[i] = fma524287s(a3[i], b3[i], c3[i]);
                break;

            default:
                throw std::invalid_argument("invalid opcode");
        }
    }
}

// -- AVX2 kernel ------------------------------------------------------------
#if defined(__AVX2__) || defined(FORCE_AVX2)
#include <immintrin.h>
#define HAVE_AVX2 1
#define L 16

using vec = __m256i;
static inline vec V1(int x)        { return _mm256_set1_epi16((short)x); }
static inline vec Va(vec a, vec b) { return _mm256_add_epi16(a, b); }
static inline vec Vs(vec a, vec b) { return _mm256_sub_epi16(a, b); }
static inline vec Vm(vec a, vec b) { return _mm256_mullo_epi16(a, b); }
static inline vec Vn(vec a, vec b) { return _mm256_and_si256(a, b); }
static inline vec Vh(vec a, int s) { return _mm256_srli_epi16(a, s); }
static inline vec Ve(vec a, vec b) { return _mm256_cmpeq_epi16(a, b); }

static inline vec Vload(const uint16_t* p) {
    return _mm256_loadu_si256((const vec*)p);
}

static inline void Vstore(uint16_t* p, vec v) {
    _mm256_storeu_si256((vec*)p, v);
}

static inline vec Vload32u(const uint32_t* p) {
    return _mm256_loadu_si256((const vec*)p);
}

static inline vec fold_m3_twice(vec x) {
    const vec mask = _mm256_set1_epi64x((long long)M3_MASK);
    x = _mm256_add_epi64(_mm256_and_si256(x, mask), _mm256_srli_epi64(x, 19));
    x = _mm256_add_epi64(_mm256_and_si256(x, mask), _mm256_srli_epi64(x, 19));
    return x;
}

static inline vec fold_m3_thrice(vec x) {
    const vec mask = _mm256_set1_epi64x((long long)M3_MASK);
    x = _mm256_add_epi64(_mm256_and_si256(x, mask), _mm256_srli_epi64(x, 19));
    x = _mm256_add_epi64(_mm256_and_si256(x, mask), _mm256_srli_epi64(x, 19));
    x = _mm256_add_epi64(_mm256_and_si256(x, mask), _mm256_srli_epi64(x, 19));
    return x;
}

static inline void mul524287_const8_store(const uint32_t* x, uint32_t m, uint32_t* r) {
    vec vx = Vload32u(x);
    vec vm = _mm256_set1_epi32((int)m);

    vec even = _mm256_mul_epu32(vx, vm);
    vec odd  = _mm256_mul_epu32(_mm256_srli_epi64(vx, 32), _mm256_srli_epi64(vm, 32));

    even = fold_m3_twice(even);
    odd  = fold_m3_twice(odd);

    alignas(32) uint64_t e[4], o[4];
    _mm256_store_si256((__m256i*)e, even);
    _mm256_store_si256((__m256i*)o, odd);

    r[0] = (uint32_t)e[0]; if (r[0] >= M3) r[0] -= M3;
    r[1] = (uint32_t)o[0]; if (r[1] >= M3) r[1] -= M3;
    r[2] = (uint32_t)e[1]; if (r[2] >= M3) r[2] -= M3;
    r[3] = (uint32_t)o[1]; if (r[3] >= M3) r[3] -= M3;
    r[4] = (uint32_t)e[2]; if (r[4] >= M3) r[4] -= M3;
    r[5] = (uint32_t)o[2]; if (r[5] >= M3) r[5] -= M3;
    r[6] = (uint32_t)e[3]; if (r[6] >= M3) r[6] -= M3;
    r[7] = (uint32_t)o[3]; if (r[7] >= M3) r[7] -= M3;
}

static inline void fma524287_const8_store(const uint32_t* x, uint32_t m, uint32_t k, uint32_t* r) {
    vec vx = Vload32u(x);
    vec vm = _mm256_set1_epi32((int)m);
    vec vk = _mm256_set1_epi32((int)k);

    vec even = _mm256_mul_epu32(vx, vm);
    vec odd  = _mm256_mul_epu32(_mm256_srli_epi64(vx, 32), _mm256_srli_epi64(vm, 32));

    const vec low32mask = _mm256_set1_epi64x(0xFFFFFFFFULL);
    even = _mm256_add_epi64(even, _mm256_and_si256(vk, low32mask));
    odd  = _mm256_add_epi64(odd,  _mm256_srli_epi64(vk, 32));

    even = fold_m3_thrice(even);
    odd  = fold_m3_thrice(odd);

    alignas(32) uint64_t e[4], o[4];
    _mm256_store_si256((__m256i*)e, even);
    _mm256_store_si256((__m256i*)o, odd);

    r[0] = (uint32_t)e[0]; if (r[0] >= M3) r[0] -= M3;
    r[1] = (uint32_t)o[0]; if (r[1] >= M3) r[1] -= M3;
    r[2] = (uint32_t)e[1]; if (r[2] >= M3) r[2] -= M3;
    r[3] = (uint32_t)o[1]; if (r[3] >= M3) r[3] -= M3;
    r[4] = (uint32_t)e[2]; if (r[4] >= M3) r[4] -= M3;
    r[5] = (uint32_t)o[2]; if (r[5] >= M3) r[5] -= M3;
    r[6] = (uint32_t)e[3]; if (r[6] >= M3) r[6] -= M3;
    r[7] = (uint32_t)o[3]; if (r[7] >= M3) r[7] -= M3;
}

static inline vec r127v(vec x) {
    vec t = Va(Vn(x, V1(0x7F)), Vh(x, 7));
    t = Va(Vn(t, V1(0x7F)), Vh(t, 7));
    return Vs(t, Vn(V1(127), Ve(t, V1(127))));
}

static inline vec r8191v(vec x) {
    vec t = Va(Vn(x, V1(0x1FFF)), Vh(x, 13));
    t = Va(Vn(t, V1(0x1FFF)), Vh(t, 13));
    return Vs(t, Vn(V1(8191), Ve(t, V1(8191))));
}

static inline vec mul8191v(vec a, vec b) {
    const __m256i mk = _mm256_set1_epi32(0x1FFF);

    auto fold = [&](__m256i x) -> __m256i {
        x = _mm256_add_epi32(_mm256_and_si256(x, mk), _mm256_srli_epi32(x, 13));
        x = _mm256_add_epi32(_mm256_and_si256(x, mk), _mm256_srli_epi32(x, 13));
        return _mm256_sub_epi32(
            x,
            _mm256_and_si256(
                _mm256_cmpeq_epi32(x, _mm256_set1_epi32(8191)),
                _mm256_set1_epi32(8191)));
    };

    __m256i lo = _mm256_mullo_epi32(
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(a)),
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b)));

    __m256i hi = _mm256_mullo_epi32(
        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(a, 1)),
        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(b, 1)));

    return _mm256_permute4x64_epi64(
        _mm256_packus_epi32(fold(lo), fold(hi)), 0xD8);
}

static inline vec mul127v(vec a, vec b) {
    return r127v(Vm(a, b));
}

static void kernel_avx2(
    const uint16_t* a0, const uint16_t* a1, const uint16_t* a2, const uint32_t* a3,
    const uint16_t* b0, const uint16_t* b1, const uint16_t* b2, const uint32_t* b3,
    uint16_t* r0, uint16_t* r1, uint16_t* r2, uint32_t* r3,
    int64_t n, int op,
    const uint16_t* c0 = nullptr,
    const uint16_t* c1 = nullptr,
    const uint16_t* c2 = nullptr,
    const uint32_t* c3 = nullptr) {

    if (op == 3) {
        kernel_scalar(a0, a1, a2, a3, b0, b1, b2, b3, r0, r1, r2, r3, n, op);
        return;
    }

    const int64_t full = (n / L) * L;

    for (int64_t i = 0; i < full; i += L) {
        vec va0 = Vload(a0 + i), vb0 = Vload(b0 + i);
        vec va1 = Vload(a1 + i), vb1 = Vload(b1 + i);
        vec va2 = Vload(a2 + i), vb2 = Vload(b2 + i);
        vec vr0, vr1, vr2;

        switch (op) {
            case 0:
                vr0 = r127v(Va(va0, vb0));
                vr1 = r8191v(Va(va1, vb1));
                vr2 = Va(va2, vb2);
                for (int j = 0; j < L; j += 4) {
                    r3[i + j + 0] = add524287s(a3[i + j + 0], b3[i + j + 0]);
                    r3[i + j + 1] = add524287s(a3[i + j + 1], b3[i + j + 1]);
                    r3[i + j + 2] = add524287s(a3[i + j + 2], b3[i + j + 2]);
                    r3[i + j + 3] = add524287s(a3[i + j + 3], b3[i + j + 3]);
                }
                break;

            case 1:
                vr0 = mul127v(va0, vb0);
                vr1 = mul8191v(va1, vb1);
                vr2 = Vm(va2, vb2);
                for (int j = 0; j < L; j += 4) {
                    r3[i + j + 0] = mul524287s(a3[i + j + 0], b3[i + j + 0]);
                    r3[i + j + 1] = mul524287s(a3[i + j + 1], b3[i + j + 1]);
                    r3[i + j + 2] = mul524287s(a3[i + j + 2], b3[i + j + 2]);
                    r3[i + j + 3] = mul524287s(a3[i + j + 3], b3[i + j + 3]);
                }
                break;

            case 2:
                vr0 = r127v(Va(va0, Vs(V1(127), r127v(vb0))));
                vr1 = r8191v(Va(va1, Vs(V1(8191), r8191v(vb1))));
                vr2 = Vs(va2, vb2);
                for (int j = 0; j < L; j += 4) {
                    r3[i + j + 0] = sub524287s(a3[i + j + 0], b3[i + j + 0] % M3);
                    r3[i + j + 1] = sub524287s(a3[i + j + 1], b3[i + j + 1] % M3);
                    r3[i + j + 2] = sub524287s(a3[i + j + 2], b3[i + j + 2] % M3);
                    r3[i + j + 3] = sub524287s(a3[i + j + 3], b3[i + j + 3] % M3);
                }
                break;

            case 4: {
                vec vc0 = Vload(c0 + i);
                vec vc1 = Vload(c1 + i);
                vec vc2 = Vload(c2 + i);
                vr0 = r127v(Va(mul127v(va0, vb0), vc0));
                vr1 = r8191v(Va(mul8191v(va1, vb1), vc1));
                vr2 = Va(Vm(va2, vb2), vc2);
                for (int j = 0; j < L; j += 4) {
                    r3[i + j + 0] = fma524287s(a3[i + j + 0], b3[i + j + 0], c3[i + j + 0]);
                    r3[i + j + 1] = fma524287s(a3[i + j + 1], b3[i + j + 1], c3[i + j + 1]);
                    r3[i + j + 2] = fma524287s(a3[i + j + 2], b3[i + j + 2], c3[i + j + 2]);
                    r3[i + j + 3] = fma524287s(a3[i + j + 3], b3[i + j + 3], c3[i + j + 3]);
                }
                break;
            }

            default:
                throw std::invalid_argument("invalid opcode");
        }

        Vstore(r0 + i, vr0);
        Vstore(r1 + i, vr1);
        Vstore(r2 + i, vr2);
    }

    if (full < n) {
        kernel_scalar(
            a0 + full, a1 + full, a2 + full, a3 + full,
            b0 + full, b1 + full, b2 + full, b3 + full,
            r0 + full, r1 + full, r2 + full, r3 + full,
            n - full, op,
            c0 ? c0 + full : nullptr,
            c1 ? c1 + full : nullptr,
            c2 ? c2 + full : nullptr,
            c3 ? c3 + full : nullptr);
    }
}

#else
#define HAVE_AVX2 0
#define L 16
#endif

// -- Dispatch ---------------------------------------------------------------
static void kernel(
    const uint16_t* a0, const uint16_t* a1, const uint16_t* a2, const uint32_t* a3,
    const uint16_t* b0, const uint16_t* b1, const uint16_t* b2, const uint32_t* b3,
    uint16_t* r0, uint16_t* r1, uint16_t* r2, uint32_t* r3,
    int64_t n, int op,
    const uint16_t* c0 = nullptr,
    const uint16_t* c1 = nullptr,
    const uint16_t* c2 = nullptr,
    const uint32_t* c3 = nullptr) {
#if HAVE_AVX2
    kernel_avx2(a0, a1, a2, a3, b0, b1, b2, b3, r0, r1, r2, r3, n, op, c0, c1, c2, c3);
#else
    kernel_scalar(a0, a1, a2, a3, b0, b1, b2, b3, r0, r1, r2, r3, n, op, c0, c1, c2, c3);
#endif
}

// -- Native repeated affine loop --------------------------------------------
py::tuple py_affine_repeat(const arr16& x0, const arr16& x1, const arr16& x2, const arr32& x3,
                           const arr16& m0, const arr16& m1, const arr16& m2, const arr32& m3,
                           const arr16& k0, const arr16& k1, const arr16& k2, const arr32& k3,
                           int64_t iterations) {
    if (iterations < 0) {
        throw std::invalid_argument("iterations must be >= 0");
    }

    int64_t n = require_1d_len(x0, "x0");
    require_same_len(x0, "x0", x1, "x1");
    require_same_len(x0, "x0", x2, "x2");
    require_same_len(x0, "x0", x3, "x3");
    require_same_len(x0, "x0", m0, "m0");
    require_same_len(x0, "x0", m1, "m1");
    require_same_len(x0, "x0", m2, "m2");
    require_same_len(x0, "x0", m3, "m3");
    require_same_len(x0, "x0", k0, "k0");
    require_same_len(x0, "x0", k1, "k1");
    require_same_len(x0, "x0", k2, "k2");
    require_same_len(x0, "x0", k3, "k3");

    arr16 cur0({n}), cur1({n}), cur2({n});
    arr32 cur3({n});
    arr16 tmp0({n}), tmp1({n}), tmp2({n});
    arr32 tmp3({n});

    std::memcpy(cur0.mutable_data(), x0.data(), (size_t)n * sizeof(uint16_t));
    std::memcpy(cur1.mutable_data(), x1.data(), (size_t)n * sizeof(uint16_t));
    std::memcpy(cur2.mutable_data(), x2.data(), (size_t)n * sizeof(uint16_t));
    std::memcpy(cur3.mutable_data(), x3.data(), (size_t)n * sizeof(uint32_t));

    if (iterations == 0) {
        return py::make_tuple(cur0, cur1, cur2, cur3);
    }

    bool flip = false;
    for (int64_t it = 0; it < iterations; ++it) {
        if (!flip) {
            kernel(cur0.data(), cur1.data(), cur2.data(), cur3.data(),
                   m0.data(), m1.data(), m2.data(), m3.data(),
                   tmp0.mutable_data(), tmp1.mutable_data(), tmp2.mutable_data(), tmp3.mutable_data(),
                   n, 4,
                   k0.data(), k1.data(), k2.data(), k3.data());
        } else {
            kernel(tmp0.data(), tmp1.data(), tmp2.data(), tmp3.data(),
                   m0.data(), m1.data(), m2.data(), m3.data(),
                   cur0.mutable_data(), cur1.mutable_data(), cur2.mutable_data(), cur3.mutable_data(),
                   n, 4,
                   k0.data(), k1.data(), k2.data(), k3.data());
        }
        flip = !flip;
    }

    if (flip) {
        return py::make_tuple(tmp0, tmp1, tmp2, tmp3);
    }
    return py::make_tuple(cur0, cur1, cur2, cur3);
}


py::tuple py_affine_repeat_u64(const arr16& x0, const arr16& x1, const arr16& x2, const arr32& x3,
                               uint64_t multiplier, uint64_t addend,
                               int64_t iterations) {
    if (iterations < 0) {
        throw std::invalid_argument("iterations must be >= 0");
    }

    int64_t n = require_1d_len(x0, "x0");
    require_same_len(x0, "x0", x1, "x1");
    require_same_len(x0, "x0", x2, "x2");
    require_same_len(x0, "x0", x3, "x3");

    const uint16_t m0 = (uint16_t)(multiplier % M0);
    const uint16_t m1 = (uint16_t)(multiplier % M1);
    const uint16_t m2 = (uint16_t)(multiplier % M2);
    const uint32_t m3 = (uint32_t)(multiplier % M3);

    const uint16_t k0 = (uint16_t)(addend % M0);
    const uint16_t k1 = (uint16_t)(addend % M1);
    const uint16_t k2 = (uint16_t)(addend % M2);
    const uint32_t k3 = (uint32_t)(addend % M3);

    arr16 cur0({n}), cur1({n}), cur2({n});
    arr32 cur3({n});
    arr16 tmp0({n}), tmp1({n}), tmp2({n});
    arr32 tmp3({n});

    std::memcpy(cur0.mutable_data(), x0.data(), (size_t)n * sizeof(uint16_t));
    std::memcpy(cur1.mutable_data(), x1.data(), (size_t)n * sizeof(uint16_t));
    std::memcpy(cur2.mutable_data(), x2.data(), (size_t)n * sizeof(uint16_t));
    std::memcpy(cur3.mutable_data(), x3.data(), (size_t)n * sizeof(uint32_t));

    if (iterations == 0) {
        return py::make_tuple(cur0, cur1, cur2, cur3);
    }

    bool flip = false;

#if HAVE_AVX2
    const vec vm0v = V1((int)m0);
    const vec vm1v = V1((int)m1);
    const vec vm2v = V1((int)m2);
    const vec vk0v = V1((int)k0);
    const vec vk1v = V1((int)k1);
    const vec vk2v = V1((int)k2);

    const int64_t full = (n / L) * L;

    for (int64_t it = 0; it < iterations; ++it) {
        const uint16_t* in0 = flip ? tmp0.data() : cur0.data();
        const uint16_t* in1 = flip ? tmp1.data() : cur1.data();
        const uint16_t* in2 = flip ? tmp2.data() : cur2.data();
        const uint32_t* in3 = flip ? tmp3.data() : cur3.data();

        uint16_t* out0 = flip ? cur0.mutable_data() : tmp0.mutable_data();
        uint16_t* out1 = flip ? cur1.mutable_data() : tmp1.mutable_data();
        uint16_t* out2 = flip ? cur2.mutable_data() : tmp2.mutable_data();
        uint32_t* out3 = flip ? cur3.mutable_data() : tmp3.mutable_data();

        for (int64_t i = 0; i < full; i += L) {
            vec vx0 = Vload(in0 + i);
            vec vx1 = Vload(in1 + i);
            vec vx2 = Vload(in2 + i);

            vec vr0 = r127v(Va(mul127v(vx0, vm0v), vk0v));
            vec vr1 = r8191v(Va(mul8191v(vx1, vm1v), vk1v));
            vec vr2 = Va(Vm(vx2, vm2v), vk2v);

            Vstore(out0 + i, vr0);
            Vstore(out1 + i, vr1);
            Vstore(out2 + i, vr2);

            fma524287_const8_store(in3 + i,     m3, k3, out3 + i);
            fma524287_const8_store(in3 + i + 8, m3, k3, out3 + i + 8);
        }

        for (int64_t i = full; i < n; ++i) {
            out0[i] = r127s((uint32_t)in0[i] * m0 + k0);
            out1[i] = r8191s((uint32_t)in1[i] * m1 + k1);
            out2[i] = (uint16_t)((uint32_t)in2[i] * m2 + k2);
            out3[i] = fma524287s(in3[i], m3, k3);
        }

        flip = !flip;
    }
#else
    for (int64_t it = 0; it < iterations; ++it) {
        const uint16_t* in0 = flip ? tmp0.data() : cur0.data();
        const uint16_t* in1 = flip ? tmp1.data() : cur1.data();
        const uint16_t* in2 = flip ? tmp2.data() : cur2.data();
        const uint32_t* in3 = flip ? tmp3.data() : cur3.data();

        uint16_t* out0 = flip ? cur0.mutable_data() : tmp0.mutable_data();
        uint16_t* out1 = flip ? cur1.mutable_data() : tmp1.mutable_data();
        uint16_t* out2 = flip ? cur2.mutable_data() : tmp2.mutable_data();
        uint32_t* out3 = flip ? cur3.mutable_data() : tmp3.mutable_data();

        for (int64_t i = 0; i < n; ++i) {
            out0[i] = r127s((uint32_t)in0[i] * m0 + k0);
            out1[i] = r8191s((uint32_t)in1[i] * m1 + k1);
            out2[i] = (uint16_t)((uint32_t)in2[i] * m2 + k2);
            out3[i] = fma524287s(in3[i], m3, k3);
        }

        flip = !flip;
    }
#endif

    if (flip) {
        return py::make_tuple(tmp0, tmp1, tmp2, tmp3);
    }
    return py::make_tuple(cur0, cur1, cur2, cur3);
}



py::tuple py_mul_u64(const arr16& x0, const arr16& x1, const arr16& x2, const arr32& x3,
                     uint64_t multiplier) {
    int64_t n = require_1d_len(x0, "x0");
    require_same_len(x0, "x0", x1, "x1");
    require_same_len(x0, "x0", x2, "x2");
    require_same_len(x0, "x0", x3, "x3");

    const uint16_t m0 = (uint16_t)(multiplier % M0);
    const uint16_t m1 = (uint16_t)(multiplier % M1);
    const uint16_t m2 = (uint16_t)(multiplier % M2);
    const uint32_t m3 = (uint32_t)(multiplier % M3);

    arr16 r0({n}), r1({n}), r2({n});
    arr32 r3({n});

#if HAVE_AVX2
    const vec vm0v = V1((int)m0);
    const vec vm1v = V1((int)m1);
    const vec vm2v = V1((int)m2);

    const int64_t full = (n / L) * L;

    for (int64_t i = 0; i < full; i += L) {
        vec vx0 = Vload(x0.data() + i);
        vec vx1 = Vload(x1.data() + i);
        vec vx2 = Vload(x2.data() + i);

        vec vr0 = mul127v(vx0, vm0v);
        vec vr1 = mul8191v(vx1, vm1v);
        vec vr2 = Vm(vx2, vm2v);

        Vstore(r0.mutable_data() + i, vr0);
        Vstore(r1.mutable_data() + i, vr1);
        Vstore(r2.mutable_data() + i, vr2);

        mul524287_const8_store(x3.data() + i,     m3, r3.mutable_data() + i);
        mul524287_const8_store(x3.data() + i + 8, m3, r3.mutable_data() + i + 8);
    }

    for (int64_t i = full; i < n; ++i) {
        r0.mutable_data()[i] = r127s((uint32_t)x0.data()[i] * m0);
        r1.mutable_data()[i] = r8191s((uint32_t)x1.data()[i] * m1);
        r2.mutable_data()[i] = (uint16_t)((uint32_t)x2.data()[i] * m2);
        r3.mutable_data()[i] = mul524287s(x3.data()[i], m3);
    }
#else
    for (int64_t i = 0; i < n; ++i) {
        r0.mutable_data()[i] = r127s((uint32_t)x0.data()[i] * m0);
        r1.mutable_data()[i] = r8191s((uint32_t)x1.data()[i] * m1);
        r2.mutable_data()[i] = (uint16_t)((uint32_t)x2.data()[i] * m2);
        r3.mutable_data()[i] = mul524287s(x3.data()[i], m3);
    }
#endif

    return py::make_tuple(r0, r1, r2, r3);
}

py::tuple py_fma_u64(const arr16& x0, const arr16& x1, const arr16& x2, const arr32& x3,
                     uint64_t multiplier, uint64_t addend) {
    int64_t n = require_1d_len(x0, "x0");
    require_same_len(x0, "x0", x1, "x1");
    require_same_len(x0, "x0", x2, "x2");
    require_same_len(x0, "x0", x3, "x3");

    const uint16_t m0 = (uint16_t)(multiplier % M0);
    const uint16_t m1 = (uint16_t)(multiplier % M1);
    const uint16_t m2 = (uint16_t)(multiplier % M2);
    const uint32_t m3 = (uint32_t)(multiplier % M3);

    const uint16_t k0 = (uint16_t)(addend % M0);
    const uint16_t k1 = (uint16_t)(addend % M1);
    const uint16_t k2 = (uint16_t)(addend % M2);
    const uint32_t k3 = (uint32_t)(addend % M3);

    arr16 r0({n}), r1({n}), r2({n});
    arr32 r3({n});

#if HAVE_AVX2
    const vec vm0v = V1((int)m0);
    const vec vm1v = V1((int)m1);
    const vec vm2v = V1((int)m2);
    const vec vk0v = V1((int)k0);
    const vec vk1v = V1((int)k1);
    const vec vk2v = V1((int)k2);

    const int64_t full = (n / L) * L;

    for (int64_t i = 0; i < full; i += L) {
        vec vx0 = Vload(x0.data() + i);
        vec vx1 = Vload(x1.data() + i);
        vec vx2 = Vload(x2.data() + i);

        vec vr0 = r127v(Va(mul127v(vx0, vm0v), vk0v));
        vec vr1 = r8191v(Va(mul8191v(vx1, vm1v), vk1v));
        vec vr2 = Va(Vm(vx2, vm2v), vk2v);

        Vstore(r0.mutable_data() + i, vr0);
        Vstore(r1.mutable_data() + i, vr1);
        Vstore(r2.mutable_data() + i, vr2);

        fma524287_const8_store(x3.data() + i,     m3, k3, r3.mutable_data() + i);
        fma524287_const8_store(x3.data() + i + 8, m3, k3, r3.mutable_data() + i + 8);
    }

    for (int64_t i = full; i < n; ++i) {
        r0.mutable_data()[i] = r127s((uint32_t)x0.data()[i] * m0 + k0);
        r1.mutable_data()[i] = r8191s((uint32_t)x1.data()[i] * m1 + k1);
        r2.mutable_data()[i] = (uint16_t)((uint32_t)x2.data()[i] * m2 + k2);
        r3.mutable_data()[i] = fma524287s(x3.data()[i], m3, k3);
    }
#else
    for (int64_t i = 0; i < n; ++i) {
        r0.mutable_data()[i] = r127s((uint32_t)x0.data()[i] * m0 + k0);
        r1.mutable_data()[i] = r8191s((uint32_t)x1.data()[i] * m1 + k1);
        r2.mutable_data()[i] = (uint16_t)((uint32_t)x2.data()[i] * m2 + k2);
        r3.mutable_data()[i] = fma524287s(x3.data()[i], m3, k3);
    }
#endif

    return py::make_tuple(r0, r1, r2, r3);
}


// -- Python interface -------------------------------------------------------
py::tuple py_encode(const arr64& x_in) {
    int64_t n = require_1d_len(x_in, "x");

    arr16 o0({n}), o1({n}), o2({n});
    arr32 o3({n});

    const uint64_t* x = x_in.data();
    uint16_t* p0 = o0.mutable_data();
    uint16_t* p1 = o1.mutable_data();
    uint16_t* p2 = o2.mutable_data();
    uint32_t* p3 = o3.mutable_data();

    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        uint64_t v0 = x[i + 0];
        uint64_t v1 = x[i + 1];
        uint64_t v2 = x[i + 2];
        uint64_t v3 = x[i + 3];

        p0[i + 0] = (uint16_t)(v0 % M0);
        p1[i + 0] = (uint16_t)(v0 % M1);
        p2[i + 0] = (uint16_t)v0;
        p3[i + 0] = (uint32_t)(v0 % M3);

        p0[i + 1] = (uint16_t)(v1 % M0);
        p1[i + 1] = (uint16_t)(v1 % M1);
        p2[i + 1] = (uint16_t)v1;
        p3[i + 1] = (uint32_t)(v1 % M3);

        p0[i + 2] = (uint16_t)(v2 % M0);
        p1[i + 2] = (uint16_t)(v2 % M1);
        p2[i + 2] = (uint16_t)v2;
        p3[i + 2] = (uint32_t)(v2 % M3);

        p0[i + 3] = (uint16_t)(v3 % M0);
        p1[i + 3] = (uint16_t)(v3 % M1);
        p2[i + 3] = (uint16_t)v3;
        p3[i + 3] = (uint32_t)(v3 % M3);
    }

    for (; i < n; ++i) {
        uint64_t v = x[i];
        p0[i] = (uint16_t)(v % M0);
        p1[i] = (uint16_t)(v % M1);
        p2[i] = (uint16_t)v;
        p3[i] = (uint32_t)(v % M3);
    }

    return py::make_tuple(o0, o1, o2, o3);
}


arr64 py_decode(const arr16& r0_, const arr16& r1_, const arr16& r2_, const arr32& r3_) {
    int64_t n = require_1d_len(r0_, "r0");
    require_same_len(r0_, "r0", r1_, "r1");
    require_same_len(r0_, "r0", r2_, "r2");
    require_same_len(r0_, "r0", r3_, "r3");

    arr64 out({n});

    const uint16_t* p0 = r0_.data();
    const uint16_t* p1 = r1_.data();
    const uint16_t* p2 = r2_.data();
    const uint32_t* p3 = r3_.data();
    uint64_t* o = out.mutable_data();

    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        o[i + 0] = garner4(p0[i + 0], p1[i + 0], p2[i + 0], p3[i + 0]);
        o[i + 1] = garner4(p0[i + 1], p1[i + 1], p2[i + 1], p3[i + 1]);
        o[i + 2] = garner4(p0[i + 2], p1[i + 2], p2[i + 2], p3[i + 2]);
        o[i + 3] = garner4(p0[i + 3], p1[i + 3], p2[i + 3], p3[i + 3]);
    }
    for (; i < n; ++i) {
        o[i] = garner4(p0[i], p1[i], p2[i], p3[i]);
    }

    return out;
}


py::tuple py_op(const arr16& a0, const arr16& a1, const arr16& a2, const arr32& a3,
                const arr16& b0, const arr16& b1, const arr16& b2, const arr32& b3,
                int opcode) {
    if (opcode < 0 || opcode > 3) {
        throw std::invalid_argument("opcode must be 0=add 1=mul 2=sub 3=div");
    }

    int64_t n = require_1d_len(a0, "a0");
    require_same_len(a0, "a0", a1, "a1");
    require_same_len(a0, "a0", a2, "a2");
    require_same_len(a0, "a0", a3, "a3");
    require_same_len(a0, "a0", b0, "b0");
    require_same_len(a0, "a0", b1, "b1");
    require_same_len(a0, "a0", b2, "b2");
    require_same_len(a0, "a0", b3, "b3");

    if (opcode == 3) {
        require_divisors_invertible(b0, b1, b2, b3);
    }

    arr16 r0({n}), r1({n}), r2({n});
    arr32 r3({n});

    kernel(a0.data(), a1.data(), a2.data(), a3.data(),
           b0.data(), b1.data(), b2.data(), b3.data(),
           r0.mutable_data(), r1.mutable_data(), r2.mutable_data(), r3.mutable_data(),
           n, opcode);

    return py::make_tuple(r0, r1, r2, r3);
}

py::tuple py_fma(const arr16& a0, const arr16& a1, const arr16& a2, const arr32& a3,
                 const arr16& b0, const arr16& b1, const arr16& b2, const arr32& b3,
                 const arr16& c0, const arr16& c1, const arr16& c2, const arr32& c3) {
    int64_t n = require_1d_len(a0, "a0");
    require_same_len(a0, "a0", a1, "a1");
    require_same_len(a0, "a0", a2, "a2");
    require_same_len(a0, "a0", a3, "a3");
    require_same_len(a0, "a0", b0, "b0");
    require_same_len(a0, "a0", b1, "b1");
    require_same_len(a0, "a0", b2, "b2");
    require_same_len(a0, "a0", b3, "b3");
    require_same_len(a0, "a0", c0, "c0");
    require_same_len(a0, "a0", c1, "c1");
    require_same_len(a0, "a0", c2, "c2");
    require_same_len(a0, "a0", c3, "c3");

    arr16 r0({n}), r1({n}), r2({n});
    arr32 r3({n});

    kernel(a0.data(), a1.data(), a2.data(), a3.data(),
           b0.data(), b1.data(), b2.data(), b3.data(),
           r0.mutable_data(), r1.mutable_data(), r2.mutable_data(), r3.mutable_data(),
           n, 4,
           c0.data(), c1.data(), c2.data(), c3.data());

    return py::make_tuple(r0, r1, r2, r3);
}

// -- Module -----------------------------------------------------------------

arr64 py_fma_u64_io(const arr64& x_in, uint64_t multiplier, uint64_t addend) {
    int64_t n = require_1d_len(x_in, "x");

    const uint16_t m0 = (uint16_t)(multiplier % M0);
    const uint16_t m1 = (uint16_t)(multiplier % M1);
    const uint16_t m2 = (uint16_t)(multiplier % M2);
    const uint32_t m3 = (uint32_t)(multiplier % M3);

    const uint16_t k0 = (uint16_t)(addend % M0);
    const uint16_t k1 = (uint16_t)(addend % M1);
    const uint16_t k2 = (uint16_t)(addend % M2);
    const uint32_t k3 = (uint32_t)(addend % M3);

    arr16 e0({n}), e1({n}), e2({n});
    arr32 e3({n});
    arr64 out({n});

    const uint64_t* x = x_in.data();
    uint16_t* r0 = e0.mutable_data();
    uint16_t* r1 = e1.mutable_data();
    uint16_t* r2 = e2.mutable_data();
    uint32_t* r3 = e3.mutable_data();
    uint64_t* o = out.mutable_data();

    // encode
    int64_t j = 0;
    for (; j + 3 < n; j += 4) {
        uint64_t v0 = x[j + 0];
        uint64_t v1 = x[j + 1];
        uint64_t v2 = x[j + 2];
        uint64_t v3 = x[j + 3];

        r0[j + 0] = (uint16_t)(v0 % M0);
        r1[j + 0] = (uint16_t)(v0 % M1);
        r2[j + 0] = (uint16_t)v0;
        r3[j + 0] = (uint32_t)(v0 % M3);

        r0[j + 1] = (uint16_t)(v1 % M0);
        r1[j + 1] = (uint16_t)(v1 % M1);
        r2[j + 1] = (uint16_t)v1;
        r3[j + 1] = (uint32_t)(v1 % M3);

        r0[j + 2] = (uint16_t)(v2 % M0);
        r1[j + 2] = (uint16_t)(v2 % M1);
        r2[j + 2] = (uint16_t)v2;
        r3[j + 2] = (uint32_t)(v2 % M3);

        r0[j + 3] = (uint16_t)(v3 % M0);
        r1[j + 3] = (uint16_t)(v3 % M1);
        r2[j + 3] = (uint16_t)v3;
        r3[j + 3] = (uint32_t)(v3 % M3);
    }
    for (; j < n; ++j) {
        uint64_t v = x[j];
        r0[j] = (uint16_t)(v % M0);
        r1[j] = (uint16_t)(v % M1);
        r2[j] = (uint16_t)v;
        r3[j] = (uint32_t)(v % M3);
    }

#if HAVE_AVX2
    const vec vm0v = V1((int)m0);
    const vec vm1v = V1((int)m1);
    const vec vm2v = V1((int)m2);
    const vec vk0v = V1((int)k0);
    const vec vk1v = V1((int)k1);
    const vec vk2v = V1((int)k2);

    const int64_t full = (n / L) * L;

    for (int64_t i = 0; i < full; i += L) {
        vec vx0 = Vload(r0 + i);
        vec vx1 = Vload(r1 + i);
        vec vx2 = Vload(r2 + i);

        vec vy0 = r127v(Va(mul127v(vx0, vm0v), vk0v));
        vec vy1 = r8191v(Va(mul8191v(vx1, vm1v), vk1v));
        vec vy2 = Va(Vm(vx2, vm2v), vk2v);

        Vstore(r0 + i, vy0);
        Vstore(r1 + i, vy1);
        Vstore(r2 + i, vy2);

        fma524287_const8_store(r3 + i,     m3, k3, r3 + i);
        fma524287_const8_store(r3 + i + 8, m3, k3, r3 + i + 8);
    }

    for (int64_t i = full; i < n; ++i) {
        r0[i] = r127s((uint32_t)r0[i] * m0 + k0);
        r1[i] = r8191s((uint32_t)r1[i] * m1 + k1);
        r2[i] = (uint16_t)((uint32_t)r2[i] * m2 + k2);
        r3[i] = fma524287s(r3[i], m3, k3);
    }
#else
    for (int64_t i = 0; i < n; ++i) {
        r0[i] = r127s((uint32_t)r0[i] * m0 + k0);
        r1[i] = r8191s((uint32_t)r1[i] * m1 + k1);
        r2[i] = (uint16_t)((uint32_t)r2[i] * m2 + k2);
        r3[i] = fma524287s(r3[i], m3, k3);
    }
#endif

    // decode
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        o[i + 0] = garner4(r0[i + 0], r1[i + 0], r2[i + 0], r3[i + 0]);
        o[i + 1] = garner4(r0[i + 1], r1[i + 1], r2[i + 1], r3[i + 1]);
        o[i + 2] = garner4(r0[i + 2], r1[i + 2], r2[i + 2], r3[i + 2]);
        o[i + 3] = garner4(r0[i + 3], r1[i + 3], r2[i + 3], r3[i + 3]);
    }
    for (; i < n; ++i) {
        o[i] = garner4(r0[i], r1[i], r2[i], r3[i]);
    }

    return out;
}



arr64 py_affine_repeat_u64_io(const arr64& x_in,
                              uint64_t multiplier,
                              uint64_t addend,
                              int64_t iterations) {
    int64_t n = require_1d_len(x_in, "x");

    arr16 e0({n}), e1({n}), e2({n});
    arr32 e3({n});
    arr64 out({n});

    const uint64_t* x = x_in.data();
    uint16_t* r0 = e0.mutable_data();
    uint16_t* r1 = e1.mutable_data();
    uint16_t* r2 = e2.mutable_data();
    uint32_t* r3 = e3.mutable_data();
    uint64_t* o = out.mutable_data();

    // encode
    int64_t j = 0;
    for (; j + 3 < n; j += 4) {
        uint64_t v0 = x[j + 0];
        uint64_t v1 = x[j + 1];
        uint64_t v2 = x[j + 2];
        uint64_t v3 = x[j + 3];

        r0[j + 0] = (uint16_t)(v0 % M0);
        r1[j + 0] = (uint16_t)(v0 % M1);
        r2[j + 0] = (uint16_t)v0;
        r3[j + 0] = (uint32_t)(v0 % M3);

        r0[j + 1] = (uint16_t)(v1 % M0);
        r1[j + 1] = (uint16_t)(v1 % M1);
        r2[j + 1] = (uint16_t)v1;
        r3[j + 1] = (uint32_t)(v1 % M3);

        r0[j + 2] = (uint16_t)(v2 % M0);
        r1[j + 2] = (uint16_t)(v2 % M1);
        r2[j + 2] = (uint16_t)v2;
        r3[j + 2] = (uint32_t)(v2 % M3);

        r0[j + 3] = (uint16_t)(v3 % M0);
        r1[j + 3] = (uint16_t)(v3 % M1);
        r2[j + 3] = (uint16_t)v3;
        r3[j + 3] = (uint32_t)(v3 % M3);
    }
    for (; j < n; ++j) {
        uint64_t v = x[j];
        r0[j] = (uint16_t)(v % M0);
        r1[j] = (uint16_t)(v % M1);
        r2[j] = (uint16_t)v;
        r3[j] = (uint32_t)(v % M3);
    }

    // affine-power transform params
    const uint16_t m0 = (uint16_t)(multiplier % M0);
    const uint16_t m1 = (uint16_t)(multiplier % M1);
    const uint16_t m2 = (uint16_t)(multiplier % M2);
    const uint32_t m3 = (uint32_t)(multiplier % M3);

    const uint16_t k0 = (uint16_t)(addend % M0);
    const uint16_t k1 = (uint16_t)(addend % M1);
    const uint16_t k2 = (uint16_t)(addend % M2);
    const uint32_t k3 = (uint32_t)(addend % M3);

    uint16_t ra0 = 1, rb0 = 0, ca0 = m0, cb0 = k0;
    uint16_t ra1 = 1, rb1 = 0, ca1 = m1, cb1 = k1;
    uint16_t ra2 = 1, rb2 = 0, ca2 = m2, cb2 = k2;
    uint32_t ra3 = 1, rb3 = 0, ca3 = m3, cb3 = k3;

    int64_t it = iterations;
    while (it > 0) {
        if (it & 1) {
            rb0 = r127s((uint32_t)ca0 * rb0 + cb0);
            ra0 = r127s((uint32_t)ca0 * ra0);

            rb1 = r8191s((uint32_t)ca1 * rb1 + cb1);
            ra1 = r8191s((uint32_t)ca1 * ra1);

            rb2 = (uint16_t)((uint32_t)ca2 * rb2 + cb2);
            ra2 = (uint16_t)((uint32_t)ca2 * ra2);

            rb3 = fma524287s(rb3, ca3, cb3);
            ra3 = mul524287s(ra3, ca3);
        }

        cb0 = r127s((uint32_t)ca0 * cb0 + cb0);
        ca0 = r127s((uint32_t)ca0 * ca0);

        cb1 = r8191s((uint32_t)ca1 * cb1 + cb1);
        ca1 = r8191s((uint32_t)ca1 * ca1);

        cb2 = (uint16_t)((uint32_t)ca2 * cb2 + cb2);
        ca2 = (uint16_t)((uint32_t)ca2 * ca2);

        cb3 = fma524287s(cb3, ca3, cb3);
        ca3 = mul524287s(ca3, ca3);

        it >>= 1;
    }

#if HAVE_AVX2
    const vec vra0 = V1((int)ra0);
    const vec vra1 = V1((int)ra1);
    const vec vra2 = V1((int)ra2);
    const vec vrb0 = V1((int)rb0);
    const vec vrb1 = V1((int)rb1);
    const vec vrb2 = V1((int)rb2);

    const int64_t full = (n / L) * L;

    for (int64_t i = 0; i < full; i += L) {
        vec vx0 = Vload(r0 + i);
        vec vx1 = Vload(r1 + i);
        vec vx2 = Vload(r2 + i);

        vec vy0 = r127v(Va(mul127v(vx0, vra0), vrb0));
        vec vy1 = r8191v(Va(mul8191v(vx1, vra1), vrb1));
        vec vy2 = Va(Vm(vx2, vra2), vrb2);

        Vstore(r0 + i, vy0);
        Vstore(r1 + i, vy1);
        Vstore(r2 + i, vy2);

        fma524287_const8_store(r3 + i,     ra3, rb3, r3 + i);
        fma524287_const8_store(r3 + i + 8, ra3, rb3, r3 + i + 8);
    }

    for (int64_t i = full; i < n; ++i) {
        r0[i] = r127s((uint32_t)r0[i] * ra0 + rb0);
        r1[i] = r8191s((uint32_t)r1[i] * ra1 + rb1);
        r2[i] = (uint16_t)((uint32_t)r2[i] * ra2 + rb2);
        r3[i] = fma524287s(r3[i], ra3, rb3);
    }
#else
    for (int64_t i = 0; i < n; ++i) {
        r0[i] = r127s((uint32_t)r0[i] * ra0 + rb0);
        r1[i] = r8191s((uint32_t)r1[i] * ra1 + rb1);
        r2[i] = (uint16_t)((uint32_t)r2[i] * ra2 + rb2);
        r3[i] = fma524287s(r3[i], ra3, rb3);
    }
#endif

    // decode
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        o[i + 0] = garner4(r0[i + 0], r1[i + 0], r2[i + 0], r3[i + 0]);
        o[i + 1] = garner4(r0[i + 1], r1[i + 1], r2[i + 1], r3[i + 1]);
        o[i + 2] = garner4(r0[i + 2], r1[i + 2], r2[i + 2], r3[i + 2]);
        o[i + 3] = garner4(r0[i + 3], r1[i + 3], r2[i + 3], r3[i + 3]);
    }
    for (; i < n; ++i) {
        o[i] = garner4(r0[i], r1[i], r2[i], r3[i]);
    }

    return out;
}




arr64 py_fma_u64_io_omp(const arr64& x_in, uint64_t multiplier, uint64_t addend) {
    int64_t n = require_1d_len(x_in, "x");

    const uint16_t m0 = (uint16_t)(multiplier % M0);
    const uint16_t m1 = (uint16_t)(multiplier % M1);
    const uint16_t m2 = (uint16_t)(multiplier % M2);
    const uint32_t m3 = (uint32_t)(multiplier % M3);

    const uint16_t k0 = (uint16_t)(addend % M0);
    const uint16_t k1 = (uint16_t)(addend % M1);
    const uint16_t k2 = (uint16_t)(addend % M2);
    const uint32_t k3 = (uint32_t)(addend % M3);

    arr64 out({n});
    const uint64_t* x = x_in.data();
    uint64_t* o = out.mutable_data();

    constexpr int64_t BLOCK = 16384;
    const int64_t num_blocks = (n + BLOCK - 1) / BLOCK;

#pragma omp parallel for schedule(static)
    for (int64_t blk_id = 0; blk_id < num_blocks; ++blk_id) {
        const int64_t base = blk_id * BLOCK;
        const int64_t blk  = ((n - base) < BLOCK) ? (n - base) : BLOCK;

        alignas(64) uint16_t b0[BLOCK], b1[BLOCK], b2[BLOCK];
        alignas(64) uint32_t b3[BLOCK];

        int64_t j = 0;
        for (; j + 3 < blk; j += 4) {
            uint64_t v0 = x[base + j + 0];
            uint64_t v1 = x[base + j + 1];
            uint64_t v2 = x[base + j + 2];
            uint64_t v3 = x[base + j + 3];

            b0[j + 0] = (uint16_t)(v0 % M0);
            b1[j + 0] = (uint16_t)(v0 % M1);
            b2[j + 0] = (uint16_t)v0;
            b3[j + 0] = (uint32_t)(v0 % M3);

            b0[j + 1] = (uint16_t)(v1 % M0);
            b1[j + 1] = (uint16_t)(v1 % M1);
            b2[j + 1] = (uint16_t)v1;
            b3[j + 1] = (uint32_t)(v1 % M3);

            b0[j + 2] = (uint16_t)(v2 % M0);
            b1[j + 2] = (uint16_t)(v2 % M1);
            b2[j + 2] = (uint16_t)v2;
            b3[j + 2] = (uint32_t)(v2 % M3);

            b0[j + 3] = (uint16_t)(v3 % M0);
            b1[j + 3] = (uint16_t)(v3 % M1);
            b2[j + 3] = (uint16_t)v3;
            b3[j + 3] = (uint32_t)(v3 % M3);
        }
        for (; j < blk; ++j) {
            uint64_t v = x[base + j];
            b0[j] = (uint16_t)(v % M0);
            b1[j] = (uint16_t)(v % M1);
            b2[j] = (uint16_t)v;
            b3[j] = (uint32_t)(v % M3);
        }

#if HAVE_AVX2
        const vec vm0v = V1((int)m0);
        const vec vm1v = V1((int)m1);
        const vec vm2v = V1((int)m2);
        const vec vk0v = V1((int)k0);
        const vec vk1v = V1((int)k1);
        const vec vk2v = V1((int)k2);

        const int64_t full = (blk / L) * L;

        for (int64_t i = 0; i < full; i += L) {
            vec vx0 = Vload(b0 + i);
            vec vx1 = Vload(b1 + i);
            vec vx2 = Vload(b2 + i);

            vec vy0 = r127v(Va(mul127v(vx0, vm0v), vk0v));
            vec vy1 = r8191v(Va(mul8191v(vx1, vm1v), vk1v));
            vec vy2 = Va(Vm(vx2, vm2v), vk2v);

            Vstore(b0 + i, vy0);
            Vstore(b1 + i, vy1);
            Vstore(b2 + i, vy2);

            fma524287_const8_store(b3 + i,     m3, k3, b3 + i);
            fma524287_const8_store(b3 + i + 8, m3, k3, b3 + i + 8);
        }

        for (int64_t i = full; i < blk; ++i) {
            b0[i] = r127s((uint32_t)b0[i] * m0 + k0);
            b1[i] = r8191s((uint32_t)b1[i] * m1 + k1);
            b2[i] = (uint16_t)((uint32_t)b2[i] * m2 + k2);
            b3[i] = fma524287s(b3[i], m3, k3);
        }
#else
        for (int64_t i = 0; i < blk; ++i) {
            b0[i] = r127s((uint32_t)b0[i] * m0 + k0);
            b1[i] = r8191s((uint32_t)b1[i] * m1 + k1);
            b2[i] = (uint16_t)((uint32_t)b2[i] * m2 + k2);
            b3[i] = fma524287s(b3[i], m3, k3);
        }
#endif

        int64_t i = 0;
        for (; i + 3 < blk; i += 4) {
            o[base + i + 0] = garner4(b0[i + 0], b1[i + 0], b2[i + 0], b3[i + 0]);
            o[base + i + 1] = garner4(b0[i + 1], b1[i + 1], b2[i + 1], b3[i + 1]);
            o[base + i + 2] = garner4(b0[i + 2], b1[i + 2], b2[i + 2], b3[i + 2]);
            o[base + i + 3] = garner4(b0[i + 3], b1[i + 3], b2[i + 3], b3[i + 3]);
        }
        for (; i < blk; ++i) {
            o[base + i] = garner4(b0[i], b1[i], b2[i], b3[i]);
        }
    }

    return out;
}

arr64 py_affine_repeat_u64_io_omp(const arr64& x_in,
                                  uint64_t multiplier,
                                  uint64_t addend,
                                  int64_t iterations) {
    int64_t n = require_1d_len(x_in, "x");

    if (iterations < 0) {
        throw std::invalid_argument("iterations must be >= 0");
    }

    arr64 out({n});
    const uint64_t* x = x_in.data();
    uint64_t* o = out.mutable_data();

    if (iterations == 0) {
        std::memcpy(o, x, (size_t)n * sizeof(uint64_t));
        return out;
    }

    const uint16_t m0 = (uint16_t)(multiplier % M0);
    const uint16_t m1 = (uint16_t)(multiplier % M1);
    const uint16_t m2 = (uint16_t)(multiplier % M2);
    const uint32_t m3 = (uint32_t)(multiplier % M3);

    const uint16_t k0 = (uint16_t)(addend % M0);
    const uint16_t k1 = (uint16_t)(addend % M1);
    const uint16_t k2 = (uint16_t)(addend % M2);
    const uint32_t k3 = (uint32_t)(addend % M3);

    uint16_t ra0 = 1, rb0 = 0, ca0 = m0, cb0 = k0;
    uint16_t ra1 = 1, rb1 = 0, ca1 = m1, cb1 = k1;
    uint16_t ra2 = 1, rb2 = 0, ca2 = m2, cb2 = k2;
    uint32_t ra3 = 1, rb3 = 0, ca3 = m3, cb3 = k3;

    int64_t it = iterations;
    while (it > 0) {
        if (it & 1) {
            rb0 = r127s((uint32_t)ca0 * rb0 + cb0);
            ra0 = r127s((uint32_t)ca0 * ra0);

            rb1 = r8191s((uint32_t)ca1 * rb1 + cb1);
            ra1 = r8191s((uint32_t)ca1 * ra1);

            rb2 = (uint16_t)((uint32_t)ca2 * rb2 + cb2);
            ra2 = (uint16_t)((uint32_t)ca2 * ra2);

            rb3 = fma524287s(rb3, ca3, cb3);
            ra3 = mul524287s(ra3, ca3);
        }

        cb0 = r127s((uint32_t)ca0 * cb0 + cb0);
        ca0 = r127s((uint32_t)ca0 * ca0);

        cb1 = r8191s((uint32_t)ca1 * cb1 + cb1);
        ca1 = r8191s((uint32_t)ca1 * ca1);

        cb2 = (uint16_t)((uint32_t)ca2 * cb2 + cb2);
        ca2 = (uint16_t)((uint32_t)ca2 * ca2);

        cb3 = fma524287s(cb3, ca3, cb3);
        ca3 = mul524287s(ca3, ca3);

        it >>= 1;
    }

    constexpr int64_t BLOCK = 16384;
    const int64_t num_blocks = (n + BLOCK - 1) / BLOCK;

#pragma omp parallel for schedule(static)
    for (int64_t blk_id = 0; blk_id < num_blocks; ++blk_id) {
        const int64_t base = blk_id * BLOCK;
        const int64_t blk  = ((n - base) < BLOCK) ? (n - base) : BLOCK;

        alignas(64) uint16_t b0[BLOCK], b1[BLOCK], b2[BLOCK];
        alignas(64) uint32_t b3[BLOCK];

        int64_t j = 0;
        for (; j + 3 < blk; j += 4) {
            uint64_t v0 = x[base + j + 0];
            uint64_t v1 = x[base + j + 1];
            uint64_t v2 = x[base + j + 2];
            uint64_t v3 = x[base + j + 3];

            b0[j + 0] = (uint16_t)(v0 % M0);
            b1[j + 0] = (uint16_t)(v0 % M1);
            b2[j + 0] = (uint16_t)v0;
            b3[j + 0] = (uint32_t)(v0 % M3);

            b0[j + 1] = (uint16_t)(v1 % M0);
            b1[j + 1] = (uint16_t)(v1 % M1);
            b2[j + 1] = (uint16_t)v1;
            b3[j + 1] = (uint32_t)(v1 % M3);

            b0[j + 2] = (uint16_t)(v2 % M0);
            b1[j + 2] = (uint16_t)(v2 % M1);
            b2[j + 2] = (uint16_t)v2;
            b3[j + 2] = (uint32_t)(v2 % M3);

            b0[j + 3] = (uint16_t)(v3 % M0);
            b1[j + 3] = (uint16_t)(v3 % M1);
            b2[j + 3] = (uint16_t)v3;
            b3[j + 3] = (uint32_t)(v3 % M3);
        }
        for (; j < blk; ++j) {
            uint64_t v = x[base + j];
            b0[j] = (uint16_t)(v % M0);
            b1[j] = (uint16_t)(v % M1);
            b2[j] = (uint16_t)v;
            b3[j] = (uint32_t)(v % M3);
        }

#if HAVE_AVX2
        const vec vra0 = V1((int)ra0);
        const vec vra1 = V1((int)ra1);
        const vec vra2 = V1((int)ra2);
        const vec vrb0 = V1((int)rb0);
        const vec vrb1 = V1((int)rb1);
        const vec vrb2 = V1((int)rb2);

        const int64_t full = (blk / L) * L;

        for (int64_t i = 0; i < full; i += L) {
            vec vx0 = Vload(b0 + i);
            vec vx1 = Vload(b1 + i);
            vec vx2 = Vload(b2 + i);

            vec vy0 = r127v(Va(mul127v(vx0, vra0), vrb0));
            vec vy1 = r8191v(Va(mul8191v(vx1, vra1), vrb1));
            vec vy2 = Va(Vm(vx2, vra2), vrb2);

            Vstore(b0 + i, vy0);
            Vstore(b1 + i, vy1);
            Vstore(b2 + i, vy2);

            fma524287_const8_store(b3 + i,     ra3, rb3, b3 + i);
            fma524287_const8_store(b3 + i + 8, ra3, rb3, b3 + i + 8);
        }

        for (int64_t i = full; i < blk; ++i) {
            b0[i] = r127s((uint32_t)b0[i] * ra0 + rb0);
            b1[i] = r8191s((uint32_t)b1[i] * ra1 + rb1);
            b2[i] = (uint16_t)((uint32_t)b2[i] * ra2 + rb2);
            b3[i] = fma524287s(b3[i], ra3, rb3);
        }
#else
        for (int64_t i = 0; i < blk; ++i) {
            b0[i] = r127s((uint32_t)b0[i] * ra0 + rb0);
            b1[i] = r8191s((uint32_t)b1[i] * ra1 + rb1);
            b2[i] = (uint16_t)((uint32_t)b2[i] * ra2 + rb2);
            b3[i] = fma524287s(b3[i], ra3, rb3);
        }
#endif

        int64_t i = 0;
        for (; i + 3 < blk; i += 4) {
            o[base + i + 0] = garner4(b0[i + 0], b1[i + 0], b2[i + 0], b3[i + 0]);
            o[base + i + 1] = garner4(b0[i + 1], b1[i + 1], b2[i + 1], b3[i + 1]);
            o[base + i + 2] = garner4(b0[i + 2], b1[i + 2], b2[i + 2], b3[i + 2]);
            o[base + i + 3] = garner4(b0[i + 3], b1[i + 3], b2[i + 3], b3[i + 3]);
        }
        for (; i < blk; ++i) {
            o[base + i] = garner4(b0[i], b1[i], b2[i], b3[i]);
        }
    }

    return out;
}

int py_omp_max_threads() {
    return omp_get_max_threads();
}


void py_omp_set_num_threads(int n) {
    if (n <= 0) {
        throw std::invalid_argument("omp_set_num_threads(n): n must be >= 1");
    }
    omp_set_num_threads(n);
}

int py_omp_num_procs() {
    return omp_get_num_procs();
}


arr64 py_fma_u64_auto(const arr64& x_in, uint64_t multiplier, uint64_t addend) {
    int64_t n = require_1d_len(x_in, "x");

#ifdef _OPENMP
    if (omp_get_max_threads() > 1 && n >= 65536) {
        return py_fma_u64_io_omp(x_in, multiplier, addend);
    }
#endif
    return py_fma_u64_io(x_in, multiplier, addend);
}

arr64 py_affine_repeat_u64_auto(const arr64& x_in,
                                uint64_t multiplier,
                                uint64_t addend,
                                int64_t iterations) {
    int64_t n = require_1d_len(x_in, "x");

#ifdef _OPENMP
    if (omp_get_max_threads() > 1 && n >= 4096) {
        return py_affine_repeat_u64_io_omp(x_in, multiplier, addend, iterations);
    }
#endif
    return py_affine_repeat_u64_io(x_in, multiplier, addend, iterations);
}


arr64 py_mul_u64_io(const arr64& x_in, uint64_t multiplier) {
    int64_t n = require_1d_len(x_in, "x");

    const uint16_t m0 = (uint16_t)(multiplier % M0);
    const uint16_t m1 = (uint16_t)(multiplier % M1);
    const uint16_t m2 = (uint16_t)(multiplier % M2);
    const uint32_t m3 = (uint32_t)(multiplier % M3);

    arr16 e0({n}), e1({n}), e2({n});
    arr32 e3({n});
    arr64 out({n});

    const uint64_t* x = x_in.data();
    uint16_t* r0 = e0.mutable_data();
    uint16_t* r1 = e1.mutable_data();
    uint16_t* r2 = e2.mutable_data();
    uint32_t* r3 = e3.mutable_data();
    uint64_t* o = out.mutable_data();

    int64_t j = 0;
    for (; j + 3 < n; j += 4) {
        uint64_t v0 = x[j + 0];
        uint64_t v1 = x[j + 1];
        uint64_t v2 = x[j + 2];
        uint64_t v3 = x[j + 3];

        r0[j + 0] = (uint16_t)(v0 % M0);
        r1[j + 0] = (uint16_t)(v0 % M1);
        r2[j + 0] = (uint16_t)v0;
        r3[j + 0] = (uint32_t)(v0 % M3);

        r0[j + 1] = (uint16_t)(v1 % M0);
        r1[j + 1] = (uint16_t)(v1 % M1);
        r2[j + 1] = (uint16_t)v1;
        r3[j + 1] = (uint32_t)(v1 % M3);

        r0[j + 2] = (uint16_t)(v2 % M0);
        r1[j + 2] = (uint16_t)(v2 % M1);
        r2[j + 2] = (uint16_t)v2;
        r3[j + 2] = (uint32_t)(v2 % M3);

        r0[j + 3] = (uint16_t)(v3 % M0);
        r1[j + 3] = (uint16_t)(v3 % M1);
        r2[j + 3] = (uint16_t)v3;
        r3[j + 3] = (uint32_t)(v3 % M3);
    }
    for (; j < n; ++j) {
        uint64_t v = x[j];
        r0[j] = (uint16_t)(v % M0);
        r1[j] = (uint16_t)(v % M1);
        r2[j] = (uint16_t)v;
        r3[j] = (uint32_t)(v % M3);
    }

#if HAVE_AVX2
    const vec vm0v = V1((int)m0);
    const vec vm1v = V1((int)m1);
    const vec vm2v = V1((int)m2);

    const int64_t full = (n / L) * L;

    for (int64_t i = 0; i < full; i += L) {
        vec vx0 = Vload(r0 + i);
        vec vx1 = Vload(r1 + i);
        vec vx2 = Vload(r2 + i);

        vec vy0 = r127v(mul127v(vx0, vm0v));
        vec vy1 = r8191v(mul8191v(vx1, vm1v));
        vec vy2 = Vm(vx2, vm2v);

        Vstore(r0 + i, vy0);
        Vstore(r1 + i, vy1);
        Vstore(r2 + i, vy2);

        fma524287_const8_store(r3 + i,     m3, 0, r3 + i);
        fma524287_const8_store(r3 + i + 8, m3, 0, r3 + i + 8);
    }

    for (int64_t i = full; i < n; ++i) {
        r0[i] = r127s((uint32_t)r0[i] * m0);
        r1[i] = r8191s((uint32_t)r1[i] * m1);
        r2[i] = (uint16_t)((uint32_t)r2[i] * m2);
        r3[i] = mul524287s(r3[i], m3);
    }
#else
    for (int64_t i = 0; i < n; ++i) {
        r0[i] = r127s((uint32_t)r0[i] * m0);
        r1[i] = r8191s((uint32_t)r1[i] * m1);
        r2[i] = (uint16_t)((uint32_t)r2[i] * m2);
        r3[i] = mul524287s(r3[i], m3);
    }
#endif

    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        o[i + 0] = garner4(r0[i + 0], r1[i + 0], r2[i + 0], r3[i + 0]);
        o[i + 1] = garner4(r0[i + 1], r1[i + 1], r2[i + 1], r3[i + 1]);
        o[i + 2] = garner4(r0[i + 2], r1[i + 2], r2[i + 2], r3[i + 2]);
        o[i + 3] = garner4(r0[i + 3], r1[i + 3], r2[i + 3], r3[i + 3]);
    }
    for (; i < n; ++i) {
        o[i] = garner4(r0[i], r1[i], r2[i], r3[i]);
    }

    return out;
}

arr64 py_mul_u64_io_omp(const arr64& x_in, uint64_t multiplier) {
    int64_t n = require_1d_len(x_in, "x");

    const uint16_t m0 = (uint16_t)(multiplier % M0);
    const uint16_t m1 = (uint16_t)(multiplier % M1);
    const uint16_t m2 = (uint16_t)(multiplier % M2);
    const uint32_t m3 = (uint32_t)(multiplier % M3);

    arr64 out({n});
    const uint64_t* x = x_in.data();
    uint64_t* o = out.mutable_data();

    constexpr int64_t BLOCK = 16384;
    const int64_t num_blocks = (n + BLOCK - 1) / BLOCK;

#pragma omp parallel for schedule(static)
    for (int64_t blk_id = 0; blk_id < num_blocks; ++blk_id) {
        const int64_t base = blk_id * BLOCK;
        const int64_t blk  = ((n - base) < BLOCK) ? (n - base) : BLOCK;

        alignas(64) uint16_t b0[BLOCK], b1[BLOCK], b2[BLOCK];
        alignas(64) uint32_t b3[BLOCK];

        int64_t j = 0;
        for (; j + 3 < blk; j += 4) {
            uint64_t v0 = x[base + j + 0];
            uint64_t v1 = x[base + j + 1];
            uint64_t v2 = x[base + j + 2];
            uint64_t v3 = x[base + j + 3];

            b0[j + 0] = (uint16_t)(v0 % M0);
            b1[j + 0] = (uint16_t)(v0 % M1);
            b2[j + 0] = (uint16_t)v0;
            b3[j + 0] = (uint32_t)(v0 % M3);

            b0[j + 1] = (uint16_t)(v1 % M0);
            b1[j + 1] = (uint16_t)(v1 % M1);
            b2[j + 1] = (uint16_t)v1;
            b3[j + 1] = (uint32_t)(v1 % M3);

            b0[j + 2] = (uint16_t)(v2 % M0);
            b1[j + 2] = (uint16_t)(v2 % M1);
            b2[j + 2] = (uint16_t)v2;
            b3[j + 2] = (uint32_t)(v2 % M3);

            b0[j + 3] = (uint16_t)(v3 % M0);
            b1[j + 3] = (uint16_t)(v3 % M1);
            b2[j + 3] = (uint16_t)v3;
            b3[j + 3] = (uint32_t)(v3 % M3);
        }
        for (; j < blk; ++j) {
            uint64_t v = x[base + j];
            b0[j] = (uint16_t)(v % M0);
            b1[j] = (uint16_t)(v % M1);
            b2[j] = (uint16_t)v;
            b3[j] = (uint32_t)(v % M3);
        }

#if HAVE_AVX2
        const vec vm0v = V1((int)m0);
        const vec vm1v = V1((int)m1);
        const vec vm2v = V1((int)m2);

        const int64_t full = (blk / L) * L;

        for (int64_t i = 0; i < full; i += L) {
            vec vx0 = Vload(b0 + i);
            vec vx1 = Vload(b1 + i);
            vec vx2 = Vload(b2 + i);

            vec vy0 = r127v(mul127v(vx0, vm0v));
            vec vy1 = r8191v(mul8191v(vx1, vm1v));
            vec vy2 = Vm(vx2, vm2v);

            Vstore(b0 + i, vy0);
            Vstore(b1 + i, vy1);
            Vstore(b2 + i, vy2);

            fma524287_const8_store(b3 + i,     m3, 0, b3 + i);
            fma524287_const8_store(b3 + i + 8, m3, 0, b3 + i + 8);
        }

        for (int64_t i = full; i < blk; ++i) {
            b0[i] = r127s((uint32_t)b0[i] * m0);
            b1[i] = r8191s((uint32_t)b1[i] * m1);
            b2[i] = (uint16_t)((uint32_t)b2[i] * m2);
            b3[i] = mul524287s(b3[i], m3);
        }
#else
        for (int64_t i = 0; i < blk; ++i) {
            b0[i] = r127s((uint32_t)b0[i] * m0);
            b1[i] = r8191s((uint32_t)b1[i] * m1);
            b2[i] = (uint16_t)((uint32_t)b2[i] * m2);
            b3[i] = mul524287s(b3[i], m3);
        }
#endif

        int64_t i = 0;
        for (; i + 3 < blk; i += 4) {
            o[base + i + 0] = garner4(b0[i + 0], b1[i + 0], b2[i + 0], b3[i + 0]);
            o[base + i + 1] = garner4(b0[i + 1], b1[i + 1], b2[i + 1], b3[i + 1]);
            o[base + i + 2] = garner4(b0[i + 2], b1[i + 2], b2[i + 2], b3[i + 2]);
            o[base + i + 3] = garner4(b0[i + 3], b1[i + 3], b2[i + 3], b3[i + 3]);
        }
        for (; i < blk; ++i) {
            o[base + i] = garner4(b0[i], b1[i], b2[i], b3[i]);
        }
    }

    return out;
}

arr64 py_mul_u64_auto(const arr64& x_in, uint64_t multiplier) {
    int64_t n = require_1d_len(x_in, "x");

#ifdef _OPENMP
    if (omp_get_max_threads() > 1 && n >= 32768) {
        return py_mul_u64_io_omp(x_in, multiplier);
    }
#endif
    return py_mul_u64_io(x_in, multiplier);
}


arr64 py_add_u64_io(const arr64& x_in, uint64_t addend) {
    int64_t n = require_1d_len(x_in, "x");

    const uint16_t k0 = (uint16_t)(addend % M0);
    const uint16_t k1 = (uint16_t)(addend % M1);
    const uint16_t k2 = (uint16_t)(addend % M2);
    const uint32_t k3 = (uint32_t)(addend % M3);

    arr16 e0({n}), e1({n}), e2({n});
    arr32 e3({n});
    arr64 out({n});

    const uint64_t* x = x_in.data();
    uint16_t* r0 = e0.mutable_data();
    uint16_t* r1 = e1.mutable_data();
    uint16_t* r2 = e2.mutable_data();
    uint32_t* r3 = e3.mutable_data();
    uint64_t* o = out.mutable_data();

    int64_t j = 0;
    for (; j + 3 < n; j += 4) {
        uint64_t v0 = x[j + 0];
        uint64_t v1 = x[j + 1];
        uint64_t v2 = x[j + 2];
        uint64_t v3 = x[j + 3];

        r0[j + 0] = (uint16_t)(v0 % M0);
        r1[j + 0] = (uint16_t)(v0 % M1);
        r2[j + 0] = (uint16_t)v0;
        r3[j + 0] = (uint32_t)(v0 % M3);

        r0[j + 1] = (uint16_t)(v1 % M0);
        r1[j + 1] = (uint16_t)(v1 % M1);
        r2[j + 1] = (uint16_t)v1;
        r3[j + 1] = (uint32_t)(v1 % M3);

        r0[j + 2] = (uint16_t)(v2 % M0);
        r1[j + 2] = (uint16_t)(v2 % M1);
        r2[j + 2] = (uint16_t)v2;
        r3[j + 2] = (uint32_t)(v2 % M3);

        r0[j + 3] = (uint16_t)(v3 % M0);
        r1[j + 3] = (uint16_t)(v3 % M1);
        r2[j + 3] = (uint16_t)v3;
        r3[j + 3] = (uint32_t)(v3 % M3);
    }
    for (; j < n; ++j) {
        uint64_t v = x[j];
        r0[j] = (uint16_t)(v % M0);
        r1[j] = (uint16_t)(v % M1);
        r2[j] = (uint16_t)v;
        r3[j] = (uint32_t)(v % M3);
    }

#if HAVE_AVX2
    const vec vk0v = V1((int)k0);
    const vec vk1v = V1((int)k1);
    const vec vk2v = V1((int)k2);

    const int64_t full = (n / L) * L;

    for (int64_t i = 0; i < full; i += L) {
        vec vx0 = Vload(r0 + i);
        vec vx1 = Vload(r1 + i);
        vec vx2 = Vload(r2 + i);

        vec vy0 = r127v(Va(vx0, vk0v));
        vec vy1 = r8191v(Va(vx1, vk1v));
        vec vy2 = Va(vx2, vk2v);

        Vstore(r0 + i, vy0);
        Vstore(r1 + i, vy1);
        Vstore(r2 + i, vy2);

        fma524287_const8_store(r3 + i,     1, k3, r3 + i);
        fma524287_const8_store(r3 + i + 8, 1, k3, r3 + i + 8);
    }

    for (int64_t i = full; i < n; ++i) {
        r0[i] = r127s((uint32_t)r0[i] + k0);
        r1[i] = r8191s((uint32_t)r1[i] + k1);
        r2[i] = (uint16_t)((uint32_t)r2[i] + k2);
        r3[i] = fma524287s(r3[i], 1, k3);
    }
#else
    for (int64_t i = 0; i < n; ++i) {
        r0[i] = r127s((uint32_t)r0[i] + k0);
        r1[i] = r8191s((uint32_t)r1[i] + k1);
        r2[i] = (uint16_t)((uint32_t)r2[i] + k2);
        r3[i] = fma524287s(r3[i], 1, k3);
    }
#endif

    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        o[i + 0] = garner4(r0[i + 0], r1[i + 0], r2[i + 0], r3[i + 0]);
        o[i + 1] = garner4(r0[i + 1], r1[i + 1], r2[i + 1], r3[i + 1]);
        o[i + 2] = garner4(r0[i + 2], r1[i + 2], r2[i + 2], r3[i + 2]);
        o[i + 3] = garner4(r0[i + 3], r1[i + 3], r2[i + 3], r3[i + 3]);
    }
    for (; i < n; ++i) {
        o[i] = garner4(r0[i], r1[i], r2[i], r3[i]);
    }

    return out;
}

arr64 py_add_u64_io_omp(const arr64& x_in, uint64_t addend) {
    int64_t n = require_1d_len(x_in, "x");

    const uint16_t k0 = (uint16_t)(addend % M0);
    const uint16_t k1 = (uint16_t)(addend % M1);
    const uint16_t k2 = (uint16_t)(addend % M2);
    const uint32_t k3 = (uint32_t)(addend % M3);

    arr64 out({n});
    const uint64_t* x = x_in.data();
    uint64_t* o = out.mutable_data();

    constexpr int64_t BLOCK = 16384;
    const int64_t num_blocks = (n + BLOCK - 1) / BLOCK;

#pragma omp parallel for schedule(static)
    for (int64_t blk_id = 0; blk_id < num_blocks; ++blk_id) {
        const int64_t base = blk_id * BLOCK;
        const int64_t blk  = ((n - base) < BLOCK) ? (n - base) : BLOCK;

        alignas(64) uint16_t b0[BLOCK], b1[BLOCK], b2[BLOCK];
        alignas(64) uint32_t b3[BLOCK];

        int64_t j = 0;
        for (; j + 3 < blk; j += 4) {
            uint64_t v0 = x[base + j + 0];
            uint64_t v1 = x[base + j + 1];
            uint64_t v2 = x[base + j + 2];
            uint64_t v3 = x[base + j + 3];

            b0[j + 0] = (uint16_t)(v0 % M0);
            b1[j + 0] = (uint16_t)(v0 % M1);
            b2[j + 0] = (uint16_t)v0;
            b3[j + 0] = (uint32_t)(v0 % M3);

            b0[j + 1] = (uint16_t)(v1 % M0);
            b1[j + 1] = (uint16_t)(v1 % M1);
            b2[j + 1] = (uint16_t)v1;
            b3[j + 1] = (uint32_t)(v1 % M3);

            b0[j + 2] = (uint16_t)(v2 % M0);
            b1[j + 2] = (uint16_t)(v2 % M1);
            b2[j + 2] = (uint16_t)v2;
            b3[j + 2] = (uint32_t)(v2 % M3);

            b0[j + 3] = (uint16_t)(v3 % M0);
            b1[j + 3] = (uint16_t)(v3 % M1);
            b2[j + 3] = (uint16_t)v3;
            b3[j + 3] = (uint32_t)(v3 % M3);
        }
        for (; j < blk; ++j) {
            uint64_t v = x[base + j];
            b0[j] = (uint16_t)(v % M0);
            b1[j] = (uint16_t)(v % M1);
            b2[j] = (uint16_t)v;
            b3[j] = (uint32_t)(v % M3);
        }

#if HAVE_AVX2
        const vec vk0v = V1((int)k0);
        const vec vk1v = V1((int)k1);
        const vec vk2v = V1((int)k2);

        const int64_t full = (blk / L) * L;

        for (int64_t i = 0; i < full; i += L) {
            vec vx0 = Vload(b0 + i);
            vec vx1 = Vload(b1 + i);
            vec vx2 = Vload(b2 + i);

            vec vy0 = r127v(Va(vx0, vk0v));
            vec vy1 = r8191v(Va(vx1, vk1v));
            vec vy2 = Va(vx2, vk2v);

            Vstore(b0 + i, vy0);
            Vstore(b1 + i, vy1);
            Vstore(b2 + i, vy2);

            fma524287_const8_store(b3 + i,     1, k3, b3 + i);
            fma524287_const8_store(b3 + i + 8, 1, k3, b3 + i + 8);
        }

        for (int64_t i = full; i < blk; ++i) {
            b0[i] = r127s((uint32_t)b0[i] + k0);
            b1[i] = r8191s((uint32_t)b1[i] + k1);
            b2[i] = (uint16_t)((uint32_t)b2[i] + k2);
            b3[i] = fma524287s(b3[i], 1, k3);
        }
#else
        for (int64_t i = 0; i < blk; ++i) {
            b0[i] = r127s((uint32_t)b0[i] + k0);
            b1[i] = r8191s((uint32_t)b1[i] + k1);
            b2[i] = (uint16_t)((uint32_t)b2[i] + k2);
            b3[i] = fma524287s(b3[i], 1, k3);
        }
#endif

        int64_t i = 0;
        for (; i + 3 < blk; i += 4) {
            o[base + i + 0] = garner4(b0[i + 0], b1[i + 0], b2[i + 0], b3[i + 0]);
            o[base + i + 1] = garner4(b0[i + 1], b1[i + 1], b2[i + 1], b3[i + 1]);
            o[base + i + 2] = garner4(b0[i + 2], b1[i + 2], b2[i + 2], b3[i + 2]);
            o[base + i + 3] = garner4(b0[i + 3], b1[i + 3], b2[i + 3], b3[i + 3]);
        }
        for (; i < blk; ++i) {
            o[base + i] = garner4(b0[i], b1[i], b2[i], b3[i]);
        }
    }

    return out;
}


arr64 py_add_u64_auto(const arr64& x_in, uint64_t addend) {
    int64_t n = require_1d_len(x_in, "x");

#ifdef _OPENMP
    if (omp_get_max_threads() > 1 && n >= 16384) {
        return py_add_u64_io_omp(x_in, addend);
    }
#endif
    return py_add_u64_io(x_in, addend);
}


arr64 py_sub_u64_io(const arr64& x_in, uint64_t subtrahend) {
    int64_t n = require_1d_len(x_in, "x");

    const uint16_t s0 = (uint16_t)((subtrahend % M0) ? (M0 - (subtrahend % M0)) : 0);
    const uint16_t s1 = (uint16_t)((subtrahend % M1) ? (M1 - (subtrahend % M1)) : 0);
    const uint16_t s2 = (uint16_t)(0 - (uint16_t)(subtrahend % M2));
    const uint32_t s3 = (uint32_t)((subtrahend % M3) ? (M3 - (subtrahend % M3)) : 0);

    arr16 e0({n}), e1({n}), e2({n});
    arr32 e3({n});
    arr64 out({n});

    const uint64_t* x = x_in.data();
    uint16_t* r0 = e0.mutable_data();
    uint16_t* r1 = e1.mutable_data();
    uint16_t* r2 = e2.mutable_data();
    uint32_t* r3 = e3.mutable_data();
    uint64_t* o = out.mutable_data();

    int64_t j = 0;
    for (; j + 3 < n; j += 4) {
        uint64_t v0 = x[j + 0];
        uint64_t v1 = x[j + 1];
        uint64_t v2 = x[j + 2];
        uint64_t v3 = x[j + 3];

        r0[j + 0] = (uint16_t)(v0 % M0);
        r1[j + 0] = (uint16_t)(v0 % M1);
        r2[j + 0] = (uint16_t)v0;
        r3[j + 0] = (uint32_t)(v0 % M3);

        r0[j + 1] = (uint16_t)(v1 % M0);
        r1[j + 1] = (uint16_t)(v1 % M1);
        r2[j + 1] = (uint16_t)v1;
        r3[j + 1] = (uint32_t)(v1 % M3);

        r0[j + 2] = (uint16_t)(v2 % M0);
        r1[j + 2] = (uint16_t)(v2 % M1);
        r2[j + 2] = (uint16_t)v2;
        r3[j + 2] = (uint32_t)(v2 % M3);

        r0[j + 3] = (uint16_t)(v3 % M0);
        r1[j + 3] = (uint16_t)(v3 % M1);
        r2[j + 3] = (uint16_t)v3;
        r3[j + 3] = (uint32_t)(v3 % M3);
    }
    for (; j < n; ++j) {
        uint64_t v = x[j];
        r0[j] = (uint16_t)(v % M0);
        r1[j] = (uint16_t)(v % M1);
        r2[j] = (uint16_t)v;
        r3[j] = (uint32_t)(v % M3);
    }

#if HAVE_AVX2
    const vec vs0v = V1((int)s0);
    const vec vs1v = V1((int)s1);
    const vec vs2v = V1((int)s2);

    const int64_t full = (n / L) * L;

    for (int64_t i = 0; i < full; i += L) {
        vec vx0 = Vload(r0 + i);
        vec vx1 = Vload(r1 + i);
        vec vx2 = Vload(r2 + i);

        vec vy0 = r127v(Va(vx0, vs0v));
        vec vy1 = r8191v(Va(vx1, vs1v));
        vec vy2 = Va(vx2, vs2v);

        Vstore(r0 + i, vy0);
        Vstore(r1 + i, vy1);
        Vstore(r2 + i, vy2);

        fma524287_const8_store(r3 + i,     1, s3, r3 + i);
        fma524287_const8_store(r3 + i + 8, 1, s3, r3 + i + 8);
    }

    for (int64_t i = full; i < n; ++i) {
        r0[i] = r127s((uint32_t)r0[i] + s0);
        r1[i] = r8191s((uint32_t)r1[i] + s1);
        r2[i] = (uint16_t)((uint32_t)r2[i] + s2);
        r3[i] = fma524287s(r3[i], 1, s3);
    }
#else
    for (int64_t i = 0; i < n; ++i) {
        r0[i] = r127s((uint32_t)r0[i] + s0);
        r1[i] = r8191s((uint32_t)r1[i] + s1);
        r2[i] = (uint16_t)((uint32_t)r2[i] + s2);
        r3[i] = fma524287s(r3[i], 1, s3);
    }
#endif

    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        o[i + 0] = garner4(r0[i + 0], r1[i + 0], r2[i + 0], r3[i + 0]);
        o[i + 1] = garner4(r0[i + 1], r1[i + 1], r2[i + 1], r3[i + 1]);
        o[i + 2] = garner4(r0[i + 2], r1[i + 2], r2[i + 2], r3[i + 2]);
        o[i + 3] = garner4(r0[i + 3], r1[i + 3], r2[i + 3], r3[i + 3]);
    }
    for (; i < n; ++i) {
        o[i] = garner4(r0[i], r1[i], r2[i], r3[i]);
    }

    return out;
}

arr64 py_sub_u64_io_omp(const arr64& x_in, uint64_t subtrahend) {
    int64_t n = require_1d_len(x_in, "x");

    const uint16_t s0 = (uint16_t)((subtrahend % M0) ? (M0 - (subtrahend % M0)) : 0);
    const uint16_t s1 = (uint16_t)((subtrahend % M1) ? (M1 - (subtrahend % M1)) : 0);
    const uint16_t s2 = (uint16_t)(0 - (uint16_t)(subtrahend % M2));
    const uint32_t s3 = (uint32_t)((subtrahend % M3) ? (M3 - (subtrahend % M3)) : 0);

    arr64 out({n});
    const uint64_t* x = x_in.data();
    uint64_t* o = out.mutable_data();

    constexpr int64_t BLOCK = 16384;
    const int64_t num_blocks = (n + BLOCK - 1) / BLOCK;

#pragma omp parallel for schedule(static)
    for (int64_t blk_id = 0; blk_id < num_blocks; ++blk_id) {
        const int64_t base = blk_id * BLOCK;
        const int64_t blk  = ((n - base) < BLOCK) ? (n - base) : BLOCK;

        alignas(64) uint16_t b0[BLOCK], b1[BLOCK], b2[BLOCK];
        alignas(64) uint32_t b3[BLOCK];

        int64_t j = 0;
        for (; j + 3 < blk; j += 4) {
            uint64_t v0 = x[base + j + 0];
            uint64_t v1 = x[base + j + 1];
            uint64_t v2 = x[base + j + 2];
            uint64_t v3 = x[base + j + 3];

            b0[j + 0] = (uint16_t)(v0 % M0);
            b1[j + 0] = (uint16_t)(v0 % M1);
            b2[j + 0] = (uint16_t)v0;
            b3[j + 0] = (uint32_t)(v0 % M3);

            b0[j + 1] = (uint16_t)(v1 % M0);
            b1[j + 1] = (uint16_t)(v1 % M1);
            b2[j + 1] = (uint16_t)v1;
            b3[j + 1] = (uint32_t)(v1 % M3);

            b0[j + 2] = (uint16_t)(v2 % M0);
            b1[j + 2] = (uint16_t)(v2 % M1);
            b2[j + 2] = (uint16_t)v2;
            b3[j + 2] = (uint32_t)(v2 % M3);

            b0[j + 3] = (uint16_t)(v3 % M0);
            b1[j + 3] = (uint16_t)(v3 % M1);
            b2[j + 3] = (uint16_t)v3;
            b3[j + 3] = (uint32_t)(v3 % M3);
        }
        for (; j < blk; ++j) {
            uint64_t v = x[base + j];
            b0[j] = (uint16_t)(v % M0);
            b1[j] = (uint16_t)(v % M1);
            b2[j] = (uint16_t)v;
            b3[j] = (uint32_t)(v % M3);
        }

#if HAVE_AVX2
        const vec vs0v = V1((int)s0);
        const vec vs1v = V1((int)s1);
        const vec vs2v = V1((int)s2);

        const int64_t full = (blk / L) * L;

        for (int64_t i = 0; i < full; i += L) {
            vec vx0 = Vload(b0 + i);
            vec vx1 = Vload(b1 + i);
            vec vx2 = Vload(b2 + i);

            vec vy0 = r127v(Va(vx0, vs0v));
            vec vy1 = r8191v(Va(vx1, vs1v));
            vec vy2 = Va(vx2, vs2v);

            Vstore(b0 + i, vy0);
            Vstore(b1 + i, vy1);
            Vstore(b2 + i, vy2);

            fma524287_const8_store(b3 + i,     1, s3, b3 + i);
            fma524287_const8_store(b3 + i + 8, 1, s3, b3 + i + 8);
        }

        for (int64_t i = full; i < blk; ++i) {
            b0[i] = r127s((uint32_t)b0[i] + s0);
            b1[i] = r8191s((uint32_t)b1[i] + s1);
            b2[i] = (uint16_t)((uint32_t)b2[i] + s2);
            b3[i] = fma524287s(b3[i], 1, s3);
        }
#else
        for (int64_t i = 0; i < blk; ++i) {
            b0[i] = r127s((uint32_t)b0[i] + s0);
            b1[i] = r8191s((uint32_t)b1[i] + s1);
            b2[i] = (uint16_t)((uint32_t)b2[i] + s2);
            b3[i] = fma524287s(b3[i], 1, s3);
        }
#endif

        int64_t i = 0;
        for (; i + 3 < blk; i += 4) {
            o[base + i + 0] = garner4(b0[i + 0], b1[i + 0], b2[i + 0], b3[i + 0]);
            o[base + i + 1] = garner4(b0[i + 1], b1[i + 1], b2[i + 1], b3[i + 1]);
            o[base + i + 2] = garner4(b0[i + 2], b1[i + 2], b2[i + 2], b3[i + 2]);
            o[base + i + 3] = garner4(b0[i + 3], b1[i + 3], b2[i + 3], b3[i + 3]);
        }
        for (; i < blk; ++i) {
            o[base + i] = garner4(b0[i], b1[i], b2[i], b3[i]);
        }
    }

    return out;
}


arr64 py_sub_u64_auto(const arr64& x_in, uint64_t subtrahend) {
    int64_t n = require_1d_len(x_in, "x");

#ifdef _OPENMP
    if (omp_get_max_threads() > 1 && n >= 16384) {
        return py_sub_u64_io_omp(x_in, subtrahend);
    }
#endif
    return py_sub_u64_io(x_in, subtrahend);
}

PYBIND11_MODULE(_core, m) {
    m.doc() =
        "_core v0.4.0rc1\n"
        "4-rail exact integer arithmetic.\n"
        "Best-known speed baseline + native affine_repeat.\n"
        "Moduli: 127 x 8191 x 65536 x 524287.\n";

    m.attr("M")        = (uint64_t)BM;
    m.attr("M0")       = (uint32_t)M0;
    m.attr("M1")       = (uint32_t)M1;
    m.attr("M2")       = (uint32_t)M2;
    m.attr("M3")       = (uint32_t)M3;
    m.attr("HAS_AVX2") = (bool)HAVE_AVX2;

    m.def("encode", &py_encode,
          "uint64[] -> (r0:u16, r1:u16, r2:u16, r3:u32) residue arrays.");
    m.def("decode", &py_decode,
          "(r0:u16, r1:u16, r2:u16, r3:u32) -> uint64[] via 4-rail Garner CRT.");
    m.def("op", &py_op,
          "opcode: 0=add 1=mul 2=sub 3=div");
    m.def("add",
          [](const arr16& a0, const arr16& a1, const arr16& a2, const arr32& a3,
             const arr16& b0, const arr16& b1, const arr16& b2, const arr32& b3) {
              return py_op(a0, a1, a2, a3, b0, b1, b2, b3, 0);
          },
          "Exact addition mod each rail.");
    m.def("sub",
          [](const arr16& a0, const arr16& a1, const arr16& a2, const arr32& a3,
             const arr16& b0, const arr16& b1, const arr16& b2, const arr32& b3) {
              return py_op(a0, a1, a2, a3, b0, b1, b2, b3, 2);
          },
          "Exact subtraction mod each rail.");
    m.def("mul",
          [](const arr16& a0, const arr16& a1, const arr16& a2, const arr32& a3,
             const arr16& b0, const arr16& b1, const arr16& b2, const arr32& b3) {
              return py_op(a0, a1, a2, a3, b0, b1, b2, b3, 1);
          },
          "Exact multiplication mod each rail.");
    m.def("div_",
          [](const arr16& a0, const arr16& a1, const arr16& a2, const arr32& a3,
             const arr16& b0, const arr16& b1, const arr16& b2, const arr32& b3) {
              return py_op(a0, a1, a2, a3, b0, b1, b2, b3, 3);
          },
          "Exact division. Divisor must be invertible on every rail.");
    m.def("sub_u64_io", &py_sub_u64_io,
          "Raw uint64[] -> encode + subtract + decode in one native call.");
    m.def("add_u64_io", &py_add_u64_io,
          "Raw uint64[] -> encode + add + decode in one native call.");
    m.def("mul_u64_io", &py_mul_u64_io,
          "Raw uint64[] -> encode + mul_u64 + decode in one native call.");
    m.def("fma_u64_io", &py_fma_u64_io,
          "Raw uint64[] -> encode + fma_u64 + decode in one native call.");
    m.def("affine_repeat_u64_io", &py_affine_repeat_u64_io,
          "Raw uint64[] -> encode + affine_repeat_u64 + decode in one native call.");
    m.def("add_u64_auto", &py_add_u64_auto,
          "Auto-dispatch add: fused single-thread for small N, OMP path for large N.");
    m.def("sub_u64_auto", &py_sub_u64_auto,
          "Auto-dispatch subtract: fused single-thread for small N, OMP path for large N.");
    m.def("sub_u64_io_omp", &py_sub_u64_io_omp,
          "OpenMP blocked fused path: raw uint64[] -> encode + subtract + decode.");
    m.def("add_u64_io_omp", &py_add_u64_io_omp,
          "OpenMP blocked fused path: raw uint64[] -> encode + add + decode.");
    m.def("mul_u64_io_omp", &py_mul_u64_io_omp,
          "OpenMP blocked fused path: raw uint64[] -> encode + mul + decode.");
    m.def("fma_u64_io_omp", &py_fma_u64_io_omp,
          "OpenMP blocked fused path: raw uint64[] -> encode + fma + decode.");
    m.def("affine_repeat_u64_io_omp", &py_affine_repeat_u64_io_omp,
          "OpenMP blocked fused path: raw uint64[] -> encode + affine-repeat + decode.");
    m.def("omp_max_threads", &py_omp_max_threads,
          "Return OpenMP max thread count.");
    m.def("omp_set_num_threads", &py_omp_set_num_threads,
          "Set OpenMP thread count for subsequent parallel regions.");
    m.def("omp_num_procs", &py_omp_num_procs,
          "Return number of processors visible to OpenMP.");
    m.def("mul_u64_auto", &py_mul_u64_auto,
          "Auto-dispatch multiply: fused single-thread for small N, OMP path for large N.");
    m.def("fma_u64_auto", &py_fma_u64_auto,
          "Auto-dispatch FMA: fused single-thread for small N, OMP path for large N.");
    m.def("affine_repeat_u64_auto", &py_affine_repeat_u64_auto,
          "Auto-dispatch affine-repeat: fused single-thread for very small N, OMP path otherwise.");
    m.def("fma", &py_fma,
          "Fused multiply-add: (a*b)+c in one kernel call.");
    m.def("affine_repeat", &py_affine_repeat,
          "Repeated affine loop in native code: x = x*m + k for 'iterations' steps.\n"
          "Inputs and outputs are encoded rails.");

    m.def("mul_u64", &py_mul_u64,
          "Single-step multiply with scalar multiplier broadcast inside C++.\n"
          "Input/output are encoded rails.");
    m.def("fma_u64", &py_fma_u64,
          "Single-step fused multiply-add with scalar multiplier/addend broadcast inside C++.\n"
          "Input/output are encoded rails.");
    m.def("affine_repeat_u64", &py_affine_repeat_u64,
          "Repeated affine loop with scalar multiplier/addend broadcast inside C++.\n"
          "This avoids loading N-sized constant arrays every iteration.\n"
          "Inputs/outputs are encoded rails.");

}
