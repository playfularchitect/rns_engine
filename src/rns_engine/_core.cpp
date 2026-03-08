/*
 * rns_engine/_core.cpp  —  v0.3.0
 *
 * 3-rail RNS exact integer arithmetic.
 * Optional AVX2 acceleration when compiled with AVX2 enabled.
 *
 * v0.3.0 highlights:
 *   - Full input length validation for op() and fma().
 *   - C-contiguous NumPy arrays enforced at the binding layer
 *     (non-contiguous inputs are safely copied/cast by pybind11).
 *   - Division now raises on non-invertible divisors instead of silently
 *     producing invalid results.
 *   - Version/doc cleanup for the v0.3.0 release.
 *
 * Moduli:  M0=127  M1=8191  M2=65536
 * Dynamic range:  127 × 8191 × 65536 = 68,174,282,752  (~36 bits)
 * Garner constants:  INV01=129  INV012=24705
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdint.h>
#include <stdexcept>
#include <string>

namespace py = pybind11;

// Require C-contiguous arrays; pybind11 will cast/copy as needed.
using arr16 = py::array_t<uint16_t, py::array::c_style | py::array::forcecast>;
using arr64 = py::array_t<uint64_t, py::array::c_style | py::array::forcecast>;

// ── Constants ──────────────────────────────────────────────────────────────
static constexpr uint32_t M0     = 127;
static constexpr uint32_t M1     = 8191;
static constexpr uint32_t M2     = 65536;
static constexpr uint64_t BM     = (uint64_t)M0 * M1 * M2;  // 68,174,282,752
static constexpr uint32_t INV01  = 129;    // inv(127) mod 8191
static constexpr uint32_t INV012 = 24705;  // inv(127*8191) mod 65536

// ── Validation helpers ─────────────────────────────────────────────────────
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

// ── Scalar helpers ─────────────────────────────────────────────────────────
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

// Garner CRT reconstruction: (r0 mod 127, r1 mod 8191, r2 mod 65536) → integer
static inline uint64_t garner(uint16_t r0, uint16_t r1, uint16_t r2) {
    uint32_t t0 = r0;
    uint32_t t1 = (uint32_t)(
        ((int64_t)r1 - (int64_t)(t0 % M1) + M1) % M1
        * (uint64_t)INV01 % M1);
    uint64_t base = t0 + (uint64_t)t1 * M0;
    int64_t d = ((int64_t)r2 - (int64_t)(base % M2) + (int64_t)M2 * 2) % (int64_t)M2;
    return base + (uint64_t)(d * (uint64_t)INV012 % M2) * (uint64_t)M0 * M1;
}

static void require_divisors_invertible(const arr16& b0,
                                        const arr16& b1,
                                        const arr16& b2) {
    require_same_len(b0, "b0", b1, "b1");
    require_same_len(b0, "b0", b2, "b2");

    auto rb0 = b0.unchecked<1>();
    auto rb1 = b1.unchecked<1>();
    auto rb2 = b2.unchecked<1>();
    int64_t n = (int64_t)b0.shape(0);

    for (int64_t i = 0; i < n; ++i) {
        // Mod 127 and mod 8191 are prime: only residue 0 is non-invertible.
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
        // Mod 65536 = 2^16: only odd residues are invertible.
        if ((rb2(i) & 1u) == 0u) {
            throw std::invalid_argument(
                "division error: b2 contains a non-invertible residue mod 65536 at index " +
                std::to_string(i));
        }
    }
}

// ── Scalar kernel ──────────────────────────────────────────────────────────
// op: 0=add  1=mul  2=sub  3=div  4=fma (a*b+c using a,b,c arrays)
static void kernel_scalar(
    const uint16_t* a0, const uint16_t* a1, const uint16_t* a2,
    const uint16_t* b0, const uint16_t* b1, const uint16_t* b2,
    uint16_t* r0, uint16_t* r1, uint16_t* r2,
    int64_t n, int op,
    // fma extra operand (c)
    const uint16_t* c0 = nullptr,
    const uint16_t* c1 = nullptr,
    const uint16_t* c2 = nullptr) {
    for (int64_t i = 0; i < n; i++) {
        switch (op) {
            case 0: // add
                r0[i] = r127s(a0[i] + b0[i]);
                r1[i] = r8191s(a1[i] + b1[i]);
                r2[i] = (uint16_t)((a2[i] + b2[i]) & 0xFFFF);
                break;
            case 1: // mul
                r0[i] = r127s((uint32_t)a0[i] * b0[i]);
                r1[i] = r8191s((uint32_t)a1[i] * b1[i]);
                r2[i] = (uint16_t)((uint32_t)a2[i] * b2[i]);
                break;
            case 2: // sub
                r0[i] = r127s(M0 + a0[i] - r127s(b0[i]));
                r1[i] = r8191s(M1 + a1[i] - r8191s(b1[i]));
                r2[i] = (uint16_t)((M2 + a2[i] - (b2[i] % M2)) & 0xFFFF);
                break;
            case 3: // div
                r0[i] = r127s((uint32_t)a0[i] * inv_s(b0[i], M0));
                r1[i] = r8191s((uint32_t)a1[i] * inv_s(b1[i], M1));
                r2[i] = (uint16_t)(((uint32_t)a2[i] * inv_s(b2[i], M2)) & 0xFFFF);
                break;
            case 4: // fma: a*b+c
                r0[i] = r127s((uint32_t)a0[i] * b0[i] + c0[i]);
                r1[i] = r8191s((uint32_t)a1[i] * b1[i] + c1[i]);
                r2[i] = (uint16_t)((uint32_t)a2[i] * b2[i] + c2[i]);
                break;
            default:
                throw std::invalid_argument("invalid opcode");
        }
    }
}

// ── AVX2 kernel ────────────────────────────────────────────────────────────
#if defined(__AVX2__)
#include <immintrin.h>
#define HAVE_AVX2 1
#define L 16  // uint16 lanes per 256-bit register

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

// Mersenne folding: x mod 127
static inline vec r127v(vec x) {
    vec t = Va(Vn(x, V1(0x7F)), Vh(x, 7));
    t = Va(Vn(t, V1(0x7F)), Vh(t, 7));
    return Vs(t, Vn(V1(127), Ve(t, V1(127))));
}

// Mersenne folding: x mod 8191
static inline vec r8191v(vec x) {
    vec t = Va(Vn(x, V1(0x1FFF)), Vh(x, 13));
    t = Va(Vn(t, V1(0x1FFF)), Vh(t, 13));
    return Vs(t, Vn(V1(8191), Ve(t, V1(8191))));
}

// 8191 multiplication: expand to 32-bit, multiply, reduce, pack back to 16-bit
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

// 127 multiplication: 126*126=15876, so uint16 intermediate is enough
static inline vec mul127v(vec a, vec b) {
    return r127v(Vm(a, b));
}

static void kernel_avx2(
    const uint16_t* a0, const uint16_t* a1, const uint16_t* a2,
    const uint16_t* b0, const uint16_t* b1, const uint16_t* b2,
    uint16_t* r0, uint16_t* r1, uint16_t* r2,
    int64_t n, int op,
    const uint16_t* c0 = nullptr,
    const uint16_t* c1 = nullptr,
    const uint16_t* c2 = nullptr) {
    // Division falls back to scalar (needs modular inverse per element)
    if (op == 3) {
        kernel_scalar(a0, a1, a2, b0, b1, b2, r0, r1, r2, n, op);
        return;
    }

    const int64_t full = (n / L) * L;

    for (int64_t i = 0; i < full; i += L) {
        vec va0 = Vload(a0 + i), vb0 = Vload(b0 + i);
        vec va1 = Vload(a1 + i), vb1 = Vload(b1 + i);
        vec va2 = Vload(a2 + i), vb2 = Vload(b2 + i);
        vec vr0, vr1, vr2;

        switch (op) {
            case 0: // add
                vr0 = r127v(Va(va0, vb0));
                vr1 = r8191v(Va(va1, vb1));
                vr2 = Va(va2, vb2);  // mod 65536: free uint16 wrap
                break;

            case 1: // mul
                vr0 = mul127v(va0, vb0);
                vr1 = mul8191v(va1, vb1);
                vr2 = Vm(va2, vb2);  // mod 65536: free uint16 wrap
                break;

            case 2: // sub
                vr0 = r127v(Va(va0, Vs(V1(127), r127v(vb0))));
                vr1 = r8191v(Va(va1, Vs(V1(8191), r8191v(vb1))));
                vr2 = Vs(va2, vb2);  // mod 65536: free uint16 wrap
                break;

            case 4: { // fma
                vec vc0 = Vload(c0 + i);
                vec vc1 = Vload(c1 + i);
                vec vc2 = Vload(c2 + i);
                vr0 = r127v(Va(mul127v(va0, vb0), vc0));
                vr1 = r8191v(Va(mul8191v(va1, vb1), vc1));
                vr2 = Va(Vm(va2, vb2), vc2);
                break;
            }

            default:
                throw std::invalid_argument("invalid opcode");
        }

        Vstore(r0 + i, vr0);
        Vstore(r1 + i, vr1);
        Vstore(r2 + i, vr2);
    }

    // Handle remainder with scalar
    if (full < n) {
        kernel_scalar(
            a0 + full, a1 + full, a2 + full,
            b0 + full, b1 + full, b2 + full,
            r0 + full, r1 + full, r2 + full,
            n - full, op,
            c0 ? c0 + full : nullptr,
            c1 ? c1 + full : nullptr,
            c2 ? c2 + full : nullptr);
    }
}

#else
#define HAVE_AVX2 0
#define L 16
#endif

// ── Dispatch ───────────────────────────────────────────────────────────────
static void kernel(
    const uint16_t* a0, const uint16_t* a1, const uint16_t* a2,
    const uint16_t* b0, const uint16_t* b1, const uint16_t* b2,
    uint16_t* r0, uint16_t* r1, uint16_t* r2,
    int64_t n, int op,
    const uint16_t* c0 = nullptr,
    const uint16_t* c1 = nullptr,
    const uint16_t* c2 = nullptr) {
#if HAVE_AVX2
    kernel_avx2(a0, a1, a2, b0, b1, b2, r0, r1, r2, n, op, c0, c1, c2);
#else
    kernel_scalar(a0, a1, a2, b0, b1, b2, r0, r1, r2, n, op, c0, c1, c2);
#endif
}

// ── Python interface ───────────────────────────────────────────────────────
py::tuple py_encode(const arr64& x_in) {
    int64_t n = require_1d_len(x_in, "x");
    auto x = x_in.unchecked<1>();

    arr16 o0({n}), o1({n}), o2({n});
    auto p0 = o0.mutable_unchecked<1>();
    auto p1 = o1.mutable_unchecked<1>();
    auto p2 = o2.mutable_unchecked<1>();

    for (int64_t i = 0; i < n; i++) {
        uint64_t v = x(i) % BM;
        p0(i) = (uint16_t)(v % M0);
        p1(i) = (uint16_t)(v % M1);
        p2(i) = (uint16_t)(v % M2);
    }
    return py::make_tuple(o0, o1, o2);
}

arr64 py_decode(const arr16& r0_, const arr16& r1_, const arr16& r2_) {
    int64_t n = require_1d_len(r0_, "r0");
    require_same_len(r0_, "r0", r1_, "r1");
    require_same_len(r0_, "r0", r2_, "r2");

    arr64 out({n});
    auto r0 = r0_.unchecked<1>();
    auto r1 = r1_.unchecked<1>();
    auto r2 = r2_.unchecked<1>();
    auto o  = out.mutable_unchecked<1>();

    for (int64_t i = 0; i < n; i++) {
        o(i) = garner(r0(i), r1(i), r2(i));
    }
    return out;
}

py::tuple py_op(const arr16& a0, const arr16& a1, const arr16& a2,
                const arr16& b0, const arr16& b1, const arr16& b2,
                int opcode) {
    if (opcode < 0 || opcode > 3) {
        throw std::invalid_argument("opcode must be 0=add 1=mul 2=sub 3=div");
    }

    int64_t n = require_1d_len(a0, "a0");
    require_same_len(a0, "a0", a1, "a1");
    require_same_len(a0, "a0", a2, "a2");
    require_same_len(a0, "a0", b0, "b0");
    require_same_len(a0, "a0", b1, "b1");
    require_same_len(a0, "a0", b2, "b2");

    if (opcode == 3) {
        require_divisors_invertible(b0, b1, b2);
    }

    arr16 r0({n}), r1({n}), r2({n});
    kernel(a0.data(), a1.data(), a2.data(),
           b0.data(), b1.data(), b2.data(),
           r0.mutable_data(), r1.mutable_data(), r2.mutable_data(),
           n, opcode);
    return py::make_tuple(r0, r1, r2);
}

// fma: compute a*b + c in a single kernel call
py::tuple py_fma(const arr16& a0, const arr16& a1, const arr16& a2,
                 const arr16& b0, const arr16& b1, const arr16& b2,
                 const arr16& c0, const arr16& c1, const arr16& c2) {
    int64_t n = require_1d_len(a0, "a0");
    require_same_len(a0, "a0", a1, "a1");
    require_same_len(a0, "a0", a2, "a2");
    require_same_len(a0, "a0", b0, "b0");
    require_same_len(a0, "a0", b1, "b1");
    require_same_len(a0, "a0", b2, "b2");
    require_same_len(a0, "a0", c0, "c0");
    require_same_len(a0, "a0", c1, "c1");
    require_same_len(a0, "a0", c2, "c2");

    arr16 r0({n}), r1({n}), r2({n});
    kernel(a0.data(), a1.data(), a2.data(),
           b0.data(), b1.data(), b2.data(),
           r0.mutable_data(), r1.mutable_data(), r2.mutable_data(),
           n, 4,
           c0.data(), c1.data(), c2.data());
    return py::make_tuple(r0, r1, r2);
}

// ── Module ─────────────────────────────────────────────────────────────────
PYBIND11_MODULE(_core, m) {
    m.doc() =
        "rns_engine._core v0.3.0\n"
        "3-rail exact integer arithmetic with optional AVX2 acceleration.\n"
        "All rail arrays are uint16.\n"
        "Moduli: 127 x 8191 x 65536 = 68,174,282,752 (~36-bit dynamic range).\n"
        "HAS_AVX2 indicates whether this extension was compiled with AVX2.";

    m.attr("M")        = (uint64_t)BM;
    m.attr("M0")       = (uint32_t)M0;
    m.attr("M1")       = (uint32_t)M1;
    m.attr("M2")       = (uint32_t)M2;
    m.attr("HAS_AVX2") = (bool)HAVE_AVX2;

    m.def("encode", &py_encode,
          "uint64[] -> (r0:u16, r1:u16, r2:u16) residue arrays.");
    m.def("decode", &py_decode,
          "(r0:u16, r1:u16, r2:u16) -> uint64[] via Garner CRT.");
    m.def("op", &py_op,
          "opcode: 0=add 1=mul 2=sub 3=div");
    m.def("add",
          [](const arr16& a0, const arr16& a1, const arr16& a2,
             const arr16& b0, const arr16& b1, const arr16& b2) {
              return py_op(a0, a1, a2, b0, b1, b2, 0);
          },
          "Exact addition mod each rail.");
    m.def("sub",
          [](const arr16& a0, const arr16& a1, const arr16& a2,
             const arr16& b0, const arr16& b1, const arr16& b2) {
              return py_op(a0, a1, a2, b0, b1, b2, 2);
          },
          "Exact subtraction mod each rail.");
    m.def("mul",
          [](const arr16& a0, const arr16& a1, const arr16& a2,
             const arr16& b0, const arr16& b1, const arr16& b2) {
              return py_op(a0, a1, a2, b0, b1, b2, 1);
          },
          "Exact multiplication mod each rail.");
    m.def("div_",
          [](const arr16& a0, const arr16& a1, const arr16& a2,
             const arr16& b0, const arr16& b1, const arr16& b2) {
              return py_op(a0, a1, a2, b0, b1, b2, 3);
          },
          "Exact division. Divisor must be invertible on every rail.");
    m.def("fma", &py_fma,
          "Fused multiply-add: (a*b)+c in one kernel call.\n"
          "Equivalent to add(*mul(a,b), c) but faster.");
}
