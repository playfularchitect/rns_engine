/*
 * rns_engine/_core.cpp
 *
 * 3-rail Residue Number System exact integer arithmetic.
 * Moduli: 127 × 8191 × 65536  →  dynamic range [0, 68,174,282,752)
 *
 * AVX2 fast path auto-selected at compile time.
 * Scalar fallback used on non-x86 or older hardware.
 *
 * Operations: encode, decode, add, sub, mul, div
 * All operations are exact — no floating point, no approximation.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdint.h>
#include <stdexcept>

namespace py = pybind11;
using arr16 = py::array_t<uint16_t>;
using arr32 = py::array_t<uint32_t>;
using arr64 = py::array_t<uint64_t>;

// ── constants ─────────────────────────────────────────────────────────────
static constexpr uint32_t M0    = 127;
static constexpr uint32_t M1    = 8191;
static constexpr uint32_t M2    = 65536;
static constexpr uint64_t BM    = (uint64_t)M0 * M1 * M2;   // 68,174,282,752
static constexpr uint32_t INV01 = 129;    // inv(127) mod 8191
static constexpr uint32_t INV012= 24705;  // inv(127*8191) mod 65536
#define L 16

// ── scalar helpers ────────────────────────────────────────────────────────
static inline uint16_t r127s(uint32_t x) {
    x = (x & 0x7F) + (x >> 7);
    x = (x & 0x7F) + (x >> 7);
    return x == 127 ? 0 : (uint16_t)x;
}
static inline uint32_t r8191s(uint64_t x) {
    x = (x & 0x1FFF) + (x >> 13);
    x = (x & 0x1FFF) + (x >> 13);
    return x == 8191 ? 0 : (uint32_t)x;
}
static int64_t egcd(int64_t a, int64_t b, int64_t *x, int64_t *y) {
    if (!a) { *x = 0; *y = 1; return b; }
    int64_t x1, y1;
    int64_t g = egcd(b % a, a, &x1, &y1);
    *x = y1 - (b / a) * x1; *y = x1;
    return g;
}
static uint32_t inv_s(int64_t a, int64_t m) {
    a = ((a % m) + m) % m;
    if (!a) return 0;
    int64_t x, y;
    if (egcd(a, m, &x, &y) != 1) return 0;
    return (uint32_t)(((x % m) + m) % m);
}
static inline uint64_t garner(uint16_t r0, uint32_t r1, uint16_t r2) {
    uint32_t t0 = r0;
    uint32_t t1 = (uint32_t)(
        ((int64_t)r1 - (int64_t)(t0 % 8191) + 8191) % 8191
        * (uint64_t)INV01 % 8191);
    uint64_t base = t0 + (uint64_t)t1 * 127;
    int64_t d = ((int64_t)r2 - (int64_t)(base % 65536) + 131072LL) % 65536;
    return base + (uint64_t)(d * (uint64_t)INV012 % 65536) * 127ULL * 8191ULL;
}

// ── scalar kernel (all platforms) ─────────────────────────────────────────
static void kernel_scalar(
    const uint16_t *a0, const uint32_t *a1, const uint16_t *a2,
    const uint16_t *b0, const uint32_t *b1, const uint16_t *b2,
    uint16_t *r0,       uint32_t *r1,       uint16_t *r2,
    ssize_t n, int op)
{
    for (ssize_t i = 0; i < n; i++) {
        switch (op) {
        case 0: // ADD
            r0[i] = r127s(a0[i] + b0[i]);
            r1[i] = r8191s((uint64_t)a1[i] + b1[i]);
            r2[i] = (uint16_t)((a2[i] + b2[i]) & 0xFFFF);
            break;
        case 1: // MUL
            r0[i] = r127s((uint32_t)a0[i] * b0[i]);
            r1[i] = r8191s((uint64_t)a1[i] * b1[i]);
            r2[i] = (uint16_t)((uint32_t)a2[i] * b2[i]);
            break;
        case 2: // SUB
            r0[i] = r127s(127  + a0[i] - b0[i] % 127);
            r1[i] = r8191s(8191 + (uint64_t)a1[i] - b1[i] % 8191);
            r2[i] = (uint16_t)((65536 + a2[i] - b2[i] % 65536) & 0xFFFF);
            break;
        case 3: // DIV
            r0[i] = r127s ((uint32_t)a0[i] * inv_s(b0[i], 127));
            r1[i] = r8191s((uint64_t)a1[i] * inv_s(b1[i], 8191));
            r2[i] = (uint16_t)(((uint32_t)a2[i] * inv_s(b2[i], 65536)) & 0xFFFF);
            break;
        }
    }
}

// ── AVX2 fast path ────────────────────────────────────────────────────────
#if defined(__AVX2__)
#include <immintrin.h>
using vec16 = __m256i;

static inline vec16 V1(int x)    { return _mm256_set1_epi16((short)x); }
static inline vec16 Va(vec16 a, vec16 b) { return _mm256_add_epi16(a, b); }
static inline vec16 Vs(vec16 a, vec16 b) { return _mm256_sub_epi16(a, b); }
static inline vec16 Vm(vec16 a, vec16 b) { return _mm256_mullo_epi16(a, b); }
static inline vec16 Vn(vec16 a, vec16 b) { return _mm256_and_si256(a, b); }
static inline vec16 Vh(vec16 a, int s)   { return _mm256_srli_epi16(a, s); }
static inline vec16 Ve(vec16 a, vec16 b) { return _mm256_cmpeq_epi16(a, b); }

static inline vec16 r127v(vec16 x) {
    vec16 t = Va(Vn(x, V1(0x7F)), Vh(x, 7));
    t = Va(Vn(t, V1(0x7F)), Vh(t, 7));
    return Vs(t, Vn(V1(127), Ve(t, V1(127))));
}
static inline vec16 r8191v(vec16 x) {
    vec16 t = Va(Vn(x, V1(0x1FFF)), Vh(x, 13));
    return Vs(t, Vn(V1(8191), Ve(t, V1(8191))));
}
static inline vec16 mul8191v(vec16 a, vec16 b) {
    __m256i mk = _mm256_set1_epi32(0x1FFF);
    auto f = [&](__m256i x) {
        x = _mm256_add_epi32(_mm256_and_si256(x, mk), _mm256_srli_epi32(x, 13));
        x = _mm256_add_epi32(_mm256_and_si256(x, mk), _mm256_srli_epi32(x, 13));
        return _mm256_sub_epi32(x, _mm256_and_si256(
            _mm256_cmpeq_epi32(x, _mm256_set1_epi32(8191)),
            _mm256_set1_epi32(8191)));
    };
    __m256i pl = _mm256_mullo_epi32(
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(a)),
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b)));
    __m256i ph = _mm256_mullo_epi32(
        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(a, 1)),
        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(b, 1)));
    return _mm256_permute4x64_epi64(
        _mm256_packus_epi32(f(pl), f(ph)), 0b11011000);
}

static void kernel_avx2(
    const uint16_t *a0, const uint32_t *a1, const uint16_t *a2,
    const uint16_t *b0, const uint32_t *b1, const uint16_t *b2,
    uint16_t *r0,       uint32_t *r1,       uint16_t *r2,
    ssize_t n, int op)
{
    // DIV has no vectorized path — fall back to scalar
    if (op == 3) { kernel_scalar(a0,a1,a2,b0,b1,b2,r0,r1,r2,n,op); return; }

    ssize_t full = (n / L) * L;
    for (ssize_t base = 0; base < full; base += L) {
        alignas(32) int16_t ta0[L],tb0[L],ta1[L],tb1[L],ta2[L],tb2[L];
        for (int l = 0; l < L; l++) {
            ta0[l]=(int16_t)a0[base+l]; tb0[l]=(int16_t)b0[base+l];
            ta1[l]=(int16_t)a1[base+l]; tb1[l]=(int16_t)b1[base+l];
            ta2[l]=(int16_t)a2[base+l]; tb2[l]=(int16_t)b2[base+l];
        }
        vec16 va0=_mm256_load_si256((vec16*)ta0), vb0=_mm256_load_si256((vec16*)tb0);
        vec16 va1=_mm256_load_si256((vec16*)ta1), vb1=_mm256_load_si256((vec16*)tb1);
        vec16 va2=_mm256_load_si256((vec16*)ta2), vb2=_mm256_load_si256((vec16*)tb2);
        vec16 vr0, vr1, vr2;
        if (op == 0) {
            vr0 = r127v(Va(va0, vb0));
            vr1 = r8191v(Va(va1, vb1));
            vr2 = Va(va2, vb2);
        } else if (op == 1) {
            vr0 = r127v(Vm(va0, vb0));
            vr1 = mul8191v(va1, vb1);
            vr2 = Vm(va2, vb2);
        } else { // SUB
            vr0 = r127v (Va(va0, r127v (Vs(V1(127),  vb0))));
            vr1 = r8191v(Va(va1, r8191v(Vs(V1(8191), vb1))));
            vr2 = Va(va2, Vs(V1(0), vb2));
        }
        alignas(32) int16_t tr0[L], tr1[L], tr2[L];
        _mm256_store_si256((vec16*)tr0, vr0);
        _mm256_store_si256((vec16*)tr1, vr1);
        _mm256_store_si256((vec16*)tr2, vr2);
        for (int l = 0; l < L; l++) {
            r0[base+l] = (uint16_t)tr0[l];
            r1[base+l] = (uint16_t)tr1[l];
            r2[base+l] = (uint16_t)tr2[l];
        }
    }
    // scalar tail
    kernel_scalar(a0+full, a1+full, a2+full,
                  b0+full, b1+full, b2+full,
                  r0+full, r1+full, r2+full,
                  n-full, op);
}

static bool HAS_AVX2 = true;
#else
static bool HAS_AVX2 = false;
#endif

// ── dispatch ──────────────────────────────────────────────────────────────
static void kernel(
    const uint16_t *a0, const uint32_t *a1, const uint16_t *a2,
    const uint16_t *b0, const uint32_t *b1, const uint16_t *b2,
    uint16_t *r0,       uint32_t *r1,       uint16_t *r2,
    ssize_t n, int op)
{
#if defined(__AVX2__)
    kernel_avx2(a0,a1,a2, b0,b1,b2, r0,r1,r2, n, op);
#else
    kernel_scalar(a0,a1,a2, b0,b1,b2, r0,r1,r2, n, op);
#endif
}

// ── Python-facing functions ───────────────────────────────────────────────
py::tuple py_encode(arr64 x_in) {
    auto x = x_in.unchecked<1>();
    ssize_t n = x_in.shape(0);
    arr16 o0({n}); arr32 o1({n}); arr16 o2({n});
    auto p0 = o0.mutable_unchecked<1>();
    auto p1 = o1.mutable_unchecked<1>();
    auto p2 = o2.mutable_unchecked<1>();
    for (ssize_t i = 0; i < n; i++) {
        uint64_t v = x(i) % BM;
        p0(i) = (uint16_t)(v % 127);
        p1(i) = (uint32_t)(v % 8191);
        p2(i) = (uint16_t)(v % 65536);
    }
    return py::make_tuple(o0, o1, o2);
}

arr64 py_decode(arr16 r0_, arr32 r1_, arr16 r2_) {
    ssize_t n = r0_.shape(0);
    if (r1_.shape(0) != n || r2_.shape(0) != n)
        throw std::invalid_argument("array length mismatch");
    arr64 out({n});
    auto r0 = r0_.unchecked<1>(), r2 = r2_.unchecked<1>();
    auto r1 = r1_.unchecked<1>();
    auto o  = out.mutable_unchecked<1>();
    for (ssize_t i = 0; i < n; i++) o(i) = garner(r0(i), r1(i), r2(i));
    return out;
}

py::tuple py_op(arr16 a0, arr32 a1, arr16 a2,
                arr16 b0, arr32 b1, arr16 b2, int opcode) {
    if (opcode < 0 || opcode > 3)
        throw std::invalid_argument("opcode must be 0=add 1=mul 2=sub 3=div");
    ssize_t n = a0.shape(0);
    arr16 r0({n}); arr32 r1({n}); arr16 r2({n});
    kernel(a0.data(), a1.data(), a2.data(),
           b0.data(), b1.data(), b2.data(),
           r0.mutable_data(), r1.mutable_data(), r2.mutable_data(),
           n, opcode);
    return py::make_tuple(r0, r1, r2);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = R"doc(
        rns_engine._core — AVX2-accelerated 3-rail RNS exact integer arithmetic.

        Dynamic range: [0, 68,174,282,752)  =  127 × 8191 × 65536
        All operations are exact. No floating point. No approximation.

        Division requires b to be coprime to all three moduli
        (odd mod 65536, nonzero mod 127, nonzero mod 8191).
    )doc";

    m.attr("M")      = (uint64_t)BM;
    m.attr("M0")     = (uint32_t)M0;
    m.attr("M1")     = (uint32_t)M1;
    m.attr("M2")     = (uint32_t)M2;
    m.attr("HAS_AVX2") = HAS_AVX2;

    m.def("encode", &py_encode,
          "encode(x: uint64 array) -> (r0: uint16, r1: uint32, r2: uint16)\n"
          "Convert integers into RNS residue representation.");

    m.def("decode", &py_decode,
          "decode(r0, r1, r2) -> uint64 array\n"
          "Reconstruct integers from residues via Garner's algorithm.");

    m.def("op", &py_op,
          "op(r0a,r1a,r2a, r0b,r1b,r2b, opcode) -> (r0,r1,r2)\n"
          "opcode: 0=add  1=mul  2=sub  3=div\n"
          "Operate on residue-encoded arrays directly.");

    m.def("add",  [](arr16 a0,arr32 a1,arr16 a2, arr16 b0,arr32 b1,arr16 b2)
          { return py_op(a0,a1,a2,b0,b1,b2,0); },
          "Exact addition in RNS space.");
    m.def("sub",  [](arr16 a0,arr32 a1,arr16 a2, arr16 b0,arr32 b1,arr16 b2)
          { return py_op(a0,a1,a2,b0,b1,b2,2); },
          "Exact subtraction in RNS space.");
    m.def("mul",  [](arr16 a0,arr32 a1,arr16 a2, arr16 b0,arr32 b1,arr16 b2)
          { return py_op(a0,a1,a2,b0,b1,b2,1); },
          "Exact multiplication in RNS space.");
    m.def("div_", [](arr16 a0,arr32 a1,arr16 a2, arr16 b0,arr32 b1,arr16 b2)
          { return py_op(a0,a1,a2,b0,b1,b2,3); },
          "Exact division in RNS space.\n"
          "b must be coprime to all moduli (odd mod 65536, nonzero mod 127 and 8191).");
}
