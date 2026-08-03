#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdint.h>
#include <stdexcept>
#include <string>
#include <cstring>

namespace py = pybind11;

using arr_i32 = py::array_t<int32_t, py::array::c_style>;
using arr16 = py::array_t<uint16_t, py::array::c_style>;
using arr32 = py::array_t<uint32_t, py::array::c_style>;
using arr64 = py::array_t<uint64_t, py::array::c_style | py::array::forcecast>;

static constexpr uint32_t M0      = 127;
static constexpr uint32_t M1      = 8191;
static constexpr uint32_t M2      = 65536;
static constexpr uint32_t M3      = 524287;
static constexpr uint32_t M3_MASK = 0x7FFFFu;
static constexpr uint64_t BM      = (uint64_t)M0 * M1 * M2 * M3;

static int64_t require_1d_len(const py::array& a, const char* name) {
    if (a.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1D array");
    }
    return (int64_t)a.shape(0);
}

static inline uint16_t r127s(uint32_t x) {
    x = (x & 0x7F) + (x >> 7);
    x = (x & 0x7F) + (x >> 7);
    return x >= M0 ? x - M0 : (uint16_t)x;
}

static inline uint16_t r8191s(uint32_t x) {
    x = (x & 0x1FFF) + (x >> 13);
    x = (x & 0x1FFF) + (x >> 13);
    return x >= M1 ? x - M1 : (uint16_t)x;
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

py::tuple weighted_int32(const arr_i32& partials, const arr64& weights) {
    if (partials.ndim() != 2) {
        throw std::invalid_argument("partials must be a 2D int32 array");
    }

    const int64_t terms = (int64_t)partials.shape(0);
    const int64_t n = (int64_t)partials.shape(1);
    const int64_t weight_count = require_1d_len(weights, "weights");
    if (weight_count != terms) {
        throw std::invalid_argument(
            "weights length must match partial term count (" +
            std::to_string(weight_count) + " != " + std::to_string(terms) + ")");
    }

    arr16 o0({n}), o1({n}), o2({n});
    arr32 o3({n});
    arr64 bounds({terms});

    uint16_t* p0 = o0.mutable_data();
    uint16_t* p1 = o1.mutable_data();
    uint16_t* p2 = o2.mutable_data();
    uint32_t* p3 = o3.mutable_data();
    uint64_t* pb = bounds.mutable_data();

    if (n > 0) {
        std::memset(p0, 0, (size_t)n * sizeof(uint16_t));
        std::memset(p1, 0, (size_t)n * sizeof(uint16_t));
        std::memset(p2, 0, (size_t)n * sizeof(uint16_t));
        std::memset(p3, 0, (size_t)n * sizeof(uint32_t));
    }
    if (terms > 0) {
        std::memset(pb, 0, (size_t)terms * sizeof(uint64_t));
    }

    const int32_t* x = partials.data();
    const uint64_t* w = weights.data();

    {
        py::gil_scoped_release release;
        for (int64_t term = 0; term < terms; ++term) {
            const uint64_t weight = w[term] % BM;
            const uint16_t w0 = (uint16_t)(weight % M0);
            const uint16_t w1 = (uint16_t)(weight % M1);
            const uint16_t w2 = (uint16_t)weight;
            const uint32_t w3 = mod524287_u64(weight);
            const int32_t* row = x + term * n;
            uint64_t max_abs = 0;

            for (int64_t i = 0; i < n; ++i) {
                const int32_t signed_value = row[i];
                const uint64_t magnitude = signed_value < 0
                    ? (uint64_t)(-(int64_t)signed_value)
                    : (uint64_t)signed_value;
                if (magnitude > max_abs) max_abs = magnitude;

                if (weight == 0) continue;

                const uint64_t canonical = signed_value < 0
                    ? (uint64_t)((int64_t)BM + (int64_t)signed_value)
                    : (uint64_t)signed_value;
                const uint16_t x0 = (uint16_t)(canonical % M0);
                const uint16_t x1 = (uint16_t)(canonical % M1);
                const uint16_t x2 = (uint16_t)canonical;
                const uint32_t x3 = mod524287_u64(canonical);

                p0[i] = r127s((uint32_t)p0[i] + (uint32_t)x0 * w0);
                p1[i] = r8191s((uint32_t)p1[i] + (uint32_t)x1 * w1);
                p2[i] = (uint16_t)((uint32_t)p2[i] + (uint32_t)x2 * w2);
                p3[i] = fma524287s(x3, w3, p3[i]);
            }
            pb[term] = max_abs;
        }
    }

    return py::make_tuple(o0, o1, o2, o3, bounds);
}

PYBIND11_MODULE(_weighted, m) {
    m.doc() = "Fused weighted signed INT32 accumulation for rns_engine.";
    m.def(
        "weighted_int32",
        &weighted_int32,
        "Fused signed int32 weighting and four-rail accumulation with per-term bounds.");
}
