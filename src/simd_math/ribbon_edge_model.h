#pragma once
// ============================================================================
// ribbon_edge_model.h
//
// The edge loop of the client's ribbon emitter update, sub_980090, from
// 0x00980341 to 0x009804EA. Shared by the module and by the offline harness
// that runs the client's own instruction sequence against it, so there is one
// copy of the arithmetic and the harness measures that copy.
//
// What the client does for every edge from head to tail round the ring:
//
//   v28      = ((age * 2 + dt) * k) * dt                 x87 at 53 bits
//   edge[2]  = float(edge[2] + v28)
//   edge[8]  = float(v28 + edge[8])
//   bounds   = min/max of edge[0..2] and edge[6..8], strict compares
//   age      = float(dt + age)
//   edge[4]  = edge[10] = float(((age * c5C) * c58) + c70)
//   edge[5]  = obj[6C], edge[11] = obj[74]               through the x87 stack
//
// The client does the twelve compares as fcomp / fnstsw / test / branch on
// values it has just stored, one scalar at a time. Here they are two packed
// min and two packed max an edge. minps(x, m) answers x when x < m and m
// otherwise, which is the client's "replace when strictly smaller" including
// for equal values, signed zeros and NaN (an unordered compare leaves the
// bound alone in both). The arithmetic is IEEE double with one rounding a
// step, which is what the x87 does at a 53-bit precision control.
//
// Not the same bits in one case: a NaN that reaches an addition together with
// another NaN. The x87 keeps the larger significand, SSE keeps the first
// operand. Both answers are a NaN, and an edge with a NaN in its position is
// garbage in the client as well.
// ============================================================================

#include <cstdint>
#include <emmintrin.h>

namespace RibbonEdgeModel {

// Offsets into the emitter object, as the client reads them.
constexpr unsigned kCapacity = 0x08;   // ring size, in edges
constexpr unsigned kAges     = 0x0C;   // float[capacity]
constexpr unsigned kTail     = 0x14;
constexpr unsigned kHead     = 0x18;    // the client's loop runs from here
constexpr unsigned kEdges    = 0x40;   // 48 bytes an edge
constexpr unsigned kScaleA   = 0x58;
constexpr unsigned kScaleB   = 0x5C;
constexpr unsigned kPassA    = 0x6C;
constexpr unsigned kBias     = 0x70;
constexpr unsigned kPassB    = 0x74;
constexpr unsigned kK        = 0x17C;
constexpr unsigned kBounds   = 0xF4;   // min x y z, then max x y z
constexpr unsigned kObjSpan  = 0x200;  // everything above lies inside this

constexpr uint32_t kMaxCapacity = 1u << 20;

enum Result { kDeclined = 0, kEmpty = 1, kDone = 2 };

template <class T> inline T& At(unsigned char* o, unsigned off) {
    return *reinterpret_cast<T*>(o + off);
}

// A float through the x87 stack and back. Only a signalling NaN changes, and
// it is quieted, which the double round trip reproduces.
inline float ThroughX87(float f) {
    return _mm_cvtss_f32(_mm_cvtsd_ss(_mm_setzero_ps(),
                                      _mm_cvtss_sd(_mm_setzero_pd(), _mm_set_ss(f))));
}

// obj is the emitter, dt the double the client holds on its x87 stack at the
// top of the loop. Declines, touching nothing, when the ring or its buffers do
// not have the shape the loop below assumes.
inline Result Run(unsigned char* obj, double dt) {
    const uint32_t head = At<uint32_t>(obj, kHead);
    const uint32_t tail = At<uint32_t>(obj, kTail);
    if (head == tail) return kEmpty;

    // The x87 keeps a 15-bit exponent, a double 11. Between 1e-30 and 1e30 no
    // product of the loop (three floats and two of dt) can leave the double
    // range, so every step rounds where the client's does. Outside it an
    // underflow lands on a different signed zero (found by the harness at
    // dt = 1e-300), so the client's own loop runs. A frame time is never there.
    const double adt = dt < 0.0 ? -dt : dt;
    if (!(dt == 0.0 || (adt >= 1e-30 && adt <= 1e30))) return kDeclined;

    const uint32_t cap = At<uint32_t>(obj, kCapacity);
    float* const ages = At<float*>(obj, kAges);
    float* const seg  = At<float*>(obj, kEdges);
    if (cap == 0 || cap > kMaxCapacity || head >= cap || tail >= cap) return kDeclined;
    if (!ages || !seg) return kDeclined;

    // The loop keeps the object's constants in registers and writes the bounds
    // once. That is only the same thing when the two buffers are not inside
    // the object and do not overlap each other.
    const uintptr_t o0 = (uintptr_t)obj, o1 = o0 + kObjSpan;
    const uintptr_t a0 = (uintptr_t)ages, a1 = a0 + (uintptr_t)cap * 4u;
    const uintptr_t s0 = (uintptr_t)seg,  s1 = s0 + (uintptr_t)cap * 48u;
    if (a1 < a0 || s1 < s0) return kDeclined;
    if (a0 < o1 && o0 < a1) return kDeclined;
    if (s0 < o1 && o0 < s1) return kDeclined;
    if (a0 < s1 && s0 < a1) return kDeclined;

    const double k   = (double)At<float>(obj, kK);
    const double cB  = (double)At<float>(obj, kScaleB);
    const double cA  = (double)At<float>(obj, kScaleA);
    const double bi  = (double)At<float>(obj, kBias);
    const float  qA  = ThroughX87(At<float>(obj, kPassA));
    const float  qB  = ThroughX87(At<float>(obj, kPassB));

    float* const bnd = reinterpret_cast<float*>(obj + kBounds);
    __m128 vmin = _mm_set_ps(0.0f, bnd[2], bnd[1], bnd[0]);
    __m128 vmax = _mm_set_ps(0.0f, bnd[5], bnd[4], bnd[3]);

    uint32_t i = head;
    do {
        float* const s = seg + (size_t)i * 12;
        const double age = (double)ages[i];
        const double v28 = (((age * 2.0) + dt) * k) * dt;
        const float z0 = (float)((double)s[2] + v28);
        const float z1 = (float)(v28 + (double)s[8]);
        s[2] = z0;
        s[8] = z1;

        const __m128 p = _mm_set_ps(0.0f, z0, s[1], s[0]);
        const __m128 r = _mm_set_ps(0.0f, z1, s[7], s[6]);
        vmin = _mm_min_ps(p, vmin);
        vmax = _mm_max_ps(p, vmax);
        vmin = _mm_min_ps(r, vmin);
        vmax = _mm_max_ps(r, vmax);

        const float na = (float)(dt + (double)ages[i]);
        ages[i] = na;
        const float v = (float)((((double)na * cB) * cA) + bi);
        s[4]  = v;
        s[5]  = qA;
        s[10] = v;
        s[11] = qB;

        if (++i >= cap) i -= cap;
    } while (i != tail);

    alignas(16) float mn[4], mx[4];
    _mm_store_ps(mn, vmin);
    _mm_store_ps(mx, vmax);
    bnd[0] = mn[0]; bnd[1] = mn[1]; bnd[2] = mn[2];
    bnd[3] = mx[0]; bnd[4] = mx[1]; bnd[5] = mx[2];
    return kDone;
}

// One fixed emitter: eight edges, the ring running from 6 round to 3, with
// deterministic contents. obj is kObjSpan bytes, ages eight floats and seg
// 8 * 12.
inline void BuildFixed(unsigned char* obj, float* ages, float* seg) {
    for (unsigned i = 0; i < kObjSpan; ++i) obj[i] = 0;

    uint32_t s = 12345u;
    for (unsigned i = 0; i < 8; ++i) {
        s = s * 1664525u + 1013904223u;
        ages[i] = (float)((int)(s >> 8) % 5000) * 0.0001f;
    }
    for (unsigned i = 0; i < 8 * 12; ++i) {
        s = s * 1664525u + 1013904223u;
        seg[i] = (float)((int)(s >> 8) % 20001 - 10000) * 0.01f;
    }
    At<uint32_t>(obj, kCapacity) = 8;
    At<uint32_t>(obj, kTail)     = 3;
    At<uint32_t>(obj, kHead)     = 6;
    At<float*>(obj, kAges)       = ages;
    At<float*>(obj, kEdges)      = seg;
    At<float>(obj, kScaleA) = 1.5f;
    At<float>(obj, kScaleB) = 0.75f;
    At<float>(obj, kPassA)  = 0.25f;
    At<float>(obj, kBias)   = -0.5f;
    At<float>(obj, kPassB)  = 0.125f;
    At<float>(obj, kK)      = 0.9f;
    float* const b = reinterpret_cast<float*>(obj + kBounds);
    b[0] = b[1] = b[2] = 3.4028235e38f;
    b[3] = b[4] = b[5] = -3.4028235e38f;
}

// FNV-1a over everything the loop can write, leaving out the two buffer
// pointers, which differ between two copies by construction.
inline uint32_t ImageChecksum(const unsigned char* obj, const float* ages, const float* seg) {
    uint32_t h = 0x811C9DC5u;
    auto mix = [&h](const void* p, unsigned n) {
        const unsigned char* q = static_cast<const unsigned char*>(p);
        for (unsigned i = 0; i < n; ++i) h = (h ^ q[i]) * 0x01000193u;
    };
    mix(obj, kAges);                               // up to the ages pointer
    mix(obj + kAges + 4, kEdges - kAges - 4);      // between the two pointers
    mix(obj + kEdges + 4, kObjSpan - kEdges - 4);  // and the rest
    mix(ages, 8 * sizeof(float));
    mix(seg, 8 * 12 * sizeof(float));
    return h;
}

// The fixed emitter run through whatever `run(obj, dt)` is. The harness prints
// the checksum for the client's own instruction sequence and for Run; the
// module compares Run against the harness's number at startup, which catches a
// compiler that has changed the arithmetic without a game being open.
template <class RunFn> inline uint32_t FixedScenario(RunFn run) {
    float ages[8];
    float seg[8 * 12];
    alignas(16) unsigned char obj[kObjSpan];
    BuildFixed(obj, ages, seg);
    run(obj, 0.0166667);
    return ImageChecksum(obj, ages, seg);
}

}  // namespace RibbonEdgeModel
