#pragma once
// ============================================================================
// quat_lerp_core.h
//
// The lerp and renormalise of sub_982630 / sub_982570 as one inline function,
// bit-identical to the client's x87 sequence (see quat_lerp_sse2.cpp for the
// argument and the measurement). quat_lerp_sse2.cpp hooks the client's routine
// with it; the quaternion track evaluator calls it directly once the hook has
// agreed with the client for its whole learning phase.
// ============================================================================

#include <emmintrin.h>

namespace QuatLerpCore {

// The client's constants, read out of sub_982570 as exact bit patterns:
// 3F82BE62, 3F0852F8, 3F758559, 3F26F151, 3F6A4B55 at 0x00AA2E5C upwards. They
// are float constants in the client, so each widens to double exactly.
constexpr double kC0 = 1.02143502235412598;   // flt_AA2E5C
constexpr double kC1 = 0.959065973758697510;  // flt_AA2E64
constexpr double kC2 = 0.532516002655029297;  // flt_AA2E60
constexpr double kT1 = 0.915211975574493408;  // flt_AA2E6C, one step above this
constexpr double kT2 = 0.652119696140289307;  // flt_AA2E68, two above, three below

__forceinline void LerpNormalise(float* out, float t, const float* a, const float* b) {
    // --- the lerp, sub_982630 ---
    // Four components at once, in double, then rounded to float exactly where
    // the client rounds: it computes each component in x87 and ends the
    // sequence with `fstp dword`, so the interpolated quaternion is single
    // precision before anything else touches it.
    __m128  va  = _mm_loadu_ps(a);
    __m128  vb  = _mm_loadu_ps(b);
    __m128d a01 = _mm_cvtps_pd(va);
    __m128d a23 = _mm_cvtps_pd(_mm_movehl_ps(va, va));
    __m128d b01 = _mm_cvtps_pd(vb);
    __m128d b23 = _mm_cvtps_pd(_mm_movehl_ps(vb, vb));
    __m128d td  = _mm_set1_pd((double)t);

    __m128d q01 = _mm_add_pd(_mm_mul_pd(_mm_sub_pd(b01, a01), td), a01);
    __m128d q23 = _mm_add_pd(_mm_mul_pd(_mm_sub_pd(b23, a23), td), a23);

    float q[4];
    _mm_storeu_ps(q, _mm_movelh_ps(_mm_cvtpd_ps(q01), _mm_cvtpd_ps(q23)));

    // --- the normalise, sub_982570 ---
    // Left to right, which is what the x87 stack does: x*x, then y*y and add,
    // then z*z and add, then w*w and add.
    double x = q[0], y = q[1], z = q[2], w = q[3];
    double v1 = x * x;
    v1 = v1 + y * y;
    v1 = v1 + z * z;
    v1 = v1 + w * w;

    // The client's step schedule, in double, because the client keeps every one
    // of these in an x87 register at 53-bit precision and never stores one to
    // float.
    double v2 = kC0 - (v1 - kC1) * kC2;
    double v3;
    if (v1 > kT1) {
        v3 = v2;
    } else {
        v2 = v2 * (kC0 - (v2 * v2 * v1 - kC1) * kC2);
        if (v1 > kT2) v3 = v2;
        else          v3 = (kC0 - (v1 * (v2 * v2) - kC1) * kC2) * v2;
    }

    // `fld dword [q]; fmul st,st(1); fstp dword [out]` - the float component is
    // widened, multiplied at 53 bits, and rounded once on store.
    __m128d v3d = _mm_set1_pd(v3);
    __m128  qf  = _mm_loadu_ps(q);
    __m128d p01 = _mm_mul_pd(_mm_cvtps_pd(qf), v3d);
    __m128d p23 = _mm_mul_pd(_mm_cvtps_pd(_mm_movehl_ps(qf, qf)), v3d);
    _mm_storeu_ps(out, _mm_movelh_ps(_mm_cvtpd_ps(p01), _mm_cvtpd_ps(p23)));
}

}  // namespace QuatLerpCore
