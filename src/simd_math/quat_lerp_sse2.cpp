// ============================================================================
// Description: SSE2 replacement for the M2 quaternion lerp + renormalise.
// ============================================================================
// sub_982630 is called once per animated bone per frame from the quaternion
// track, and it is four independent lerps followed by a renormalise. The client
// evaluates each component separately on the x87 stack:
//
//     out[i] = (b[i] - a[i]) * t + a[i]        i = 0..3
//     then sub_982570 scales all four by an approximation of 1/sqrt(|q|^2)
//
// The renormalise is not a sqrt. It is a polynomial estimate of the inverse
// root refined by one to three Newton steps, chosen by how close the squared
// length already is to one - cheap for the almost-unit quaternions a lerp
// produces. That structure is kept exactly as the client has it, including both
// thresholds and all three constants, because it decides how many steps run and
// a different schedule would be a different function.
//
// What changes is that the four components move together.
// ---------------------------------------------------------------------------
// The precision question, which this project has got wrong before.
//
// The note in the archive is blunt: every SIMD replacement written in single
// precision was wrong, and the "sub-ULP" comments attached to them were false.
// x87 under MSVC carries 53 bits through the intermediate steps and rounds once
// on store; packed single rounds at every operation.
//
// The conclusion drawn from that used to be that a vector replacement here can
// only ever be approximate, and this file shipped one: packed single throughout,
// worst absolute error 2.980e-07.
//
// That conclusion had a gap in it. SSE2 does not only have packed single. x87
// under MSVC runs at 53-bit precision control, and 53 bits is exactly what a
// packed double lane carries, so a sequence of packed double operations in the
// client's own order reproduces the client's arithmetic step for step. Two
// lanes instead of four, and bit-exact instead of close.
//
// Measured on 3000000 randomised near-unit quaternion pairs, against a
// reference in Python doubles rounding to single only where the client
// executes `fstp dword`:
//
//     packed double  bit-exact 100.0000%   worst 0.000e+00
//     packed single  bit-exact   6.4893%   worst 2.980e-07
//
// The single-precision row reproduces the 2.980e-07 recorded by the earlier
// measurement, which is what says the reference is the same reference.
//
// Width is the whole of it, and the summation order is not. The four squares
// are now summed left to right, x2 then +y2 then +z2 then +w2, because that is
// what the x87 stack does, and the version this replaces summed them as a
// shuffle tree. That change was measured on its own and makes no difference:
// in double, tree association and the client's order agreed on every one of
// 2000000 randomised quaternions, worst 0.000e+00, even carried through the
// nonlinear step schedule. It is written the client's way because that is
// easier to check against the disassembly, not because it was the defect.
//
// The same question in the matrix-vector hook next door has the same answer,
// and there the reason is plainer: the products of two floats are exact in
// double and the sum is rounded to float immediately, so association cannot
// survive to the result. Four million adversarial samples with the translation
// twenty binary orders above the products found no disagreement either.
//
// The harness proves the scheme. It cannot prove the transcription or that this
// machine's x87 precision control really is 53, so the module compares its
// output with the client's as bit patterns in the running game, with no
// tolerance, and disables itself on the first differing bit.
// ---------------------------------------------------------------------------

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <emmintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "quat_lerp_sse2.h"
#include "quat_lerp_core.h"
#include "MinHook.h"
#include "version.h"
#include "config.h"
#include "ab_test.h"
#include "session_verdict.h"
#include "self_bench.h"
#include "sampling_profiler.h"

extern "C" void Log(const char* fmt, ...);

namespace QuatLerpSse2 {

// Read by the quaternion track on every call, from every animation thread.
bool g_usable = false;

namespace {

constexpr uintptr_t kQuatLerp = 0x00982630;

typedef float* (__cdecl* QuatLerp_fn)(float* out, float t, const float* a, const float* b);
QuatLerp_fn orig_QuatLerp = nullptr;

// No tolerance. Every operation here is the client's operation at the client's
// width in the client's order, so the only correct outcome is the same four
// bit patterns. A tolerance would hide exactly the thing worth catching: if
// this client's x87 precision control is not 53 bits, or a constant is
// misread, or an operand is swapped, the first comparison says so and the
// module retires itself.

constexpr long kLearnCalls   = 20000;
constexpr long kResampleMask = 1023;

unsigned long g_calls      = 0;
unsigned long g_agreements = 0;
// Timed where the verification already runs both halves on the same input.
// A replacement nobody has timed against the client is a replacement nobody
// knows is one; this project already shipped a hash that was slower for its
// whole life and counted every call as a win.
int g_benchSlot = -1;
volatile LONG g_armed      = 0;
// Set at init when the A/B harness names this module, so the hot path
// tests a plain bool instead of calling out on every invocation.
bool g_abSubject = false;
volatile LONG g_dead       = 0;
bool          g_installed  = false;

inline void LerpNormalise(float* out, float t, const float* a, const float* b) {
    QuatLerpCore::LerpNormalise(out, t, a, b);
}

void Retire(const char* why) {
    g_usable = false;
    if (InterlockedExchange(&g_dead, 1) == 0) {
        Log("[QuatLerp] Disabled for this session: %s. The client's own routine "
            "runs from here on.", why);
    }
}

// The guard, and why it comes off.
//
// A field session takes 3002320120 interpolations through this hook. Each one
// ran inside a __try, and a __try region on 32-bit MSVC costs a prologue on
// every call whether anything faults or not - measured at 2.48 ns elsewhere in
// this project, which is seven seconds of main thread across a session.
//
// What the guard buys is nothing, and that is an argument about memory rather
// than probability: the three pointers are checked for null above, and when the
// handler fires control falls through to the client's own routine with the same
// pointers, which reads the same four floats from each and writes the same four.
// It cannot succeed where ours faulted. The guard does not recover a fault, it
// moves it a few instructions later into the client's code.
//
// So it is kept until it has proven that. Every call runs under it for the
// first kLerpProve; if it ever catches anything, that is logged and this stays
// guarded for the rest of the session.
constexpr unsigned long kLerpProve = 200000;
unsigned long g_lerpProved = 0;
unsigned long g_lerpFaults = 0;
volatile LONG g_lerpFaultLogged = 0;

__declspec(noinline) static bool LerpGuarded(float* out, float t, const float* a,
                                             const float* b) {
    __try {
        LerpNormalise(out, t, a, b);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        ++g_lerpFaults;
        if (InterlockedCompareExchange(&g_lerpFaultLogged, 1, 0) == 0)
            Log("[QuatLerp] the pointer guard caught a fault in the vector path. It "
                "stays on for the rest of this session, and the reasoning that it "
                "never fires is wrong - the module's own note says so.");
        return false;
    }
}

__declspec(noinline) static float* VerifyQuatLerp(float* out, float t, const float* a, const float* b);

float* __cdecl Hooked_QuatLerpBody(float* out, float t, const float* a, const float* b) {
    if (g_dead || !out || !a || !b) return orig_QuatLerp(out, t, a, b);

    unsigned long n = ++g_calls;
    bool verifying = (g_armed == 0) || ((n & kResampleMask) == 0);

    if (!verifying) {
        if (g_lerpFaults || g_lerpProved < kLerpProve) {
            ++g_lerpProved;
            if (!LerpGuarded(out, t, a, b)) {
                Retire("the vector path faulted");
                return orig_QuatLerp(out, t, a, b);
            }
            return out;
        }
        LerpNormalise(out, t, a, b);
        return out;
    }

    return VerifyQuatLerp(out, t, a, b);
}

__declspec(noinline) static float* VerifyQuatLerp(float* out, float t, const float* a, const float* b) {
    // While verifying, let the client write its own answer through `out` and put
    // ours somewhere else, then compare. The caller keeps the client's result,
    // so a session spent verifying behaves exactly like an unhooked one.
    float theirs[4];
    float mine[4];
    // Alternated, so neither half always meets the cold caches and the other the warm ones.
    static unsigned s_benchOrder = 0;
    __try {
        if (++s_benchOrder & 1) {
            const unsigned long long tA = SelfBench::Now();
            orig_QuatLerp(out, t, a, b);
            const unsigned long long tB = SelfBench::Now();
            theirs[0] = out[0]; theirs[1] = out[1]; theirs[2] = out[2]; theirs[3] = out[3];
            const unsigned long long tC = SelfBench::Now();
            LerpNormalise(mine, t, a, b);
            SelfBench::Pair(g_benchSlot, SelfBench::Now() - tC, tB - tA);
        } else {
            const unsigned long long tA = SelfBench::Now();
            LerpNormalise(mine, t, a, b);
            const unsigned long long tB = SelfBench::Now();
            orig_QuatLerp(out, t, a, b);
            SelfBench::Pair(g_benchSlot, tB - tA, SelfBench::Now() - tB);
            theirs[0] = out[0]; theirs[1] = out[1]; theirs[2] = out[2]; theirs[3] = out[3];
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        Retire("the vector path faulted during verification");
        return out;
    }

    // Compared as bit patterns, not as numbers. Two floats that differ in the
    // last bit are not equal here, and -0.0 does not pass for 0.0.
    bool same = true;
    for (int i = 0; i < 4; i++) {
        uint32_t bt, bm;
        memcpy(&bt, &theirs[i], 4);
        memcpy(&bm, &mine[i], 4);
        if (bt != bm) { same = false; break; }
    }

    if (!same) {
        Log("[QuatLerp] Differed from the client at t=%.9g. "
            "client=(%08X %08X %08X %08X) ours=(%08X %08X %08X %08X)",
            t,
            *(const uint32_t*)&theirs[0], *(const uint32_t*)&theirs[1],
            *(const uint32_t*)&theirs[2], *(const uint32_t*)&theirs[3],
            *(const uint32_t*)&mine[0], *(const uint32_t*)&mine[1],
            *(const uint32_t*)&mine[2], *(const uint32_t*)&mine[3]);
        Retire("a result was not bit-identical to the client's");
        return out;
    }

    unsigned long ok = ++g_agreements;
    if (g_armed == 0 && ok >= kLearnCalls) {
        InterlockedExchange(&g_armed, 1);
        // Under the A/B harness the OFF half has to reach the client's routine
        // through this hook, so the caller must not go round it.
        if (!g_abSubject) g_usable = true;
        Log("[QuatLerp] %lu interpolations were bit-identical to the client. "
            "Using the vector path from here; one call in %d stays checked.",
            ok, (int)(kResampleMask + 1));
    }
    return out;
}

// The detour proper, split from the body above so the A/B harness can time
// the call. A scope guard would be the natural way to close that sample on
// every return path, and MSVC refuses object unwinding in a function that
// contains __try - which the body does. This wrapper has none, so one pair
// of reads covers every path the body can leave by.
float* __cdecl Hooked_QuatLerp(float* out, float t, const float* a, const float* b) {
    if (!g_abSubject) return Hooked_QuatLerpBody(out, t, a, b);
    unsigned long long abTick = AbTest::TickIn();
    float* r = AbTest::StandAside() ? orig_QuatLerp(out, t, a, b)
                    : Hooked_QuatLerpBody(out, t, a, b);
    AbTest::TickOut(abTick);
    return r;
}

} // namespace

bool Init() {
    if (!Config::g_settings.OptQuatLerpSse2) return true;

    static const unsigned char kExp_QuatLerp[8] = { 0x55, 0x8B, 0xEC, 0x83, 0xEC, 0x10, 0x8B, 0x45 };
    if (IsBadReadPtr((void*)kQuatLerp, 8) ||
        memcmp((const void*)kQuatLerp, kExp_QuatLerp, 8) != 0) {
        Log("[QuatLerp] 0x%08X bad prologue or unreadable - not installing", (unsigned)kQuatLerp);
        return false;
    }

    if (WineSafe_CreateHook((void*)kQuatLerp, (void*)Hooked_QuatLerp,
                            (void**)&orig_QuatLerp) != MH_OK) {
        Log("[QuatLerp] hook NOT created");
        return false;
    }
    if (WO_EnableHook((void*)kQuatLerp) != MH_OK) {
        Log("[QuatLerp] hook created but could not be enabled");
        return false;
    }

    g_abSubject = AbTest::IsSubject("QuatLerpSse2", &g_abSubject);
    if (g_abSubject) {
        Log("[QuatLerp] under A/B test: it alternates on and off in stints, "
            "and AbTest reports both the frame times and the cost of this "
            "call each way. The correctness checks are unaffected.");
    }

    g_installed = true;
    SamplingProfiler::RegisterSelfSymbol("QuatLerp_SSE2", (const void*)&Hooked_QuatLerp);
    g_benchSlot = SelfBench::Register("QuatLerp");
    Log("[QuatLerp] ACTIVE on sub_982630, the per-bone quaternion interpolation. "
        "Two components at a time in double rather than one at a time on the "
        "x87 stack, which makes it bit-identical to the client rather than "
        "close to it: x87 under MSVC carries 53 bits and so does a packed "
        "double lane. The first %ld results are compared with the client's as "
        "bit patterns, with no tolerance, and one in %d stays checked after "
        "that. A single differing bit disables it for the session.",
        kLearnCalls, (int)(kResampleMask + 1));

    return true;
}

void LogStats() {
    if (!Config::g_settings.OptQuatLerpSse2) return;
    if (!g_installed) { Log("[QuatLerp] not installed - nothing measured"); return; }
    if (g_calls == 0) { Log("[QuatLerp] installed but never called"); return; }
    Log("[QuatLerp] Lerp: %lu interpolations, %lu of them compared with the client "
        "and bit-identical%s",
        g_calls, g_agreements,
        g_dead ? " - DISABLED" : (g_armed ? "" : " (still verifying)"));
    // Three states, and the middle one is the interesting one: a session
    // that ends still proving has not yet dropped the exception frame this
    // pays on every one of those interpolations.
    Log("[QuatLerp]   pointer guard: %lu call(s) ran under it, it caught %lu, "
        "and it is %s.", g_lerpProved, g_lerpFaults,
        g_lerpFaults ? "held on by a catch"
                     : (g_lerpProved >= kLerpProve
                            ? "off, so the vector path carries no exception frame"
                            : "still proving"));
}

} // namespace QuatLerpSse2
