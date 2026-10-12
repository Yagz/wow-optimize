// ============================================================================
// Module: floor_split_sse2.cpp
//
// sub_5FE800, the client's split of a float into an integer part and a
// remainder. It is a leaf of 113 instructions called from 26 sites, and every
// call costs two x87 control-word loads:
//
//     fnstcw [cw] / or eax,0C00h / fldcw [tmp] / fistp / fldcw [cw]
//
// A control-word load drains the x87 pipeline, and the conversion itself is one
// SSE2 instruction that needs no mode change. The callers that matter are the
// sky and cloud maths - sub_7EECC0 calls it twelve times and sub_7EEA90 four -
// and the sine and cosine pair sub_6F7A60, which the particle update runs once
// per particle and which calls this twice, so one pair of trigonometric values
// costs four control-word loads.
//
//   __cdecl(float x, float* outFrac, int* outInt)
//
// What it computes, transcribed from the disassembly rather than assumed:
//
//   the branch is `test ah, 41h` after comparing x with zero, so it is taken
//   when x is below zero, exactly zero, or unordered - not merely when negative
//
//     x > 0    intPart = truncate(x)          remainder = x - intPart
//     else     intPart = truncate(x) - 1      remainder = x - intPart
//
// So zero goes down the second branch: the client answers an integer part of -1
// and a remainder of 1.0 for an input of 0.0, and any replacement that treats
// zero as the ordinary case is wrong on a value that turns up constantly. The
// remainder is computed by subtracting the stored 32-bit integer from the value
// still on the x87 stack and rounding once to float, which is what this does.
//
// The positive branch converts to a 64-bit integer and keeps the low half, so
// an input past 2^31 wraps rather than saturating. Rather than reproduce that
// with a 64-bit conversion - which on 32-bit MSVC is the control-word dance
// again - anything that large, and anything unordered, is handed to the client.
// Those are not values this is here for.
//
// Measured rather than argued. An offline harness ran the client's instruction
// sequence, transcribed into inline assembly, against this code:
//
//     4000000 cases, 0 differing
//     client 0.0  -> integer -1, remainder 1      this 0.0  -> -1, 1
//     client -3.0 -> integer -4, remainder 1      this -3.0 -> -4, 1
//
// over exact integers, denormals, small fractions and whole-range values. The
// two lines below the count are the cases the reading turns on, checked by name
// rather than left to the random draw.
//
// Verification, predict-then-compare. The function writes through two pointers.
// While learning, the answer is worked out into locals, the client runs and
// writes the real ones, and the two are compared bit for bit. The first
// difference retires this for the session, and the same two reads of the cycle
// counter give the paired timing.
// ============================================================================

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstdint>
#include <cstring>
#include <emmintrin.h>

#include "floor_split_sse2.h"
#include "MinHook.h"
#include "version.h"
#include "config.h"
#include "ab_test.h"
#include "session_verdict.h"
#include "self_bench.h"
#include "sampling_profiler.h"
#include "parallel_particles.h"

extern "C" void Log(const char* fmt, ...);
MH_STATUS WineSafe_CreateHook(void* target, void* detour, void** original);
MH_STATUS WO_EnableHook(void* target);

namespace FloorSplit {

namespace {

constexpr uintptr_t kTarget = 0x005FE800;

// push ebp / mov ebp,esp / fldz / sub esp,8 / fld [ebp+8] / fcom
const unsigned char kPrologue[12] = {
    0x55, 0x8B, 0xEC, 0xD9, 0xEE, 0x83, 0xEC, 0x08, 0xD9, 0x45, 0x08, 0xD8
};

// Past this the client's 64-bit conversion stops agreeing with a 32-bit one,
// and the value is not one this is here for.
constexpr double kLimit = 2100000000.0;

constexpr unsigned long kLearnCalls   = 20000;
constexpr unsigned long kResampleMask = 4095;

typedef void (__cdecl* SplitFn)(float x, float* outFrac, int* outInt);
SplitFn g_orig = nullptr;

bool g_installed = false;
bool g_dead = false;
bool g_abSubject = false;
int  g_benchSlot = -1;

unsigned long long g_calls = 0;
unsigned long long g_answered = 0;
unsigned long long g_declined = 0;
unsigned long long g_control = 0;
unsigned long g_verified = 0;
unsigned g_mismatches = 0;

__forceinline void Split(float x, float* outFrac, int* outInt) {
    const double xd = (double)x;
    // Zero and everything below it take the client's second branch.
    int i = _mm_cvttsd_si32(_mm_set_sd(xd));
    if (!(xd > 0.0)) --i;
    *outInt = i;
    *outFrac = (float)(xd - (double)i);
}

__declspec(noinline) void Retire(float x, float mineF, int mineI,
                                 float theirsF, int theirsI) {
    ++g_mismatches;
    g_dead = true;
    uint32_t a, b;
    memcpy(&a, &mineF, 4);
    memcpy(&b, &theirsF, 4);
    Log("[FloorSplit] RETIRED: for %.9g the client answered integer %d remainder "
        "%08X and this answered %d and %08X. Every split from here on is the "
        "client's own.", x, theirsI, b, mineI, a);
    Verdict::Add(Verdict::Bad, "FloorSplit split a float differently from the "
                 "client and retired itself for this session");
}

void __cdecl Detour(float x, float* outFrac, int* outInt) {
    ++g_calls;
    if (g_dead || !outFrac || !outInt) { g_orig(x, outFrac, outInt); return; }
    // NaN fails both of these, which is the intent.
    if (!(x > -kLimit && x < kLimit)) {
        ++g_declined;
        g_orig(x, outFrac, outInt);
        return;
    }
    if (g_abSubject && AbTest::StandAside()) {
        ++g_control;
        g_orig(x, outFrac, outInt);
        return;
    }

    const bool learning = (g_verified < kLearnCalls) ||
                          ((unsigned long)g_calls & kResampleMask) == 0;
    if (!learning) {
        Split(x, outFrac, outInt);
        ++g_answered;
        return;
    }

    float mineF = 0.0f;
    int   mineI = 0;
    // Alternated, so neither half always meets the cold caches and the other the warm ones.
    static unsigned s_benchOrder = 0;
    unsigned long long oursCycles, theirsCycles;
    const unsigned long long tA = SelfBench::Now();
    if (++s_benchOrder & 1) {
        g_orig(x, outFrac, outInt);
        const unsigned long long tB = SelfBench::Now();
        Split(x, &mineF, &mineI);
        oursCycles = SelfBench::Now() - tB;
        theirsCycles = tB - tA;
    } else {
        Split(x, &mineF, &mineI);
        const unsigned long long tB = SelfBench::Now();
        g_orig(x, outFrac, outInt);
        oursCycles = tB - tA;
        theirsCycles = SelfBench::Now() - tB;
    }
    // SelfBench adds its sums without a lock; the particle workers reach this
    // function too.
    if (!ParallelParticles::OnWorkerThread())
        SelfBench::Pair(g_benchSlot, oursCycles, theirsCycles);

    uint32_t mineBits, theirBits;
    memcpy(&mineBits, &mineF, 4);
    memcpy(&theirBits, outFrac, 4);
    if (mineI != *outInt || mineBits != theirBits) {
        Retire(x, mineF, mineI, *outFrac, *outInt);
        return;
    }
    ++g_verified;
}

bool BytesMatch(uintptr_t addr, const unsigned char* want, size_t n) {
    __try {
        return memcmp((const void*)addr, want, n) == 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

}  // namespace

bool Init() {
    if (!Config::g_settings.OptFloorSplit) return true;

    if (!BytesMatch(kTarget, kPrologue, sizeof(kPrologue))) {
        Log("[FloorSplit] NOT active: the bytes at 0x%08X are not the split this was "
            "read from, so nothing was hooked.", (unsigned)kTarget);
        return false;
    }
    if (!WowOpt_ClientPatchAllowed((const void*)kTarget)) {
        Log("[FloorSplit] NOT active: No Client Patches is on, and this hooks a "
            "function inside wow.exe.");
        return false;
    }
    if (WineSafe_CreateHook((void*)kTarget, (void*)&Detour, (void**)&g_orig) != MH_OK) {
        Log("[FloorSplit] NOT active: the hook on 0x%08X could not be created.",
            (unsigned)kTarget);
        return false;
    }
    if (WO_EnableHook((void*)kTarget) != MH_OK) {
        MH_RemoveHook((void*)kTarget);
        Log("[FloorSplit] NOT active: the hook on 0x%08X could not be enabled.",
            (unsigned)kTarget);
        return false;
    }
    g_installed = true;
    g_abSubject = AbTest::IsSubject("FloorSplit", &g_abSubject);
    g_benchSlot = SelfBench::Register("FloorSplit");
    SamplingProfiler::RegisterSelfSymbol("FloorSplit", (const void*)&Detour);

    Log("[FloorSplit] ACTIVE on the float split (sub_5FE800 @ 0x%08X), a leaf called "
        "from 26 sites - twelve of them in the cloud maths and two in the sine and "
        "cosine pair the particle update runs per particle. Every call loaded the "
        "x87 control word twice to truncate; this does it with one instruction and "
        "no mode change. The first %lu calls are answered by the client and compared "
        "bit for bit, then one in %lu.",
        (unsigned)kTarget, kLearnCalls, kResampleMask + 1);
    if (g_abSubject)
        Log("[FloorSplit]   under A/B test: the control half runs the client's own "
            "function through the same hook.");
    return true;
}

void Shutdown() {
    if (!g_installed) return;
    MH_DisableHook((void*)kTarget);
    g_installed = false;
}

void LogStats() {
    if (!Config::g_settings.OptFloorSplit) return;
    if (!g_installed) {
        Log("[FloorSplit] not installed - the reason is at the top of this log");
        return;
    }
    if (g_calls == 0) {
        Log("[FloorSplit] hooked, and nothing has been split through it yet. That is "
            "a measurement: none of its 26 call sites has run since it went in.");
        return;
    }
    Log("[FloorSplit] %llu call(s), %llu answered here, %llu handed back for being "
        "too large or unordered. Plain counters, lower bounds.",
        g_calls, g_answered, g_declined);
    if (g_mismatches) {
        Log("[FloorSplit]   DISABLED after an answer that differed from the client's; "
            "the line that says which is earlier in this log.");
    } else if (g_verified < kLearnCalls) {
        Log("[FloorSplit]   %lu of %lu calls compared with the client so far, none "
            "differed.", g_verified, kLearnCalls);
    } else {
        Log("[FloorSplit]   %lu calls compared with the client bit for bit, none "
            "differed; one in %lu is still compared.", g_verified, kResampleMask + 1);
    }
    if (g_abSubject)
        Log("[FloorSplit]   %llu call(s) ran the client's function as the A/B control "
            "half.", g_control);
}

}  // namespace FloorSplit
