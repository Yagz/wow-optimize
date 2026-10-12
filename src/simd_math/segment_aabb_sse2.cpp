// ============================================================================
// Description: Replaces the segment/box test's x87 status-word round-trips.
// Safety & Threading: Main thread; the function is pure.
// ============================================================================
// sub_7F9480 tests a line segment against an axis-aligned box. It is 0.88% of
// executing time in a tester's uncapped session, and the profile's weight sits
// at 0x7F94FD - which is not arithmetic:
//
//     fld    dword ptr [ecx]
//     fcomp  dword ptr [eax+ecx]
//     fnstsw ax
//     test   ah, 41h            <- the samples land here
//     jnz    ...
//
// That is the x87 way of branching on a float comparison: the FPU status word
// has to be serialised into a general register before the flags can be tested,
// and the `test` stalls waiting for it. There are ten of those pairs in this
// function. SSE2 has no status word in the path at all - comiss puts the answer
// straight into EFLAGS - so what is being removed here is a mechanism, not a
// calculation. The same shape of finding as the Lua pool's chunk walk.
// ---------------------------------------------------------------------------
// Two tests decide by bits, and ten decide about NaN
//
// Reproducing the branches meant reading each condition-code test rather than
// the pseudocode, because several do not mean what a plain C rewrite would say.
//
// `cmp dword ptr [edi+ecx], 0` tests the delta's *bit pattern* against zero, so
// negative zero counts as non-zero and the division below it runs, producing an
// infinity. A `d != 0.0f` would have taken the other branch.
//
// `test [ebp+eax*4+var_1C], 80000000h` tests the chosen parameter's sign bit,
// so negative zero is rejected even though `-0.0f >= 0.0f` is true.
//
// Every one of the ten comparisons masks C0 and C2, or C0 and C3, in a way that
// lets an unordered result - a NaN coordinate - fall through as "not outside".
// Written in C the same way round: each test is the positive form that is false
// for a NaN, so a bad coordinate keeps the segment rather than culling it.
// Getting that backwards is not a slow bug, it is geometry vanishing.
//
// `fst dword ptr [eax+ecx]` in the second loop stores the hit point but does not
// pop it, and the two comparisons that follow read the register, not the float
// just written. So the bounds are checked against the unrounded value. Nothing
// ever reads the stored one - it is a local the caller never sees - so the store
// is not reproduced and the comparison is. This is the third time in this
// project that an x87 value has been used at full width after being stored
// narrower, and it is the detail that makes a naive rewrite wrong.
// ---------------------------------------------------------------------------
// Why double, and why that is enough
//
// The x87 control word sits at 53-bit precision, which is what a double carries,
// so every subtract, multiply, divide and compare here reproduces exactly when
// written in double - provided the roundings to float happen where the client
// puts them, which is on the delta array and on the parameter array, and
// nowhere else.
// ---------------------------------------------------------------------------
// Verification
//
// The function reads three arrays and writes only its own locals. It is pure, so
// both versions are run and their answers compared - no saving, no restoring, no
// predicting.
// ============================================================================

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstdint>
#include <cstring>

#include "segment_aabb_sse2.h"
#include "MinHook.h"
#include "version.h"
#include "config.h"
#include "sampling_profiler.h"
#include "ab_test.h"
#include "self_bench.h"
#include "session_verdict.h"

extern "C" void Log(const char* fmt, ...);

MH_STATUS WineSafe_CreateHook(void* target, void* detour, void** original);
MH_STATUS WO_EnableHook(void* target);

namespace SegmentAabb {

namespace {

constexpr uintptr_t kTest    = 0x007F9480;
constexpr uintptr_t kInitT   = 0x009E2EF4;   // what the parameters start at
constexpr uintptr_t kEpsilon = 0x009EA558;   // the bounds slack in the second pass

typedef int (__cdecl* segTest_fn)(const float* box, const float* start, const float* end);
segTest_fn orig_Test = nullptr;

bool g_installed = false;
bool g_armed     = false;
// Set at init when the A/B harness names this module, so the hot path
// tests a plain bool instead of calling out on every invocation.
bool g_abSubject = false;
bool g_dead      = false;
int  g_benchSlot = -1;

// Plain 32-bit, because a locked increment on a path this hot has eaten whole
// optimisations in this project before. But 32 bits is not enough to divide by:
// a field session has this counter at 1703651856 and climbing, against a wrap at 4294967296, and
// the call counter always wraps before the hit counter because every call
// increments it while only a hit increments the other. Past the wrap the report
// divides a real count by a wrapped one.
//
// That is not hypothetical here. The sibling module frustum_aabb printed
// "1496607690 visibility tests, 1691870587 came back visible (113.0%)" and once
// 73322.0%, which is more things visible than tests run. Same shape, same fix:
// each counter keeps its wraps and the report recombines them as a double.
unsigned long g_calls     = 0;
unsigned long g_callWraps = 0;
unsigned long g_verified  = 0;
unsigned long g_hits      = 0;
unsigned long g_hitWraps  = 0;

inline void Bump(unsigned long& low, unsigned long& wraps) {
    const unsigned long before = low;
    low = before + 1;
    if (low < before) ++wraps;
}

inline double Total(unsigned long low, unsigned long wraps) {
    return (double)low + (double)wraps * 4294967296.0;
}


constexpr unsigned long kVerifyFirst  = 20000;
constexpr unsigned long kResampleMask = 4095;

float  g_initT = -1.0f;
double g_eps   = 0.0;

inline uint32_t Bits(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    return b;
}

// The plane-coherency hint that frustum_aabb_sse2 uses was measured here too
// and is deliberately not applied. Written down so it is not re-derived.
//
// The axis loop below qualifies for it. Each iteration touches only its own
// index, `inside` is an AND across them, and an early return is an existential,
// so any axis order gives the same answer - checked over 200000 cases against
// all six permutations, with NaN, both infinities and both zeroes among the
// generated components, zero disagreements.
//
// It is not worth doing. On a segment walked against boxes in grid order the
// axes evaluated per call go from 1.336 to 1.004, and shuffled from 1.336 to
// 1.040. The saving is a third of one axis test - two float comparisons - while
// the cost is a global load on entry and a global store on every cull, on a path
// that culls 91.8% of 1225134361 calls a session. The fixed order already stops
// on the first axis most of the time, which is the whole reason the hint has
// nothing left to win.
//
// The frustum test is the opposite case and that is why it got the hint: six
// planes rather than three, the culling plane sits deeper in the fixed order,
// and the measured drop there is 3.486 to 1.854.
int Evaluate(const float* box, const float* start, const float* end) {
    const float* mn = box;
    const float* mx = box + 3;

    // The client also keeps a hit point in a local array. Nothing reads it -
    // not the caller, which only gets the int, and not the code below. It is
    // not reproduced; what it decides is reproduced, which is the comparison.
    float d[3], t[3];
    for (int i = 0; i < 3; i++) {
        d[i] = (float)((double)end[i] - (double)start[i]);
        t[i] = g_initT;
    }

    bool inside = true;

    for (int i = 0; i < 3; i++) {
        // The client branches on `min <= start`, taking the other path for an
        // unordered compare too. Written as the positive test, a NaN falls to
        // the same side it does there.
        if (mn[i] > start[i]) {
            if (mn[i] > end[i]) return 0;
            inside = false;
            if (Bits(d[i]) != 0)
                t[i] = (float)(((double)mn[i] - (double)start[i]) / (double)d[i]);
        } else if (mx[i] < start[i]) {
            if (mx[i] < end[i]) return 0;
            inside = false;
            if (Bits(d[i]) != 0)
                t[i] = (float)(((double)mx[i] - (double)start[i]) / (double)d[i]);
        }
    }

    if (inside) return 1;

    int k = (t[0] < t[1]) ? 1 : 0;
    if (t[k] < t[2]) k = 2;

    // A sign-bit test, not a comparison: negative zero is rejected here even
    // though it compares equal to zero.
    if (Bits(t[k]) & 0x80000000u) return 0;

    const double tk = (double)t[k];
    for (int i = 0; i < 3; i++) {
        if (i == k) continue;
        // `fst` stores this narrow and keeps the register, and the two tests
        // below read the register. So the bounds are checked against the
        // unrounded value, which is what h is here.
        double h = (double)d[i] * tk + (double)start[i];
        if ((double)mn[i] - g_eps > h) return 0;
        if ((double)mx[i] + g_eps < h) return 0;
    }
    return 1;
}

}  // namespace

// Faults the guard caught, while verifying or after. The fallback is the
// client's own test, which reads the same box and segment, so a fault here is
// one the fallback would take too; the armed path runs without an exception
// frame once kVerifyFirst tests have run guarded and this is still zero. If it
// ever is not, every test stays guarded.
static unsigned long g_caught = 0;

static __declspec(noinline) int TestGuarded(const float* box, const float* start,
                                            const float* end) {
    __try {
        int r = Evaluate(box, start, end);
        if (r) Bump(g_hits, g_hitWraps);
        return r;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        ++g_caught;
        return orig_Test(box, start, end);
    }
}

static __declspec(noinline) int TestVerify(const float* box, const float* start,
                                           const float* end) {
    {
        // The verification already runs both halves on the same input. Timing
        // it is the only paired comparison this project gets without asking a
        // tester to configure anything.
        // Alternated, so neither half always meets the cold caches and the other the warm ones.
        static unsigned s_benchOrder = 0;
        int mine, theirs;
        if (++s_benchOrder & 1) {
            const uint64_t tA = SelfBench::Now();
            theirs = orig_Test(box, start, end);
            const uint64_t tB = SelfBench::Now();
            __try {
                mine = Evaluate(box, start, end);
            } __except (EXCEPTION_EXECUTE_HANDLER) {
                ++g_caught;
                return theirs;
            }
            SelfBench::Pair(g_benchSlot, SelfBench::Now() - tB, tB - tA);
        } else {
            const uint64_t tA = SelfBench::Now();
            __try {
                mine = Evaluate(box, start, end);
            } __except (EXCEPTION_EXECUTE_HANDLER) {
                ++g_caught;
                return orig_Test(box, start, end);
            }
            const uint64_t tB = SelfBench::Now();
            theirs = orig_Test(box, start, end);
            SelfBench::Pair(g_benchSlot, tB - tA, SelfBench::Now() - tB);
        }
        g_verified++;

        if (mine != theirs) {
            g_dead = true;
            Verdict::Add(Verdict::Bad,
                         "SegmentAabb disagreed with the client and retired itself for "
                         "this session");
            Log("[SegmentAabb] DISAGREED with the client after %lu tests - retired "
                "for this session, every test now goes to the client's own code. "
                "It answered %d and this answered %d.", g_verified, theirs, mine);
            return theirs;
        }
        if (!g_armed && g_verified >= kVerifyFirst) {
            g_armed = true;
            Log("[SegmentAabb] armed: %lu tests agreed with the client. Now "
                "answering directly and rechecking one in %lu.",
                g_verified, kResampleMask + 1);
        }
        if (theirs) Bump(g_hits, g_hitWraps);
        return theirs;
    }
}

int __cdecl Hooked_TestBody(const float* box, const float* start, const float* end) {
    Bump(g_calls, g_callWraps);
    if (g_dead || !box || !start || !end) return orig_Test(box, start, end);
    if (!g_armed || (g_calls & kResampleMask) == 0) return TestVerify(box, start, end);
    if (g_caught) return TestGuarded(box, start, end);
    int r = Evaluate(box, start, end);
    if (r) Bump(g_hits, g_hitWraps);
    return r;
}

// The detour proper, kept apart from the body above for one reason: the
// A/B harness times the call, and a scope guard that closed the sample on
// every return path cannot be used in a function containing __try - MSVC
// refuses object unwinding alongside SEH. A wrapper has no __try of its own,
// so one pair of reads covers every path the body can take, including the
// ones it takes out of an exception handler.
//
// When no test names this module the whole thing is a branch on a false
// global followed by a direct call.
int __cdecl Hooked_Test(const float* box, const float* start, const float* end) {
    if (!g_abSubject) return Hooked_TestBody(box, start, end);
    unsigned long long t = AbTest::TickIn();
    int r = AbTest::StandAside() ? orig_Test(box, start, end)
                                       : Hooked_TestBody(box, start, end);
    AbTest::TickOut(t);
    return r;
}

bool Init() {
    if (!Config::g_settings.OptSegmentAabb) return true;

    if (IsBadReadPtr((void*)kTest, 16) || IsBadReadPtr((void*)kInitT, 4) ||
        IsBadReadPtr((void*)kEpsilon, 4)) {
        Log("[SegmentAabb] 0x%08X unreadable - not installing", (unsigned)kTest);
        return false;
    }
    static const unsigned char kExp_Test[8] = { 0x55, 0x8B, 0xEC, 0x83, 0xEC, 0x34, 0x8B, 0x45 };
    if (memcmp((const void*)kTest, kExp_Test, 8) != 0) {
        Log("[SegmentAabb] BAD PROLOGUE for SegmentAabb at 0x%08X", (unsigned)kTest);
        return false;
    }

    // Both constants come from the client rather than from a literal here.
    g_initT = *(const float*)kInitT;
    g_eps   = (double)*(const float*)kEpsilon;

    if (WineSafe_CreateHook((void*)kTest, (void*)Hooked_Test, (void**)&orig_Test) != MH_OK) {
        Log("[SegmentAabb] hook NOT created");
        return false;
    }
    if (WO_EnableHook((void*)kTest) != MH_OK) {
        Log("[SegmentAabb] hook created but could not be enabled");
        return false;
    }

    g_abSubject = AbTest::IsSubject("SegmentAabb", &g_abSubject);
    g_benchSlot = SelfBench::Register("SegmentAabb");
    if (g_abSubject) {
        Log("[SegmentAabb] under A/B test: it alternates on and off in stints "
            "and AbTest reports the frame times either way. The correctness "
            "checks are unaffected and still retire it on a disagreement.");
    }

    g_installed = true;
    SamplingProfiler::RegisterSelfSymbol("SegmentAabb_SSE2", (const void*)&Hooked_Test);
    Log("[SegmentAabb] ACTIVE on the segment/box test (sub_7F9480 @ 0x%08X), "
        "0.88%% of executing time in an uncapped tester session. The profile's "
        "weight is not on its arithmetic: it sits on the `test ah` that waits for "
        "an `fnstsw ax`, the x87 way of branching on a float comparison, and there "
        "are ten of those in the function. SSE2 puts the answer straight into the "
        "flags. Ten condition-code tests were transcribed rather than guessed - "
        "two of them decide on bit patterns, so negative zero behaves as the "
        "client has it, and every one lets a NaN coordinate keep the segment "
        "instead of culling it. Constants read from the client at 0x%08X and "
        "0x%08X. Pure function, so both answers are compared for the first %lu "
        "calls and one in %lu after.",
        (unsigned)kTest, (unsigned)kInitT, (unsigned)kEpsilon,
        kVerifyFirst, kResampleMask + 1);
    return true;
}

void LogStats() {
    if (!Config::g_settings.OptSegmentAabb) return;
    if (!g_installed) { Log("[SegmentAabb] not installed - nothing measured"); return; }
    if (g_calls == 0) { Log("[SegmentAabb] installed but never called"); return; }
    Log("[SegmentAabb]   exception guard: %lu fault(s) caught; the armed path runs %s.",
        g_caught, !g_armed ? "guarded, still verifying"
                  : (g_caught ? "guarded, because the guard has caught something"
                              : "without an exception frame"));

    const double callsTotal = Total(g_calls, g_callWraps);
    const double hitsTotal  = Total(g_hits, g_hitWraps);
    Log("[SegmentAabb] %.0f segment tests%s, %.0f intersected (%.1f%%), %lu verified "
        "against the client. Counts are lower bounds.",
        callsTotal,
        g_dead ? " - RETIRED on a disagreement"
               : (g_armed ? "" : " - still verifying, the client still answers every one"),
        hitsTotal, callsTotal > 0.0 ? 100.0 * hitsTotal / callsTotal : 0.0,
        g_verified);
}

void Shutdown() {
    if (g_installed) MH_DisableHook((void*)kTest);
}

}  // namespace SegmentAabb
