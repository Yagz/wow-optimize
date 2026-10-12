// ============================================================================
// Module: terrain_point_outcode_sse2.cpp
//
// Vectorized 3D point-vs-AABB Cohen-Sutherland outcode for terrain chunk
// culling and polygon generation (sub_7A61D0).
//
// In terrain scene traversal, sub_7D8840 (terrain collision polygon generation,
// 4,990 samples) and sub_7A6260 (terrain chunk candidate query) test vertices
// of candidate terrain cells against query bounding boxes using sub_7A61D0
// (96 bytes at 0x007A61D0).
//
// The client computes each of the six distances as (a - b) + eps in x87 at
// 53-bit precision, stores it as a float and takes the sign bit (0x007A61D9
// onwards; eps is flt_A3FDB8, 0.019444443f). The sign of the stored float is
// the sign of the unrounded value, so the outcode is the sign of each sum in
// double. This computes the six sums in packed double in the same order and
// takes their signs with movmskpd.
//
// An earlier version did the sums in packed single. Rounding a - b to float
// before adding eps can move a sum that is just below zero to exactly zero
// when the subtraction is inexact (a tiny coordinate against a box face near
// eps): pt = 1e-9, min = the float after eps gives a set bit in the client
// and a clear one in single. On two million cases chosen near zero, single
// disagreed with the client on 155,432 and double on none (Python doubles as
// the x87 reference, struct round-trips for single).
//
// A startup self-test compares it with the client's arithmetic transcribed in
// double on random and near-zero cases. At run time it is compared with the
// client on the first 10,000 calls and one in 128 after, and retires on the
// first difference. Off by default under the experimental switch
// TerrainPointOutcode.
// ============================================================================

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <emmintrin.h>
#include <cstdint>
#include <cstring>

#include "terrain_point_outcode_sse2.h"
#include "MinHook.h"
#include "version.h"
#include "config.h"
#include "ab_test.h"
#include "sampling_profiler.h"
#include "self_bench.h"

extern "C" void Log(const char* fmt, ...);
MH_STATUS WineSafe_CreateHook(void* target, void* detour, void** original);
MH_STATUS WO_EnableHook(void* target);

namespace TerrainPointOutcode {

namespace {

constexpr uintptr_t kTarget = 0x007A61D0;
constexpr uintptr_t kEpsAddr = 0x00A3FDB8;

// Prologue bytes of sub_7A61D0 (16 bytes):
// push ebp; mov ebp, esp; mov eax, [ebp+arg_4]; mov edx, [ebp+arg_0]; fld dword ptr [eax]; fsub dword ptr [edx]; push esi; fld ds:flt_A3FDB8
static const uint8_t kExpectedPrologue[16] = {
    0x55, 0x8B, 0xEC, 0x8B, 0x45, 0x0C, 0x8B, 0x55,
    0x08, 0xD9, 0x00, 0xD8, 0x22, 0x56, 0xD9, 0x05
};

typedef int (__cdecl* TerrainPointOutcodeFn)(const float* box, const float* pt);

static TerrainPointOutcodeFn g_orig = nullptr;

static bool g_installed = false;
static bool g_dead = false;
static bool g_abSubject = false;
static int  g_benchId = -1;

static __m128d g_epsVec;

// Telemetry counters:
static uint64_t g_calls = 0;
static uint64_t g_controlCalls = 0;
static uint64_t g_fastCalls = 0;
static uint64_t g_verified = 0;
static uint64_t g_mismatches = 0;


constexpr uint64_t kLearnCalls = 10000;
constexpr uint64_t kResampleMask = 127;

// Lanes: bits 0-1 are pt.x - min.x and pt.y - min.y; bits 2-3 are
// pt.z - min.z and max.x - pt.x; bits 4-5 are max.y - pt.y and max.z - pt.z.
__forceinline int Fast_TerrainPointOutcode(const float* box, const float* pt) {
    const __m128d d01 = _mm_add_pd(_mm_sub_pd(_mm_set_pd(pt[1], pt[0]),
                                              _mm_set_pd(box[1], box[0])), g_epsVec);
    const __m128d d23 = _mm_add_pd(_mm_sub_pd(_mm_set_pd(box[3], pt[2]),
                                              _mm_set_pd(pt[0], box[2])), g_epsVec);
    const __m128d d45 = _mm_add_pd(_mm_sub_pd(_mm_set_pd(box[5], box[4]),
                                              _mm_set_pd(pt[2], pt[1])), g_epsVec);
    return _mm_movemask_pd(d01) | (_mm_movemask_pd(d23) << 2) | (_mm_movemask_pd(d45) << 4);
}

__declspec(safebuffers)
static int __cdecl Hook_TerrainPointOutcode(const float* box, const float* pt) {
    ++g_calls;

    if (g_dead || !g_installed) {
        return g_orig(box, pt);
    }

    if (g_abSubject && AbTest::StandAside()) {
        ++g_controlCalls;
        return g_orig(box, pt);
    }

    ++g_fastCalls;

    if (g_calls <= kLearnCalls || (g_calls & kResampleMask) == 0) {
        // Alternated, so neither half always meets the cold caches and the other the warm ones.
        static unsigned s_benchOrder = 0;
        int fast_result, client_result;
        uint64_t fastCycles, clientCycles;
        const uint64_t t0 = SelfBench::Now();
        if (++s_benchOrder & 1) {
            client_result = g_orig(box, pt);
            const uint64_t t1 = SelfBench::Now();
            fast_result = Fast_TerrainPointOutcode(box, pt);
            fastCycles = SelfBench::Now() - t1;
            clientCycles = t1 - t0;
        } else {
            fast_result = Fast_TerrainPointOutcode(box, pt);
            const uint64_t t1 = SelfBench::Now();
            client_result = g_orig(box, pt);
            fastCycles = t1 - t0;
            clientCycles = SelfBench::Now() - t1;
        }

        ++g_verified;

        if (fast_result != client_result) {
            ++g_mismatches;
            g_dead = true;
            Log("[TerrainPointOutcode] MISMATCH: fast=%d (0x%02X), client=%d (0x%02X) at call %llu. Retiring hook.",
                fast_result, fast_result, client_result, client_result, g_calls);
            return client_result;
        }

        if (g_benchId >= 0) SelfBench::Pair(g_benchId, fastCycles, clientCycles);

        return fast_result;
    }

    return Fast_TerrainPointOutcode(box, pt);
}

// sub_7A61D0 as the disassembly has it: each sum rounded once to float by
// the fstp, and the sign bit of that float shifted into place.
int RefTerrainPointOutcode(const float* edx, const float* eax, double eps) {
    const float d[6] = {
        (float)(((double)eax[0] - edx[0]) + eps), (float)(((double)eax[1] - edx[1]) + eps),
        (float)(((double)eax[2] - edx[2]) + eps), (float)(((double)edx[3] - eax[0]) + eps),
        (float)(((double)edx[4] - eax[1]) + eps), (float)(((double)edx[5] - eax[2]) + eps) };
    int r = 0;
    for (int i = 0; i < 6; ++i) {
        uint32_t bits;
        memcpy(&bits, &d[i], 4);
        r |= (int)(bits >> 31) << i;
    }
    return r;
}

uint32_t g_rng = 0x2545F491u;
uint32_t Rng() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }
float RngF(float lo, float hi) { return lo + (hi - lo) * (float)(Rng() & 0xFFFFFF) / 16777215.0f; }
float Step(float x, int n) { uint32_t b; memcpy(&b, &x, 4); b += (uint32_t)n; memcpy(&x, &b, 4); return x; }

// Random boxes and points, and points placed so that a sum lands within a
// few ulps of zero with an inexact subtraction, where single precision and
// the client disagree.
bool RunSelfTest(double eps) {
    const float feps = (float)eps;
    for (int c = 0; c < 20000; ++c) {
        float box[6], pt[3];
        for (int i = 0; i < 3; ++i) {
            const float a = RngF(-50.0f, 50.0f), b = RngF(-50.0f, 50.0f);
            box[i] = a < b ? a : b;
            box[i + 3] = a < b ? b : a;
            pt[i] = RngF(-60.0f, 60.0f);
        }
        if (c % 2) {
            const int k = (int)(Rng() % 3);
            pt[k] = RngF(-1e-7f, 1e-7f);
            box[k] = Step(feps, (int)(Rng() % 7) - 3);
            box[k + 3] = Step(-feps, (int)(Rng() % 7) - 3);
        }
        if (Fast_TerrainPointOutcode(box, pt) != RefTerrainPointOutcode(box, pt, eps)) {
            Log("[TerrainPointOutcode] NOT active: case %d answered 0x%02X where the client's "
                "arithmetic gives 0x%02X.", c, Fast_TerrainPointOutcode(box, pt),
                RefTerrainPointOutcode(box, pt, eps));
            return false;
        }
    }
    return true;
}

} // anonymous namespace

bool Init() {
    if (!Config::g_settings.OptTerrainPointOutcode) {
        return false;
    }

    if (Config::g_settings.OptNoClientPatches) {
        Log("[TerrainPointOutcode] Client binary patches disabled by OptNoClientPatches, skipping.");
        return false;
    }

    float eps = 0.019444443f;
    __try {
        eps = *(const float*)kEpsAddr;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        eps = 0.019444443f;
    }
    g_epsVec = _mm_set1_pd((double)eps);

    if (!RunSelfTest((double)eps)) {
        return false;
    }

    if (std::memcmp((const void*)kTarget, kExpectedPrologue, sizeof(kExpectedPrologue)) != 0) {
        Log("[TerrainPointOutcode] Prologue mismatch at 0x%08X; hook not installed.", kTarget);
        return false;
    }

    MH_STATUS st = WineSafe_CreateHook((void*)kTarget, (void*)Hook_TerrainPointOutcode, (void**)&g_orig);
    if (st != MH_OK) {
        Log("[TerrainPointOutcode] Failed to create hook at 0x%08X (status=%d).", kTarget, (int)st);
        return false;
    }

    st = WO_EnableHook((void*)kTarget);
    if (st != MH_OK) {
        Log("[TerrainPointOutcode] Failed to enable hook at 0x%08X (status=%d).", kTarget, (int)st);
        return false;
    }

    g_installed = true;
    g_abSubject = AbTest::IsSubject("TerrainPointOutcode", &g_abSubject);
    g_benchId = SelfBench::Register("TerrainPointOutcode");

    SamplingProfiler::RegisterSelfSymbol("TerrainPointOutcode", (const void*)&Hook_TerrainPointOutcode);

    Log("[TerrainPointOutcode] ACTIVE on sub_7A61D0 (terrain point outcode in packed double).");
    if (g_abSubject) {
        Log("[TerrainPointOutcode]   under A/B test: its OFF stints run the client's original function.");
    }
    return true;
}

void Shutdown() {
    if (g_installed) {
        MH_DisableHook((void*)kTarget);
        g_installed = false;
        Log("[TerrainPointOutcode] Disabled.");
    }
}

void LogStats() {
    if (!g_installed && g_calls == 0) {
        return;
    }

    Log("[TerrainPointOutcode] calls=%llu (ctrl=%llu, fast=%llu), verified=%llu, mismatches=%llu (dead=%d)",
        g_calls, g_controlCalls, g_fastCalls, g_verified, g_mismatches, g_dead ? 1 : 0);
}

} // namespace TerrainPointOutcode
