// ============================================================================
// Module: collision_tri_test.cpp
//
// Fast collision triangle query frustum test and visited classification in sub_7C7660.
//
// In scene collision queries (wow_optimize_2026-09-22_17-28-51.log), sub_7C7660
// accounted for 2,893 samples (1.15% of executing main-thread time) at 0x007C76C1.
// It is called for each candidate triangle in a collision BSP leaf (sub_7C9A60)
// during camera and ray collision tests (sub_7CB0C0, sub_7CB180, etc.).
//
// Sibling function sub_7C7610 (CollisionResetVisited, 2.72% of executing time)
// clears the visited bits set by this function after each collision query.
//
// Bottlenecks in the client implementation (0x007C7660 -> 0x007C7708):
//   1. Unconditional 3-vertex frustum test: sub_7C7660 calls sub_7C71E0, which
//      calls sub_983D70 (CFrustum::IsPointVisible) three times unconditionally.
//      sub_983D70 evaluates 6 scalar x87 plane equations against the threshold
//      float at 0x00AA2E74. When vertex 0 is inside all frustum planes (outcode0 == 0),
//      the triangle cannot be culled by any plane, yet the client evaluates 12
//      more plane equations across vertices 1 and 2.
//   2. High call overhead: 4 separate function calls per candidate triangle
//      (sub_7C7660 -> sub_7C71E0 -> 3x sub_983D70) with stack pushes and x87
//      stack transitions.
//
// This replacement:
//   - Inlines the 6-plane outcode evaluation using IEEE double precision matching
//     client 53-bit x87 operation order:
//       (((plane.z * ptz + plane.y * pty) + plane.x * ptx) + plane.w) < threshold
//   - Short-circuits early: if outcode0 == 0, returns false immediately, skipping
//     vertices 1 and 2 entirely. If (outcode0 & outcode1) == 0, skips vertex 2.
//   - Unrolls all 6 plane evaluations, eliminating loop branches.
//   - Eliminates function call and frame overhead on the hot path.
//   - Zero /GS stack security cookies via __declspec(safebuffers).
//
// Verification:
//   sub_7C71E0 is a pure function that tests triangle vertices against frustum
//   planes without mutating state. During the learning phase (first 10,000 calls)
//   and periodic sampling (1 in 128 calls thereafter), the original client
//   sub_7C71E0 is executed and its return value compared against the fast result.
//   Any disagreement immediately logs both values and retires the hook.
// ============================================================================

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstdint>
#include <cstring>

#include "collision_tri_test.h"
#include "MinHook.h"
#include "version.h"
#include "config.h"
#include "ab_test.h"
#include "sampling_profiler.h"
#include "self_bench.h"

extern "C" void Log(const char* fmt, ...);
MH_STATUS WineSafe_CreateHook(void* target, void* detour, void** original);
MH_STATUS WO_EnableHook(void* target);

namespace CollisionTriTest {

namespace {

constexpr uintptr_t kTarget = 0x007C7660;
constexpr uintptr_t kFrustumTriTarget = 0x007C71E0;

// Prologue bytes of sub_7C7660:
// push ebp; mov ebp, esp; mov edx, [ecx+4]; push ebx; mov bx, [ebp+8]; movzx eax, bx; mov dl, [edx+eax*2]
static const uint8_t kExpectedPrologue[16] = {
    0x55, 0x8B, 0xEC, 0x8B, 0x51, 0x04, 0x53, 0x66,
    0x8B, 0x5D, 0x08, 0x0F, 0xB7, 0xC3, 0x8A, 0x14
};

// Prologue bytes of sub_7C71E0:
// push ebp; mov ebp, esp; sub esp, 4; mov ecx, [ebp+0Ch]; push esi; mov esi, [ebp+8]; lea eax, [ebp-4]
static const uint8_t kExpectedFrustumTriPrologue[16] = {
    0x55, 0x8B, 0xEC, 0x51, 0x8B, 0x4D, 0x0C, 0x56,
    0x8B, 0x75, 0x08, 0x8D, 0x45, 0xFC, 0x50, 0x51
};

// Global collision query state pointers:
static volatile uint32_t* const g_pVisitedCount  = (volatile uint32_t*)0x00D2DBF8;
static uint16_t* const          g_pVisitedIndices = (uint16_t*)0x00D25BF8;
static volatile uint32_t* const g_pAcceptedCount = (volatile uint32_t*)0x00D2DBFC;
static uint16_t* const          g_pAcceptedIndices = (uint16_t*)0x00D29BF8;
static const float* const       g_pThreshold      = (const float*)0x00AA2E74;

typedef char (__fastcall* TriTestFn)(void* self, void* dummy_edx, uint16_t tri_idx);
typedef bool (__cdecl* FrustumTriFn)(const float* frustum, const float* v0, const float* v1, const float* v2);

static TriTestFn g_orig = nullptr;
static FrustumTriFn g_origFrustumTri = nullptr;

static bool g_installed = false;
static bool g_dead = false;
static bool g_abSubject = false;
static int  g_benchId = -1;

// Verification schedule:
constexpr uint32_t kLearnCalls = 10000;
constexpr uint32_t kResampleMask = 0x7F;

// Diagnostic statistics:
static uint64_t g_calls = 0;
static uint64_t g_fastTests = 0;
static uint64_t g_filteredSkips = 0;
static uint64_t g_overflowSkips = 0;
static uint64_t g_acceptedTriangles = 0;
static uint64_t g_culledTriangles = 0;
static uint64_t g_earlyExitV0 = 0;
static uint64_t g_earlyExitV1 = 0;
static uint32_t g_verified = 0;
static uint32_t g_mismatches = 0;
static uint64_t g_controlCalls = 0;

static void Retire(const char* reason, uint16_t tri_idx, int fast_res, int client_res) {
    g_dead = true;
    ++g_mismatches;
    Log("[CollisionTriTest] RETIRED: %s on tri %u (fast=%d client=%d). All subsequent calls delegate to client.",
        reason, (unsigned)tri_idx, fast_res, client_res);
}

// Inlined 6-plane outcode evaluation in client 53-bit x87 operation order.
__declspec(safebuffers) static inline uint8_t Fast_PointOutcode(const float* frustum, const float* pt, double threshold) {
    const double px = (double)pt[0];
    const double py = (double)pt[1];
    const double pz = (double)pt[2];

    uint8_t code = 0;

    const double d0 = (((double)frustum[2] * pz + (double)frustum[1] * py) + (double)frustum[0] * px) + (double)frustum[3];
    if (d0 < threshold) code |= 1;

    const double d1 = (((double)frustum[6] * pz + (double)frustum[5] * py) + (double)frustum[4] * px) + (double)frustum[7];
    if (d1 < threshold) code |= 2;

    const double d2 = (((double)frustum[10] * pz + (double)frustum[9] * py) + (double)frustum[8] * px) + (double)frustum[11];
    if (d2 < threshold) code |= 4;

    const double d3 = (((double)frustum[14] * pz + (double)frustum[13] * py) + (double)frustum[12] * px) + (double)frustum[15];
    if (d3 < threshold) code |= 8;

    const double d4 = (((double)frustum[18] * pz + (double)frustum[17] * py) + (double)frustum[16] * px) + (double)frustum[19];
    if (d4 < threshold) code |= 16;

    const double d5 = (((double)frustum[22] * pz + (double)frustum[21] * py) + (double)frustum[20] * px) + (double)frustum[23];
    if (d5 < threshold) code |= 32;

    return code;
}

// Fast triangle frustum culling with early-exit short-circuiting.
__declspec(safebuffers) static inline bool Fast_TriangleFrustumCull(
    const float* frustum,
    const float* v0,
    const float* v1,
    const float* v2,
    double threshold)
{
    const uint8_t c0 = Fast_PointOutcode(frustum, v0, threshold);
    if (c0 == 0) {
        ++g_earlyExitV0;
        return false;
    }

    const uint8_t c1 = Fast_PointOutcode(frustum, v1, threshold);
    const uint8_t c01 = (uint8_t)(c0 & c1);
    if (c01 == 0) {
        ++g_earlyExitV1;
        return false;
    }

    const uint8_t c2 = Fast_PointOutcode(frustum, v2, threshold);
    return ((c01 & c2) != 0);
}

// Detour implementation for sub_7C7660.
// Callee pops 4 bytes via __fastcall (ECX = self, EDX = dummy, [esp+4] = tri_idx).
__declspec(safebuffers) static char __fastcall Hook_CollisionTriTest(void* self, void* dummy_edx, uint16_t tri_idx) {
    ++g_calls;

    if (g_dead) {
        return g_orig(self, dummy_edx, tri_idx);
    }

    if (g_abSubject && AbTest::StandAside()) {
        ++g_controlCalls;
        return g_orig(self, dummy_edx, tri_idx);
    }

    uint8_t* const pFlags = *(uint8_t**)((uint8_t*)self + 4);
    const uint8_t flag = pFlags[tri_idx * 2];
    const uint8_t mask = *(uint8_t*)((uint8_t*)self + 0x14);

    if ((flag & mask) != 0) {
        ++g_filteredSkips;
        return (char)tri_idx;
    }

    const uint32_t visited = *g_pVisitedCount;
    if (visited >= 0x2000) {
        ++g_overflowSkips;
        uint32_t* const pOverflow = *(uint32_t**)self;
        if (pOverflow) {
            *pOverflow |= 1;
        }
        return (char)tri_idx;
    }

    // Mark visited in global table and local flag byte:
    g_pVisitedIndices[visited] = tri_idx;
    *g_pVisitedCount = visited + 1;
    pFlags[tri_idx * 2] |= 0x80;

    // Fetch vertex coordinates:
    const uint16_t* const pIndices = *(const uint16_t**)((uint8_t*)self + 0x0C);
    const float* const pVertices = *(const float**)((uint8_t*)self + 0x08);
    const float* const pFrustum = *(const float**)((uint8_t*)self + 0x10);

    const uint16_t* const tri = pIndices + tri_idx * 3;
    const float* const v0 = pVertices + tri[0] * 3;
    const float* const v1 = pVertices + tri[1] * 3;
    const float* const v2 = pVertices + tri[2] * 3;

    const double threshold = (double)(*g_pThreshold);

    // Verification check. On a verified call the two halves alternate which runs
    // first, so neither always meets the cold caches and the other the warm ones.
    const bool needVerify = (g_verified < kLearnCalls) || ((g_calls & kResampleMask) == 0);
    static unsigned s_benchOrder = 0;
    const bool clientFirst = needVerify && g_origFrustumTri && (++s_benchOrder & 1);
    bool client_culled = false;
    uint64_t clientCycles = 0;
    if (clientFirst) {
        const uint64_t tClient0 = (g_benchId >= 0) ? SelfBench::Now() : 0;
        client_culled = g_origFrustumTri(pFrustum, v0, v1, v2);
        clientCycles = (g_benchId >= 0) ? (SelfBench::Now() - tClient0) : 0;
    }

    const uint64_t t0 = (g_benchId >= 0) ? SelfBench::Now() : 0;
    const bool fast_culled = Fast_TriangleFrustumCull(pFrustum, v0, v1, v2, threshold);
    const uint64_t fastCycles = (g_benchId >= 0) ? (SelfBench::Now() - t0) : 0;

    ++g_fastTests;

    if (needVerify && g_origFrustumTri) {
        if (!clientFirst) {
            const uint64_t tClient0 = (g_benchId >= 0) ? SelfBench::Now() : 0;
            client_culled = g_origFrustumTri(pFrustum, v0, v1, v2);
            clientCycles = (g_benchId >= 0) ? (SelfBench::Now() - tClient0) : 0;
        }

        if (fast_culled != client_culled) {
            Retire("frustum cull decision mismatch", tri_idx, (int)fast_culled, (int)client_culled);
            // Use client's decision for safety:
            if (!client_culled) {
                const uint32_t accepted = *g_pAcceptedCount;
                g_pAcceptedIndices[accepted] = tri_idx;
                *g_pAcceptedCount = accepted + 1;
                return (char)(*g_pAcceptedCount);
            }
            return 1;
        }

        ++g_verified;
        if (g_benchId >= 0 && clientCycles > 0) {
            SelfBench::Pair(g_benchId, fastCycles, clientCycles);
        }
    }

    if (!fast_culled) {
        ++g_acceptedTriangles;
        const uint32_t accepted = *g_pAcceptedCount;
        g_pAcceptedIndices[accepted] = tri_idx;
        *g_pAcceptedCount = accepted + 1;
        return (char)(*g_pAcceptedCount);
    }

    ++g_culledTriangles;
    return 1;
}

} // namespace

bool Init() {
    if (g_installed) return true;
    if (!Config::g_settings.OptCollisionTriTest) return true;

    if (std::memcmp((const void*)kTarget, kExpectedPrologue, sizeof(kExpectedPrologue)) != 0) {
        Log("[CollisionTriTest] NOT active: prologue at 0x%08X does not match expected bytes.", (unsigned)kTarget);
        return false;
    }

    if (std::memcmp((const void*)kFrustumTriTarget, kExpectedFrustumTriPrologue, sizeof(kExpectedFrustumTriPrologue)) != 0) {
        Log("[CollisionTriTest] WARNING: sub_7C71E0 prologue at 0x%08X mismatch; verifier will be disabled.", (unsigned)kFrustumTriTarget);
        g_origFrustumTri = nullptr;
    } else {
        g_origFrustumTri = (FrustumTriFn)kFrustumTriTarget;
    }

    if (Config::g_settings.OptNoClientPatches) {
        Log("[CollisionTriTest] NOT active: No Client Patches is on, and this hooks the client binary.");
        return false;
    }

    if (WineSafe_CreateHook((void*)kTarget, (void*)&Hook_CollisionTriTest, (void**)&g_orig) != MH_OK) {
        Log("[CollisionTriTest] NOT active: hook creation failed on 0x%08X.", (unsigned)kTarget);
        return false;
    }

    if (WO_EnableHook((void*)kTarget) != MH_OK) {
        Log("[CollisionTriTest] NOT active: hook enable failed on 0x%08X.", (unsigned)kTarget);
        return false;
    }

    g_installed = true;
    g_abSubject = AbTest::IsSubject("CollisionTriTest", &g_abSubject);
    g_benchId = SelfBench::Register("CollisionTriTest");
    SamplingProfiler::RegisterSelfSymbol("CollisionTriTest", (const void*)&Hook_CollisionTriTest);

    Log("[CollisionTriTest] ACTIVE on sub_7C7660 (0x%08X), 1.15%% of executing main-thread time in field sessions.", (unsigned)kTarget);
    if (g_abSubject) {
        Log("[CollisionTriTest]   under A/B test: the control half runs the client's original function.");
    }
    return true;
}

void Shutdown() {
    g_installed = false;
    g_dead = false;
}

void LogStats() {
    if (!Config::g_settings.OptCollisionTriTest) return;
    if (!g_installed) {
        Log("[CollisionTriTest] not installed - reason at top of log");
        return;
    }
    if (g_calls == 0) {
        Log("[CollisionTriTest] hooked, 0 calls reached so far.");
        return;
    }
    Log("[CollisionTriTest] %llu call(s): %llu fast tests (%llu accepted, %llu culled), "
        "%llu filtered, %llu overflow skips, %llu v0 early exits, %llu v1 early exits.",
        g_calls, g_fastTests, g_acceptedTriangles, g_culledTriangles,
        g_filteredSkips, g_overflowSkips, g_earlyExitV0, g_earlyExitV1);
    if (g_dead) {
        Log("[CollisionTriTest]   RETIRED after a mismatch; reason logged earlier.");
    } else if (g_verified < kLearnCalls) {
        Log("[CollisionTriTest]   %u of %u learning calls verified against client.", g_verified, kLearnCalls);
    } else {
        Log("[CollisionTriTest]   %u calls verified against client, none disagreed; running fast path.", g_verified);
    }
    if (g_controlCalls > 0) {
        Log("[CollisionTriTest]   %llu call(s) ran client code as A/B control.", g_controlCalls);
    }
}

} // namespace CollisionTriTest
