// ============================================================================
// Module: batch_chain_prefetch
//
// The M2 batch loop of sub_823130 walks a sorted index array and, for each batch,
// follows a chain of dependent loads through four separate objects before the
// batch's own work starts:
//
//     index[n]                           sorted order, read in sequence
//       -> record = base + 68 * index    the batch record: type, pointer to obj1
//            -> obj1  (+10h flags, +2Ch obj2, +0A4h texture table, +2A8h)
//                 -> obj2  (+150h the skin data)
//                      -> skin data  (+50h, +54h, +84h, read by sub_81F450)
//
// Records are visited in sorted order, which is not memory order, and each of the
// four hops is a cache miss the previous one must finish before the next can
// start. The drain profile of 2026-10-04 has the first touch of these objects as
// separate peaks in sub_81F450, sub_825750, sub_685D40 and sub_4B6CB0, each
// 0.6 to 1.3% of executing time and none of them arithmetic.
//
// The loop body between one batch and the next is thousands of cycles of calls
// into the device layer, so a miss for a batch a few iterations ahead can be in
// flight behind it. This runs a pipeline over the sorted index: each iteration
// prefetches the record of the batch twelve ahead, the first object of the batch
// eight ahead, the second five ahead and the skin data two ahead. Every stage's
// loads are of lines the stage before prefetched three iterations earlier, so
// none of them waits; a load that missed would sit at the head of the reorder
// buffer and stall the work behind it.
//
// Patch: the first six bytes of the loop body at 0x008239E0,
// `mov edx,[ebp+10h] / mov eax,[edx+edi*4]`, become a jump to a thunk that calls
// the pipeline and then executes those two instructions. eax, ecx and edx are
// written before they are read in the body, so the call may use them.
//
// Measured at the login screen of the rig, A/B harness off, three runs of 55 s
// with the pipeline switched on and off at random for each loop (a temporary
// timer on the gap between consecutive batches, not in the tree). Cycles a
// batch, with and without, over about 45000 batches each way:
//
//     5743 / 6034     4334 / 4660     5477 / 5525
//
// 1 to 7% of the whole batch, which at the login screen is mostly the driver
// (98.6% of executing time there is inside d3d9.dll). On DXVK in a world the
// client's own share is larger, and this is expected to be a larger part of
// it; that is an expectation, not a measurement.
//
// It reads and never writes game data. Its loads are of pointers inside objects
// the loop reads itself for every batch that is processed, but a batch the loop
// skips may have a stale pointer in a field only a processed batch would read, so
// the reads are inside a structured exception handler and the first fault
// switches the module off. The prefetch instructions themselves cannot fault.
// ============================================================================

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <xmmintrin.h>
#include <cstdint>
#include <cstring>

#include "batch_chain_prefetch.h"
#include "version.h"
#include "config.h"
#include "sampling_profiler.h"
#include "ab_test.h"

extern "C" void Log(const char* fmt, ...);

namespace BatchChainPrefetch {

namespace {

constexpr uintptr_t kSite    = 0x008239E0;
constexpr uintptr_t kResume  = 0x008239E6;   // lea ebx,[edx+edi*4]
constexpr unsigned  kSiteLen = 6;
// mov edx,[ebp+10h] / mov eax,[edx+edi*4]
const unsigned char kSiteWant[kSiteLen] = { 0x8B, 0x55, 0x10, 0x8B, 0x04, 0xBA };
// lea ebx,[edx+edi*4] / mov edx,[ebp+0Ch]
const unsigned char kResumeWant[6] = { 0x8D, 0x1C, 0xBA, 0x8B, 0x55, 0x0C };

constexpr unsigned kRecSize  = 68;
constexpr unsigned kRec_obj1 = 4;
constexpr unsigned kO1_flags = 0x10;
constexpr unsigned kO1_obj2  = 0x2C;
constexpr unsigned kO1_tex   = 0xA4;
constexpr unsigned kO1_far   = 0x2A8;
constexpr unsigned kO2_skin  = 0x150;
constexpr unsigned kSk_a     = 0x50;
constexpr unsigned kSk_b     = 0x84;

constexpr uint32_t kDistRec  = 12;
constexpr uint32_t kDistObj1 = 8;
constexpr uint32_t kDistObj2 = 5;
constexpr uint32_t kDistSkin = 2;

bool g_installed = false;
bool g_abSubject = false;
volatile LONG g_faulted = 0;
unsigned char g_saved[kSiteLen];
uintptr_t g_resume = kResume;

// Plain counters; once per batch.
unsigned long g_calls = 0, g_loops = 0, g_skippedAb = 0;

__forceinline const uint8_t* RecordAt(const uint8_t* base, const uint32_t* idx, uint32_t k) {
    return base + (size_t)kRecSize * idx[k];
}

__forceinline void StageRec(const uint8_t* base, const uint32_t* idx, uint32_t k) {
    const uint8_t* rec = RecordAt(base, idx, k);
    _mm_prefetch((const char*)rec, _MM_HINT_T0);
    _mm_prefetch((const char*)(rec + 0x40), _MM_HINT_T0);
}

__forceinline void StageObj1(const uint8_t* base, const uint32_t* idx, uint32_t k) {
    const uint8_t* obj1 = *(const uint8_t* const*)(RecordAt(base, idx, k) + kRec_obj1);
    if (!obj1) return;
    _mm_prefetch((const char*)(obj1 + kO1_flags), _MM_HINT_T0);
    _mm_prefetch((const char*)(obj1 + kO1_obj2), _MM_HINT_T0);
    _mm_prefetch((const char*)(obj1 + kO1_tex), _MM_HINT_T0);
    _mm_prefetch((const char*)(obj1 + kO1_far), _MM_HINT_T0);
}

__forceinline void StageObj2(const uint8_t* base, const uint32_t* idx, uint32_t k) {
    const uint8_t* obj1 = *(const uint8_t* const*)(RecordAt(base, idx, k) + kRec_obj1);
    if (!obj1) return;
    const uint8_t* obj2 = *(const uint8_t* const*)(obj1 + kO1_obj2);
    if (obj2) _mm_prefetch((const char*)(obj2 + kO2_skin), _MM_HINT_T0);
    const uint8_t* tex = *(const uint8_t* const*)(obj1 + kO1_tex);
    if (tex) _mm_prefetch((const char*)tex, _MM_HINT_T0);
}

__forceinline void StageSkin(const uint8_t* base, const uint32_t* idx, uint32_t k) {
    const uint8_t* obj1 = *(const uint8_t* const*)(RecordAt(base, idx, k) + kRec_obj1);
    if (!obj1) return;
    const uint8_t* obj2 = *(const uint8_t* const*)(obj1 + kO1_obj2);
    if (!obj2) return;
    const uint8_t* skin = *(const uint8_t* const*)(obj2 + kO2_skin);
    if (!skin) return;
    _mm_prefetch((const char*)(skin + kSk_a), _MM_HINT_T0);
    _mm_prefetch((const char*)(skin + kSk_b), _MM_HINT_T0);
}

// Called once for every batch from the thunk. n is the batch's position in the
// sorted index, count the length of the index.
void __cdecl Ahead(const uint8_t* base, const uint32_t* idx, uint32_t n, uint32_t count) {
    ++g_calls;
    if (g_faulted || !base || !idx) return;
    if (g_abSubject && AbTest::StandAside()) { ++g_skippedAb; return; }
    __try {
        if (n == 0) {
            ++g_loops;
            // Prime the pipeline for the first iterations, in stage order so each
            // stage's loads hit what the one before just asked for.
            for (uint32_t k = 0; k < kDistRec && k < count; ++k) StageRec(base, idx, k);
            for (uint32_t k = 0; k < kDistObj1 && k < count; ++k) StageObj1(base, idx, k);
            for (uint32_t k = 0; k < kDistObj2 && k < count; ++k) StageObj2(base, idx, k);
            for (uint32_t k = 0; k < kDistSkin && k < count; ++k) StageSkin(base, idx, k);
        }
        if (n + kDistRec  < count) StageRec(base, idx, n + kDistRec);
        if (n + kDistObj1 < count) StageObj1(base, idx, n + kDistObj1);
        if (n + kDistObj2 < count) StageObj2(base, idx, n + kDistObj2);
        if (n + kDistSkin < count) StageSkin(base, idx, n + kDistSkin);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        InterlockedExchange(&g_faulted, 1);
    }
}

// Entered by a jump from 0x008239E0 with edi the batch's position, esi the draw
// context, ebp the frame: arg_4 (0Ch) the record array, arg_8 (10h) the sorted
// index, arg_C (14h) its length.
__declspec(naked) void Thunk() {
    __asm {
        push  dword ptr [ebp+14h]
        push  edi
        push  dword ptr [ebp+10h]
        push  dword ptr [ebp+0Ch]
        call  Ahead
        add   esp, 10h
        mov   edx, [ebp+10h]
        mov   eax, [edx+edi*4]
        jmp   dword ptr [g_resume]
    }
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
    if (!Config::g_settings.OptBatchChainPrefetch) return true;

    if (!BytesMatch(kSite, kSiteWant, kSiteLen) ||
        !BytesMatch(kResume, kResumeWant, sizeof(kResumeWant))) {
        Log("[BatchChainPrefetch] NOT patched: the bytes at 0x%08X are not the head of "
            "the batch loop this was read from, or another module has taken them.",
            (unsigned)kSite);
        return false;
    }
    if (!WowOpt_ClientPatchAllowed((const void*)kSite)) {
        Log("[BatchChainPrefetch] NOT patched: No Client Patches is on.");
        return false;
    }

    DWORD old = 0;
    if (!VirtualProtect((void*)kSite, kSiteLen, PAGE_EXECUTE_READWRITE, &old)) {
        Log("[BatchChainPrefetch] NOT patched: could not make 0x%08X writable",
            (unsigned)kSite);
        return false;
    }
    memcpy(g_saved, (const void*)kSite, kSiteLen);
    unsigned char patch[kSiteLen];
    patch[0] = 0xE9;
    *(int32_t*)(patch + 1) = (int32_t)((uintptr_t)&Thunk - (kSite + 5));
    patch[5] = 0x90;
    memcpy((void*)kSite, patch, kSiteLen);
    DWORD ignored = 0;
    VirtualProtect((void*)kSite, kSiteLen, old, &ignored);
    FlushInstructionCache(GetCurrentProcess(), (void*)kSite, kSiteLen);

    g_installed = true;
    g_abSubject = AbTest::IsSubject("BatchChainPrefetch", &g_abSubject);
    SamplingProfiler::RegisterSelfSymbol("BatchChainPrefetch", (const void*)&Thunk);
    Log("[BatchChainPrefetch] ACTIVE in the M2 batch loop (0x%08X). A pipeline over the "
        "sorted batch index prefetches the batch record %u ahead, its first object %u "
        "ahead, the second %u ahead and the skin data %u ahead, so the four dependent "
        "misses of a batch are in flight behind the work of earlier ones. Measured at the "
        "login screen: 1 to 7%% off the time of a batch, most of which is the driver there. "
        "Reads only, inside an exception handler; a fault switches it off. Not measured "
        "in a world.%s",
        (unsigned)kSite, (unsigned)kDistRec, (unsigned)kDistObj1, (unsigned)kDistObj2,
        (unsigned)kDistSkin, g_abSubject ? " Under A/B test." : "");
    return true;
}

void Shutdown() {
    if (!g_installed) return;
    DWORD old = 0;
    if (VirtualProtect((void*)kSite, kSiteLen, PAGE_EXECUTE_READWRITE, &old)) {
        memcpy((void*)kSite, g_saved, kSiteLen);
        DWORD ignored = 0;
        VirtualProtect((void*)kSite, kSiteLen, old, &ignored);
        FlushInstructionCache(GetCurrentProcess(), (void*)kSite, kSiteLen);
    }
    g_installed = false;
}

void LogStats() {
    if (!Config::g_settings.OptBatchChainPrefetch) return;
    if (!g_installed) {
        Log("[BatchChainPrefetch] not patched - the reason is at the top of this log");
        return;
    }
    if (g_calls == 0) {
        Log("[BatchChainPrefetch] patched, and the batch loop has not run yet.");
        return;
    }
    Log("[BatchChainPrefetch] %lu batch(es) in %lu loop(s) went through the pipeline%s%s. "
        "Plain counters, lower bounds.",
        g_calls, g_loops,
        g_skippedAb ? ", some left out for the A/B half" : "",
        g_faulted ? " - SWITCHED OFF by a fault" : "");
}

}  // namespace BatchChainPrefetch
