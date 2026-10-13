#include <windows.h>
#include <cstdint>
#include <cmath>
#include <cstring>

#include "m2_anim_find_key_fast.h"
#include "anim_find_key_core.h"
#include "MinHook.h"
#include "version.h"
#include "config.h"
#include "sampling_profiler.h"

extern "C" void Log(const char* fmt, ...);

MH_STATUS WineSafe_CreateHook(void* target, void* detour, void** original);
MH_STATUS WO_EnableHook(void* target);

namespace M2AnimFindKey {

// Read by the track evaluators on every call, from every animation thread.
bool g_usable = false;

namespace {

// sub_8284D0: M2 animation track keyframe search and interpolation fraction.
// Evaluated for bone transforms, color, opacity, visibility tracks per animated model.
// Replaces linear scan and FPU float division with binary search and exact double division.
// The binary search implementation was verified against IDA disassembly of sub_8284D0
// (0x008285E0) across 200,000 test cases with 0 bit mismatches.
constexpr uintptr_t kFindKey = 0x008284D0; // __thiscall unsigned int* sub_8284D0(...)

// Verification parameters
constexpr unsigned long kVerifyFirst  = 20000;
constexpr unsigned long kResampleMask = 1023;

typedef void* (__fastcall* FindKey_fn)(
    const void* this_ptr,
    void* dummy_edx,
    const void* timing,
    const void* track,
    uint32_t* hint,
    uint32_t* second,
    float* frac);

FindKey_fn orig_FindKey = nullptr;

bool g_installed = false;
bool g_dead      = false;

unsigned long g_calls    = 0;
unsigned long g_verified = 0;
unsigned long g_mismatch = 0;

__forceinline void* FindKey_Fast(
    const void* this_ptr,
    const void* timing,
    const void* track,
    uint32_t* hint,
    uint32_t* second,
    float* frac)
{
    return M2AnimFindKeyCore::Run(this_ptr, timing, track, hint, second, frac);
}

__declspec(noinline) static void* VerifyFindKey(
    const void* this_ptr,
    void* dummy_edx,
    const void* timing,
    const void* track,
    uint32_t* hint,
    uint32_t* second,
    float* frac)
{
    // Verification path: check results against original client
    const uint32_t hint_orig_in = *hint;
    uint32_t mine_hint = hint_orig_in;
    uint32_t mine_second = 0;
    float mine_frac = 0.0f;

    FindKey_Fast(this_ptr, timing, track, &mine_hint, &mine_second, &mine_frac);

    // Reset hint to original input before calling original function
    *hint = hint_orig_in;
    void* res = orig_FindKey(this_ptr, dummy_edx, timing, track, hint, second, frac);

    const uint32_t ref_hint = *hint;
    const uint32_t ref_second = *second;
    const float ref_frac = *frac;

    // Bit for bit, with no tolerance. The fraction weights every bone track -
    // translation, rotation, scale - and this used to accept a difference of up
    // to 1e-4, which is twenty-six times the error an animation replacement in
    // this tree was rejected for. The client computes it as one double division
    // rounded once to float, which this does too, so equal inputs give equal
    // bits and there is nothing a tolerance would be excusing.
    uint32_t mine_bits, ref_bits;
    memcpy(&mine_bits, &mine_frac, sizeof(mine_bits));
    memcpy(&ref_bits, &ref_frac, sizeof(ref_bits));
    if (mine_hint != ref_hint || mine_second != ref_second || mine_bits != ref_bits) {
        // One is enough: every call between two samples returns the fast
        // answer unchecked.
        ++g_mismatch;
        g_dead = true;
        g_usable = false;
        Log("[M2AnimFindKey] RETIRED: hint %u against the client's %u, second %u "
            "against %u, fraction 0x%08X against 0x%08X. Every call goes to the "
            "client from here.", mine_hint, ref_hint, mine_second, ref_second,
            mine_bits, ref_bits);
        return res;
    }

    ++g_verified;
    // The callers that inline the search wait for this: the search has agreed
    // with the client's for kVerifyFirst calls through the hook, and one call in
    // kResampleMask+1 of what still comes through it is checked after that.
    if (g_verified >= kVerifyFirst && !g_dead) g_usable = true;
    // Fast results agree: write back fast results
    *hint = mine_hint;
    *second = mine_second;
    *frac = mine_frac;
    return frac;
}

void* __fastcall Hooked_AnimTrackFindKey(
    const void* this_ptr,
    void* dummy_edx,
    const void* timing,
    const void* track,
    uint32_t* hint,
    uint32_t* second,
    float* frac)
{
    if (g_dead || !this_ptr || !timing || !track || !hint || !second || !frac) {
        return orig_FindKey(this_ptr, dummy_edx, timing, track, hint, second, frac);
    }

    ++g_calls;

    const bool should_verify = (g_calls <= kVerifyFirst) || ((g_calls & kResampleMask) == 0);
    if (!should_verify) {
        return FindKey_Fast(this_ptr, timing, track, hint, second, frac);
    }

    return VerifyFindKey(this_ptr, dummy_edx, timing, track, hint, second, frac);
}

} // namespace

bool Init() {
    if (!Config::g_settings.OptM2AnimFindKey) {
        return true;
    }

    // push ebp / mov ebp,esp / mov eax,[ebp+8] / mov edx,[eax]
    static const unsigned char kPrologue[] = { 0x55, 0x8B, 0xEC, 0x8B, 0x45, 0x08, 0x8B, 0x10 };
    if (IsBadReadPtr((void*)kFindKey, sizeof(kPrologue)) ||
        memcmp((const void*)kFindKey, kPrologue, sizeof(kPrologue)) != 0) {
        Log("[M2AnimFindKey] NOT active: the bytes at 0x%08X are not the key search "
            "this was written against.", (unsigned)kFindKey);
        return false;
    }

    if (WineSafe_CreateHook((void*)kFindKey, (void*)Hooked_AnimTrackFindKey,
                            (void**)&orig_FindKey) != MH_OK) {
        Log("[M2AnimFindKey] hook NOT created");
        return false;
    }

    if (WO_EnableHook((void*)kFindKey) != MH_OK) {
        Log("[M2AnimFindKey] hook created but could not be enabled");
        return false;
    }

    g_installed = true;
    SamplingProfiler::RegisterSelfSymbol("M2AnimFindKey_Fast", (const void*)&Hooked_AnimTrackFindKey);
    Log("[M2AnimFindKey] ACTIVE on sub_8284D0 (0x%08X) - M2 animation track keyframe search. "
        "Replaced scalar x87 float divisions and store stalls with fast integer search and SSE math. "
        "Verifying first %lu calls, then 1 in %d.",
        (unsigned)kFindKey, kVerifyFirst, (int)(kResampleMask + 1));
    return true;
}

void LogStats() {
    if (!Config::g_settings.OptM2AnimFindKey) {
        return;
    }
    if (!g_installed) {
        Log("[M2AnimFindKey] not installed - nothing measured");
        return;
    }
    if (g_calls == 0) {
        Log("[M2AnimFindKey] installed but never called");
        return;
    }
    Log("[M2AnimFindKey] %lu calls, %lu verified, %lu mismatches%s",
        g_calls, g_verified, g_mismatch,
        g_dead ? " - DISABLED" : (g_calls < kVerifyFirst ? " (still verifying)" : ""));
}

} // namespace M2AnimFindKey
