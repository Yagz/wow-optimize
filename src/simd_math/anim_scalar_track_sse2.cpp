// ============================================================================
// Module: anim_scalar_track_sse2
//
// Hardware double-precision SSE2 rewrite of the M2 scalar animation track
// evaluators:
//   sub_82AF40: packed int16 scalar tracks (transparency/alpha/FOV)
//   sub_82B340: float scalar tracks (colors/alpha channels)
//
// Dual-run verified against client output for bit-exact floating-point results.
// ============================================================================

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <emmintrin.h>
#include <cstdint>
#include <cstring>

#include "anim_scalar_track_sse2.h"
#include "m2_anim_find_key_fast.h"
#include "anim_find_key_core.h"
#include "MinHook.h"
#include "version.h"
#include "config.h"
#include "sampling_profiler.h"
#include "ab_test.h"
#include "session_verdict.h"

extern "C" void Log(const char* fmt, ...);

MH_STATUS WineSafe_CreateHook(void* target, void* detour, void** original);
MH_STATUS WO_EnableHook(void* target);

namespace AnimScalarTrack {

namespace {

constexpr uintptr_t kTrackScalar = 0x0082AF40;
constexpr uintptr_t kTrackColor  = 0x0082B340;
constexpr uintptr_t kFindKey     = 0x008284D0;

// Animation state layout
constexpr unsigned kS_timing1  = 0x40;
constexpr unsigned kS_trackIdx = 0x44;   // u16
constexpr unsigned kS_timing2  = 0x64;
constexpr unsigned kS_blendIdx = 0x68;   // u16
constexpr unsigned kS_blend    = 0xA8;   // float

// Track descriptor layout
constexpr unsigned kT_interp    = 0x00;  // u16; zero means take keyframe whole
constexpr unsigned kT_globalSeq = 0x02;  // u16; 0xFFFF means no global sequence
constexpr unsigned kT_count     = 0x0C;  // u32
constexpr unsigned kT_entries   = 0x10;  // u32; pointer to { u32 count; void* keys }

// sub_8284D0 is __thiscall with five stack arguments and cleans them itself.
typedef void* (__fastcall* findKey_fn)(void* obj, void* edx, void* timing, void* track,
                                       uint32_t* hint, uint32_t* second, float* frac);

typedef void (__cdecl* track_fn)(void* obj, void* state, void* track,
                                 uint32_t* out, const float* defVal);

// The key search, called directly once the hook on sub_8284D0 has proved the same
// code against the client for its whole learning phase (M2AnimFindKey::g_usable),
// and through the client's routine until then or after it retires. The direct
// call drops the trampoline, the detour's prologue and its counters.
__forceinline void FindKey(void* obj, void* timing, void* track,
                           uint32_t* hint, uint32_t* second, float* frac) {
    if (M2AnimFindKey::g_usable)
        M2AnimFindKeyCore::Run(obj, timing, track, hint, second, frac);
    else
        ((findKey_fn)kFindKey)(obj, nullptr, timing, track, hint, second, frac);
}

track_fn orig_ScalarTrack = nullptr;
track_fn orig_ColorTrack  = nullptr;

bool g_installed = false;
bool g_armed     = false;
bool g_abSubject = false;
bool g_dead      = false;

unsigned long g_callsScalar    = 0;
unsigned long g_callsColor     = 0;
unsigned long g_verifiedScalar = 0;
unsigned long g_verifiedColor  = 0;
unsigned long g_lerps          = 0;
unsigned long g_blends         = 0;

constexpr unsigned long kVerifyFirst  = 30000;
constexpr unsigned long kResampleMask = 8191;

// Evaluate packed int16 scalar track (sub_82AF40)
void EvaluateScalar(void* obj, uint8_t* state, uint8_t* track,
                    uint32_t* out, const float* defVal) {
    const uint32_t count   = *(const uint32_t*)(track + kT_count);
    const uint32_t entries = *(const uint32_t*)(track + kT_entries);
    const uint16_t interp  = *(const uint16_t*)(track + kT_interp);

    const uint16_t want = *(const uint16_t*)(state + kS_trackIdx);
    const uint32_t sel  = (want < count) ? want : 0u;
    const uint32_t* entry = (const uint32_t*)(entries + 8u * sel);

    float* o = (float*)(out + 2);
    const double scale = (double)*(const float*)0x009EA0B4;

    if (entry[0]) {
        uint32_t second = 0;
        float    frac   = 0.0f;
        FindKey(obj, state + kS_timing1, track, out, &second, &frac);
        const int16_t* keys = (const int16_t*)entry[1];
        if (interp == 0) {
            *o = (float)((double)keys[out[0]] * scale);
            return;
        }
        double a = (double)keys[out[0]] * scale;
        double b = (double)keys[second] * scale;
        *o = (float)(a + (b - a) * (double)frac);
        g_lerps++;
    } else {
        *o = *defVal;
        if (interp == 0) return;
    }

    const float blend = *(const float*)(state + kS_blend);
    if (blend == 0.0f || *(const uint16_t*)(track + kT_globalSeq) != 0xFFFFu) return;

    const uint16_t wantB = *(const uint16_t*)(state + kS_blendIdx);
    const uint32_t selB  = (wantB < count) ? wantB : 0u;
    const uint32_t* entryB = (const uint32_t*)(entries + 8u * selB);

    double v;
    if (entryB[0]) {
        uint32_t second = 0;
        float    frac   = 0.0f;
        FindKey(obj, state + kS_timing2, track, out + 1, &second, &frac);
        const int16_t* keysB = (const int16_t*)entryB[1];
        double aB = (double)keysB[out[1]] * scale;
        double bB = (double)keysB[second] * scale;
        v = aB + (bB - aB) * (double)frac;
    } else {
        v = (double)*defVal;
    }

    *o = (float)((double)*o + (v - (double)*o) * (double)blend);
    g_blends++;
}

// Evaluate float scalar track (sub_82B340)
void EvaluateColor(void* obj, uint8_t* state, uint8_t* track,
                   uint32_t* out, const float* defVal) {
    const uint32_t count   = *(const uint32_t*)(track + kT_count);
    const uint32_t entries = *(const uint32_t*)(track + kT_entries);
    const uint16_t interp  = *(const uint16_t*)(track + kT_interp);

    const uint16_t want = *(const uint16_t*)(state + kS_trackIdx);
    const uint32_t sel  = (want < count) ? want : 0u;
    const uint32_t* entry = (const uint32_t*)(entries + 8u * sel);

    float* o = (float*)(out + 2);

    if (entry[0]) {
        uint32_t second = 0;
        float    frac   = 0.0f;
        FindKey(obj, state + kS_timing1, track, out, &second, &frac);
        const float* keys = (const float*)entry[1];
        if (interp == 0) {
            *o = keys[out[0]];
            return;
        }
        double a = (double)keys[out[0]];
        double b = (double)keys[second];
        *o = (float)(a + (b - a) * (double)frac);
        g_lerps++;
    } else {
        *o = *defVal;
        if (interp == 0) return;
    }

    const float blend = *(const float*)(state + kS_blend);
    if (blend == 0.0f || *(const uint16_t*)(track + kT_globalSeq) != 0xFFFFu) return;

    const uint16_t wantB = *(const uint16_t*)(state + kS_blendIdx);
    const uint32_t selB  = (wantB < count) ? wantB : 0u;
    const uint32_t* entryB = (const uint32_t*)(entries + 8u * selB);

    double v;
    if (entryB[0]) {
        uint32_t second = 0;
        float    frac   = 0.0f;
        FindKey(obj, state + kS_timing2, track, out + 1, &second, &frac);
        const float* keysB = (const float*)entryB[1];
        double aB = (double)keysB[out[1]];
        double bB = (double)keysB[second];
        v = aB + (bB - aB) * (double)frac;
    } else {
        v = (double)*defVal;
    }

    *o = (float)((double)*o + (v - (double)*o) * (double)blend);
    g_blends++;
}

void __cdecl Hooked_ScalarTrackBody(void* obj, void* state, void* track,
                                   uint32_t* out, const float* defVal) {
    g_callsScalar++;

    if (g_dead || !out || !state || !track || !defVal) {
        orig_ScalarTrack(obj, state, track, out, defVal);
        return;
    }

    if (!g_armed || (g_callsScalar & kResampleMask) == 0) {
        uint32_t saved[3], theirs[3];
        memcpy(saved, out, sizeof(saved));
        orig_ScalarTrack(obj, state, track, out, defVal);
        memcpy(theirs, out, sizeof(theirs));
        memcpy(out, saved, sizeof(saved));

        EvaluateScalar(obj, (uint8_t*)state, (uint8_t*)track, out, defVal);
        g_verifiedScalar++;

        if (memcmp(out, theirs, sizeof(theirs)) != 0) {
            memcpy(out, theirs, sizeof(theirs));
            g_dead = true;
            Verdict::Add(Verdict::Bad,
                         "AnimScalarTrack (packed scalar) disagreed with client output");
            Log("[AnimScalarTrack] DISAGREED with client after %lu checks - "
                "retired for this session. Client gave %08X (hints %u/%u), "
                "vectorized gave %08X (hints %u/%u).",
                g_verifiedScalar, theirs[2], theirs[0], theirs[1],
                out[2], out[0], out[1]);
            return;
        }
        if (!g_armed && g_verifiedScalar >= kVerifyFirst) {
            g_armed = true;
            Log("[AnimScalarTrack] armed: %lu calls matched client bit for bit. "
                "Now evaluating directly and rechecking one call in %lu.",
                g_verifiedScalar, kResampleMask + 1);
        }
        return;
    }

    EvaluateScalar(obj, (uint8_t*)state, (uint8_t*)track, out, defVal);
}

void __cdecl Hooked_ColorTrackBody(void* obj, void* state, void* track,
                                  uint32_t* out, const float* defVal) {
    g_callsColor++;

    if (g_dead || !out || !state || !track || !defVal) {
        orig_ColorTrack(obj, state, track, out, defVal);
        return;
    }

    if (!g_armed || (g_callsColor & kResampleMask) == 0) {
        uint32_t saved[3], theirs[3];
        memcpy(saved, out, sizeof(saved));
        orig_ColorTrack(obj, state, track, out, defVal);
        memcpy(theirs, out, sizeof(theirs));
        memcpy(out, saved, sizeof(saved));

        EvaluateColor(obj, (uint8_t*)state, (uint8_t*)track, out, defVal);
        g_verifiedColor++;

        if (memcmp(out, theirs, sizeof(theirs)) != 0) {
            memcpy(out, theirs, sizeof(theirs));
            g_dead = true;
            Verdict::Add(Verdict::Bad,
                         "AnimScalarTrack (color scalar) disagreed with client output");
            Log("[AnimScalarTrack] Color track DISAGREED with client after %lu checks - "
                "retired for this session. Client gave %08X (hints %u/%u), "
                "vectorized gave %08X (hints %u/%u).",
                g_verifiedColor, theirs[2], theirs[0], theirs[1],
                out[2], out[0], out[1]);
            return;
        }
        return;
    }

    EvaluateColor(obj, (uint8_t*)state, (uint8_t*)track, out, defVal);
}

void __cdecl Hooked_ScalarTrack(void* obj, void* state, void* track,
                               uint32_t* out, const float* defVal) {
    if (g_abSubject && AbTest::StandAside()) {
        orig_ScalarTrack(obj, state, track, out, defVal);
        return;
    }
    const unsigned long long t = AbTest::TickIn();
    Hooked_ScalarTrackBody(obj, state, track, out, defVal);
    AbTest::TickOut(t);
}

void __cdecl Hooked_ColorTrack(void* obj, void* state, void* track,
                              uint32_t* out, const float* defVal) {
    if (g_abSubject && AbTest::StandAside()) {
        orig_ColorTrack(obj, state, track, out, defVal);
        return;
    }
    const unsigned long long t = AbTest::TickIn();
    Hooked_ColorTrackBody(obj, state, track, out, defVal);
    AbTest::TickOut(t);
}

}  // namespace

bool Init() {
    if (!Config::g_settings.OptAnimScalarTrack) {
        Log("[AnimScalarTrack] not installed: switched off.");
        return false;
    }

    if (!WowOpt_ClientPatchAllowed((const void*)kTrackScalar) ||
        !WowOpt_ClientPatchAllowed((const void*)kTrackColor)) {
        Log("[AnimScalarTrack] not installed: No Client Patches is active.");
        return false;
    }

    if (IsBadReadPtr((void*)kTrackScalar, 16) || IsBadReadPtr((void*)kTrackColor, 16) ||
        IsBadReadPtr((void*)kFindKey, 16)) {
        Log("[AnimScalarTrack] unreadable targets - not installing.");
        return false;
    }

    static const unsigned char kExp_TrackScalar[8] = { 0x55, 0x8B, 0xEC, 0x8B, 0x55, 0x0C, 0x0F, 0xB7 };
    static const unsigned char kExp_TrackColor[8]  = { 0x55, 0x8B, 0xEC, 0x51, 0x53, 0x8B, 0x5D, 0x0C };

    if (memcmp((const void*)kTrackScalar, kExp_TrackScalar, 8) != 0 ||
        memcmp((const void*)kTrackColor, kExp_TrackColor, 8) != 0) {
        Log("[AnimScalarTrack] unexpected prologue bytes - not installing.");
        return false;
    }

    int ok = 0;
    if (WineSafe_CreateHook((void*)kTrackScalar, (void*)Hooked_ScalarTrack,
                            (void**)&orig_ScalarTrack) == MH_OK &&
        WO_EnableHook((void*)kTrackScalar) == MH_OK) {
        ok++;
    }

    if (WineSafe_CreateHook((void*)kTrackColor, (void*)Hooked_ColorTrack,
                            (void**)&orig_ColorTrack) == MH_OK &&
        WO_EnableHook((void*)kTrackColor) == MH_OK) {
        ok++;
    }

    if (ok == 0) {
        Log("[AnimScalarTrack] not installed: MinHook failed on both track evaluators.");
        return false;
    }

    g_installed = true;
    SamplingProfiler::RegisterSelfSymbol("AnimScalarTrack_SSE2", (const void*)&Hooked_ScalarTrack);
    SamplingProfiler::RegisterSelfSymbol("AnimColorTrack_SSE2", (const void*)&Hooked_ColorTrack);
    g_abSubject = AbTest::IsSubject("AnimScalarTrack", &g_abSubject);

    Log("[AnimScalarTrack] ACTIVE on %d of 2 track evaluators (sub_82AF40 and sub_82B340). "
        "Evaluates packed int16 and float scalar tracks in hardware SSE2. "
        "Verifying the first %lu calls bit for bit with ongoing resampling.%s",
        ok, kVerifyFirst,
        Config::g_settings.OptAbTest
            ? " The A/B harness alternates the switch between stints."
            : "");
    return true;
}

void Shutdown() {
    if (!g_installed) return;
    MH_DisableHook((void*)kTrackScalar);
    MH_DisableHook((void*)kTrackColor);
    g_installed = false;
}

void LogStats() {
    if (!Config::g_settings.OptAnimScalarTrack) {
        Log("[AnimScalarTrack] not measured: switched off.");
        return;
    }
    if (!g_installed) {
        Log("[AnimScalarTrack] not measured: hooks are not installed.");
        return;
    }
    if (g_dead) {
        Log("[AnimScalarTrack] retired early: evaluation disagreed with the client.");
        return;
    }
    const unsigned long total = g_callsScalar + g_callsColor;
    if (total == 0) {
        Log("[AnimScalarTrack] measured and zero: the hooks are in and neither was reached.");
        return;
    }

    Log("[AnimScalarTrack] %lu calls (%lu scalar, %lu color), %lu lerps, %lu blends. "
        "%lu calls checked against client output bit for bit, hints included.",
        total, g_callsScalar, g_callsColor, g_lerps, g_blends,
        g_verifiedScalar + g_verifiedColor);
}

}  // namespace AnimScalarTrack
