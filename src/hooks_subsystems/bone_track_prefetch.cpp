// ============================================================================
// Module: bone_track_prefetch
//
// The bone loop of sub_82F0F0 evaluates up to three animation tracks a bone, and
// each evaluation walks a chain of dependent loads before it has any arithmetic
// to do:
//
//     bone record (+10h translation, +24h rotation, +38h scale; 58h a bone)
//       -> track +8   the table of per-animation key lists
//            -> [anim].timestamps           (nKeys, pointer)
//                 -> timestamps[hint], timestamps[hint+1]
//       -> track +10h the table of per-animation entries
//            -> entry {count, keys}
//                 -> keys[hint], keys[hint+1]
//
// Four or five loads deep, each into a different allocation of the model's data,
// and the loop body between one bone's evaluation and the next is a thousand
// instructions long, so the out-of-order window never reaches the next bone's
// chain. Every first touch is a miss that nothing overlaps.
//
// What was measured, at the login screen of the rig with one model and the
// A/B harness off, 15000 evaluations a run, three runs each way:
//
//     quaternion evaluation, first touch    off 452 400 398 cycles   on 315 395 292
//     the same evaluation run again         off 209 239 190          on 182 193 178
//     the pass itself                       115.6 cycles a bone
//
// So the first touch is about twice the repeat, which is waiting for memory, and
// the pass takes about a fifth of that off the quaternion track. The pass is not
// free: at 115 cycles a bone it has to win back more than that across a bone's
// three tracks, and at the login screen it does by a margin that is small and
// has not been measured as a frame time. A world has more models and a larger
// working set, which moves the misses and not the pass, so this is expected to
// be the lower end; that is an expectation, not a measurement.
//
// This runs once per model per pass, at the top of the bone loop, and does what
// the loop will do a moment later for every bone, in a tight loop with no
// arithmetic between the bones: it follows each track's chain with the same loads
// the evaluator makes (the same animation index, the same hint) and prefetches
// the key lines at the end. The bones are independent, so up to ten chains are in
// flight at once instead of one.
//
// Nothing here writes anything the client reads: it reads the bone records, the
// instance records and the track tables, and its only side effect is the cache.
// Reads are guarded by a structured exception handler, because it walks tracks
// the loop may skip; the first fault switches the module off for the session.
//
// Patch: the five bytes at 0x0082F418, `mov edx,[ebx+2Ch] / cmp edx,edi`, are
// replaced with a jump to a thunk that calls the pass and then executes those
// two instructions. 0x0082F418 is reached once a call, from the fall-through and
// two jumps; the loop's back edge goes to 0x0082F426. M2AnimStride and M2AnimReuse
// patch the same bytes, so this refuses to install while either is on.
// ============================================================================

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <xmmintrin.h>
#include <cstdint>
#include <cstring>

#include "bone_track_prefetch.h"
#include "version.h"
#include "config.h"
#include "sampling_profiler.h"
#include "ab_test.h"

extern "C" void Log(const char* fmt, ...);

namespace BoneTrackPrefetch {

namespace {

constexpr uintptr_t kSite    = 0x0082F418;
constexpr uintptr_t kResume  = 0x0082F41D;   // mov [ebp+arg_10], edi
constexpr unsigned  kSiteLen = 5;
// mov edx,[ebx+2Ch] / cmp edx,edi
const unsigned char kSiteWant[kSiteLen] = { 0x8B, 0x53, 0x2C, 0x3B, 0xD7 };
// mov [ebp+arg_10], edi / jbe 0x008302A3 follows
const unsigned char kResumeWant[3] = { 0x89, 0x7D, 0x18 };

// Model data
constexpr unsigned kM_boneCount = 0x2C;
constexpr unsigned kM_bones     = 0x30;   // 58h a bone
// Instance (esi)
constexpr unsigned kI_records   = 0x94;   // AC a bone
// Bone record
constexpr unsigned kB_size      = 0x58;
constexpr unsigned kB_trans     = 0x10;
constexpr unsigned kB_rot       = 0x24;
constexpr unsigned kB_scale     = 0x38;
// Instance bone record
constexpr unsigned kR_size      = 0xAC;
constexpr unsigned kR_alias     = 0x48;   // u16; 0xFFFF = an own record, else a copy of another
constexpr unsigned kR_animId    = 0x44;   // u16
constexpr unsigned kR_hintTrans = 0x00;
constexpr unsigned kR_hintRot   = 0x14;
constexpr unsigned kR_hintScale = 0x2C;
constexpr unsigned kR_blend     = 0xA8;   // float
constexpr unsigned kR_animId2   = 0x68;   // u16
// Track
constexpr unsigned kT_globalSeq = 0x02;
constexpr unsigned kT_nAnims    = 0x04;
constexpr unsigned kT_subTracks = 0x08;
constexpr unsigned kT_count     = 0x0C;
constexpr unsigned kT_entries   = 0x10;

constexpr unsigned kMaxBones = 1024;

struct SubTrack { uint32_t nKeys; const uint32_t* timestamps; };

bool g_installed  = false;
bool g_abSubject  = false;
volatile LONG g_faulted = 0;
unsigned char g_saved[kSiteLen];
uintptr_t g_resume = kResume;

// Plain counters; once per model per pass.
unsigned long g_passes = 0, g_bones = 0, g_tracks = 0, g_skippedAb = 0;

// Where the loads' results go, so the loop that makes them is not removed.
volatile uint32_t g_sink = 0;

// One track's chain, in the loads the evaluator makes. `rec` is the instance
// record, `hintAt` the hint's offset in it, `stride` the key size in bytes.
__forceinline uint32_t Touch(const uint8_t* track, const uint8_t* rec, unsigned hintAt,
                             unsigned animOff, unsigned stride) {
    uint32_t acc = 0;
    const uint32_t nAnims = *(const uint32_t*)(track + kT_nAnims);
    if (nAnims == 0) return 0;

    uint32_t anim = *(const uint16_t*)(rec + animOff);
    if (*(const uint16_t*)(track + kT_globalSeq) != 0xFFFFu) anim = 0;
    else if (anim >= nAnims) anim = 0;

    uint32_t hint = *(const uint32_t*)(rec + hintAt);

    const SubTrack* sub = *(const SubTrack* const*)(track + kT_subTracks);
    uint32_t nKeys = 0;
    if (sub) {
        nKeys = sub[anim].nKeys;
        const uint32_t* ts = sub[anim].timestamps;
        if (nKeys > 1 && ts) {
            if (hint >= nKeys) hint = 0;
            acc += ts[hint];
            if (hint + 1 < nKeys) acc += ts[hint + 1];
        }
    }

    const uint32_t count = *(const uint32_t*)(track + kT_count);
    const uint8_t* entries = *(const uint8_t* const*)(track + kT_entries);
    if (entries && count) {
        const uint32_t want = *(const uint16_t*)(rec + kR_animId);
        const uint32_t sel = (want < count) ? want : 0u;
        const uint32_t* entry = (const uint32_t*)(entries + 8u * sel);
        if (entry[0]) {
            const uint8_t* keys = (const uint8_t*)entry[1];
            if (keys) {
                _mm_prefetch((const char*)(keys + stride * hint), _MM_HINT_T0);
                _mm_prefetch((const char*)(keys + stride * (hint + 1)), _MM_HINT_T0);
            }
        }
    }
    return acc;
}

void Pass(const uint8_t* model, const uint8_t* inst) {
    ++g_passes;
    if (g_faulted) return;
    if (g_abSubject && AbTest::StandAside()) { ++g_skippedAb; return; }

    const uint32_t count = *(const uint32_t*)(model + kM_boneCount);
    if (count == 0 || count > kMaxBones) return;

    uint32_t acc = 0;
    __try {
        const uint8_t* bones = *(const uint8_t* const*)(model + kM_bones);
        const uint8_t* recs  = *(const uint8_t* const*)(inst + kI_records);
        if (!bones || !recs) return;

        for (uint32_t i = 0; i < count; ++i) {
            const uint8_t* b = bones + (size_t)i * kB_size;
            const uint8_t* r = recs + (size_t)i * kR_size;
            if (*(const uint16_t*)(r + kR_alias) != 0xFFFFu) {
                // A copy of another record's animation state: the client copies
                // it across and evaluates nothing for this bone.
                continue;
            }
            acc += Touch(b + kB_trans, r, kR_hintTrans, kR_animId, 12);
            acc += Touch(b + kB_rot,   r, kR_hintRot,   kR_animId, 8);
            acc += Touch(b + kB_scale, r, kR_hintScale, kR_animId, 12);
            // The blend animation's chain, only for a bone that is blending.
            if (*(const float*)(r + kR_blend) != 0.0f) {
                acc += Touch(b + kB_trans, r, kR_hintTrans + 4, kR_animId2, 12);
                acc += Touch(b + kB_rot,   r, kR_hintRot + 4,   kR_animId2, 8);
                acc += Touch(b + kB_scale, r, kR_hintScale + 4, kR_animId2, 12);
                g_tracks += 3;
            }
            g_tracks += 3;
        }
        g_bones += count;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        InterlockedExchange(&g_faulted, 1);
    }
    g_sink = acc;
}

void __cdecl PassFromThunk(const uint8_t* model, const uint8_t* inst) {
    Pass(model, inst);
}

// Entered by a jump from 0x0082F418. ebx is the model data, esi the instance,
// edi zero, and the x87 stack holds values the function will use, which the
// pass does not touch. eax, ecx and edx are written before they are read again.
__declspec(naked) void Thunk() {
    __asm {
        push  esi
        push  ebx
        call  PassFromThunk
        add   esp, 8
        mov   edx, [ebx+2Ch]
        cmp   edx, edi
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
    if (!Config::g_settings.OptBoneTrackPrefetch) return true;

    if (Config::g_settings.OptM2AnimStride || Config::g_settings.OptM2AnimReuse) {
        Log("[BoneTrackPrefetch] NOT patched: M2AnimStride or M2AnimReuse is on and "
            "both write the same five bytes.");
        return false;
    }
    if (!BytesMatch(kSite, kSiteWant, kSiteLen) ||
        !BytesMatch(kResume, kResumeWant, sizeof(kResumeWant))) {
        Log("[BoneTrackPrefetch] NOT patched: the bytes at 0x%08X are not the head of "
            "the bone loop this was read from, or another module has taken them.",
            (unsigned)kSite);
        return false;
    }
    if (!WowOpt_ClientPatchAllowed((const void*)kSite)) {
        Log("[BoneTrackPrefetch] NOT patched: No Client Patches is on.");
        return false;
    }

    DWORD old = 0;
    if (!VirtualProtect((void*)kSite, kSiteLen, PAGE_EXECUTE_READWRITE, &old)) {
        Log("[BoneTrackPrefetch] NOT patched: could not make 0x%08X writable",
            (unsigned)kSite);
        return false;
    }
    memcpy(g_saved, (const void*)kSite, kSiteLen);
    unsigned char patch[kSiteLen];
    patch[0] = 0xE9;
    *(int32_t*)(patch + 1) = (int32_t)((uintptr_t)&Thunk - (kSite + 5));
    memcpy((void*)kSite, patch, kSiteLen);
    DWORD ignored = 0;
    VirtualProtect((void*)kSite, kSiteLen, old, &ignored);
    FlushInstructionCache(GetCurrentProcess(), (void*)kSite, kSiteLen);

    g_installed = true;
    g_abSubject = AbTest::IsSubject("BoneTrackPrefetch", &g_abSubject);
    SamplingProfiler::RegisterSelfSymbol("BoneTrackPrefetch", (const void*)&Thunk);
    Log("[BoneTrackPrefetch] ACTIVE at the top of the bone loop (0x%08X). Follows every "
        "bone's three track chains with the loads the evaluators will make and "
        "prefetches the key lines, so the misses of different bones overlap. "
        "At the login screen the first evaluation of a quaternion track costs about "
        "420 cycles and 320 with this, and the pass costs 115 cycles a bone. Nothing "
        "it reads is written, and a fault in it switches it off. Not measured in a "
        "world.%s",
        (unsigned)kSite, g_abSubject ? " Under A/B test." : "");
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
    if (!Config::g_settings.OptBoneTrackPrefetch) return;
    if (!g_installed) {
        Log("[BoneTrackPrefetch] not patched - the reason is at the top of this log");
        return;
    }
    if (g_passes == 0) {
        Log("[BoneTrackPrefetch] patched, and no model has been animated yet.");
        return;
    }
    Log("[BoneTrackPrefetch] %lu model pass(es), %lu bone(s), %lu track chain(s) followed%s%s. "
        "Plain counters, lower bounds; several threads may add to them.",
        g_passes, g_bones, g_tracks,
        g_skippedAb ? ", with some passes left out for the A/B half" : "",
        g_faulted ? " - SWITCHED OFF by a fault" : "");
}

}  // namespace BoneTrackPrefetch
