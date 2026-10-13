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
// A/B harness off, 15000 evaluations a run, three runs each way (temporary
// timers, not in the tree):
//
//     quaternion evaluation, first touch    off 370 526 512 cycles   on 291 350 347
//     the same evaluation run again         off 187 198 198          on 188 168 165
//     the pass itself                       79 to 86 cycles a bone
//
// So the first touch is about twice the repeat, which is waiting for memory, and
// the pass takes about a third of it off the quaternion track. The pass is not
// free, and it has to win back more than its 80 cycles a bone across a bone's
// three tracks. An earlier version made the same loads as the evaluators in
// plain dependent order and cost 115 cycles a bone, because a load that misses
// sits at the head of the reorder buffer and stalls everything behind it. This
// one is a three-stage pipeline over the bones: each stage's loads are of lines
// the stage before prefetched six bones earlier, so none of them waits.
//
// Only the quaternion track was timed. The two vector tracks are assumed to
// behave alike; no frame time has been measured either way. A world has more
// models and a larger working set, which moves the misses and not the pass, so
// this is expected to do better there. That is an expectation, not a measurement.
//
// This runs once per model per pass, at the top of the bone loop. For each
// bone it prefetches the records, then the per-animation tables, then the
// timestamp and key lines, using the same animation index and hint the
// evaluators will use.
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

// The chain of one track in three stages, run for different bones at the same
// time (see Pass). A stage's loads are of lines the stage before it prefetched
// some bones earlier, so they are hits and nothing in the pass waits for memory:
// a load that missed would sit at the head of the reorder buffer and stall
// everything behind it, which is what a pass of plain dependent loads does.
//
//   stage 1  prefetch the bone record and the instance record
//   stage 2  the track's table of key lists and its entry list: prefetch
//            sub[anim] and entries[sel]
//   stage 3  sub[anim] -> the timestamps, entries[sel] -> the keys: prefetch
//            timestamps[hint] and keys[hint], both with the next one
__forceinline unsigned AnimOf(const uint8_t* track, const uint8_t* rec, unsigned animOff,
                              uint32_t nAnims) {
    uint32_t anim = *(const uint16_t*)(rec + animOff);
    if (*(const uint16_t*)(track + kT_globalSeq) != 0xFFFFu) anim = 0;
    else if (anim >= nAnims) anim = 0;
    return anim;
}

__forceinline void Stage2(const uint8_t* track, const uint8_t* rec, unsigned animOff) {
    const uint32_t nAnims = *(const uint32_t*)(track + kT_nAnims);
    if (nAnims == 0) return;
    const uint8_t* sub = *(const uint8_t* const*)(track + kT_subTracks);
    if (sub) _mm_prefetch((const char*)(sub + 8u * AnimOf(track, rec, animOff, nAnims)), _MM_HINT_T0);
    const uint32_t count = *(const uint32_t*)(track + kT_count);
    const uint8_t* entries = *(const uint8_t* const*)(track + kT_entries);
    if (entries && count) {
        const uint32_t want = *(const uint16_t*)(rec + kR_animId);
        _mm_prefetch((const char*)(entries + 8u * ((want < count) ? want : 0u)), _MM_HINT_T0);
    }
}

__forceinline void Stage3(const uint8_t* track, const uint8_t* rec, unsigned hintAt,
                          unsigned animOff, unsigned stride) {
    const uint32_t nAnims = *(const uint32_t*)(track + kT_nAnims);
    if (nAnims == 0) return;
    uint32_t hint = *(const uint32_t*)(rec + hintAt);

    const SubTrack* sub = *(const SubTrack* const*)(track + kT_subTracks);
    if (sub) {
        const SubTrack& st = sub[AnimOf(track, rec, animOff, nAnims)];
        if (st.nKeys > 1 && st.timestamps) {
            if (hint >= st.nKeys) hint = 0;
            _mm_prefetch((const char*)(st.timestamps + hint), _MM_HINT_T0);
            _mm_prefetch((const char*)(st.timestamps + hint + 1), _MM_HINT_T0);
        } else {
            hint = 0;
        }
    }

    const uint32_t count = *(const uint32_t*)(track + kT_count);
    const uint8_t* entries = *(const uint8_t* const*)(track + kT_entries);
    if (entries && count) {
        const uint32_t want = *(const uint16_t*)(rec + kR_animId);
        const uint32_t* entry = (const uint32_t*)(entries + 8u * ((want < count) ? want : 0u));
        if (entry[0]) {
            const uint8_t* keys = (const uint8_t*)entry[1];
            if (keys) {
                _mm_prefetch((const char*)(keys + stride * hint), _MM_HINT_T0);
                _mm_prefetch((const char*)(keys + stride * (hint + 1)), _MM_HINT_T0);
            }
        }
    }
}

constexpr uint32_t kDist = 6;   // bones between one stage and the next

void Pass(const uint8_t* model, const uint8_t* inst) {
    ++g_passes;
    if (g_faulted) return;
    if (g_abSubject && AbTest::StandAside()) { ++g_skippedAb; return; }

    const uint32_t count = *(const uint32_t*)(model + kM_boneCount);
    if (count == 0 || count > kMaxBones) return;

    __try {
        const uint8_t* bones = *(const uint8_t* const*)(model + kM_bones);
        const uint8_t* recs  = *(const uint8_t* const*)(inst + kI_records);
        if (!bones || !recs) return;

        const uint32_t total = count + 2 * kDist;
        for (uint32_t n = 0; n < total; ++n) {
            if (n < count) {
                const uint8_t* b = bones + (size_t)n * kB_size;
                const uint8_t* r = recs + (size_t)n * kR_size;
                _mm_prefetch((const char*)(b + kB_trans), _MM_HINT_T0);
                _mm_prefetch((const char*)(b + kB_scale), _MM_HINT_T0);
                _mm_prefetch((const char*)r, _MM_HINT_T0);
                _mm_prefetch((const char*)(r + 0x40), _MM_HINT_T0);
                _mm_prefetch((const char*)(r + 0x80), _MM_HINT_T0);
            }
            if (n >= kDist && n - kDist < count) {
                const uint32_t i = n - kDist;
                const uint8_t* b = bones + (size_t)i * kB_size;
                const uint8_t* r = recs + (size_t)i * kR_size;
                if (*(const uint16_t*)(r + kR_alias) == 0xFFFFu) {
                    Stage2(b + kB_trans, r, kR_animId);
                    Stage2(b + kB_rot,   r, kR_animId);
                    Stage2(b + kB_scale, r, kR_animId);
                    if (*(const float*)(r + kR_blend) != 0.0f) {
                        Stage2(b + kB_trans, r, kR_animId2);
                        Stage2(b + kB_rot,   r, kR_animId2);
                        Stage2(b + kB_scale, r, kR_animId2);
                    }
                }
            }
            if (n >= 2 * kDist && n - 2 * kDist < count) {
                const uint32_t i = n - 2 * kDist;
                const uint8_t* b = bones + (size_t)i * kB_size;
                const uint8_t* r = recs + (size_t)i * kR_size;
                if (*(const uint16_t*)(r + kR_alias) == 0xFFFFu) {
                    Stage3(b + kB_trans, r, kR_hintTrans, kR_animId, 12);
                    Stage3(b + kB_rot,   r, kR_hintRot,   kR_animId, 8);
                    Stage3(b + kB_scale, r, kR_hintScale, kR_animId, 12);
                    g_tracks += 3;
                    if (*(const float*)(r + kR_blend) != 0.0f) {
                        Stage3(b + kB_trans, r, kR_hintTrans + 4, kR_animId2, 12);
                        Stage3(b + kB_rot,   r, kR_hintRot + 4,   kR_animId2, 8);
                        Stage3(b + kB_scale, r, kR_hintScale + 4, kR_animId2, 12);
                        g_tracks += 3;
                    }
                }
            }
        }
        g_bones += count;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        InterlockedExchange(&g_faulted, 1);
    }
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
        "470 cycles and 330 with this, and the pass costs 80 to 86 cycles a bone. "
        "Nothing it reads is written, and a fault in it switches it off. Not "
        "measured in a world.%s",
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
