// ============================================================================
// Description: Caches the sort key a render-batch comparator re-derives.
// Safety & Threading: Main thread, inside the model render sort.
// ============================================================================
// sub_824B70 is 2.44% of executing time in a tester's uncapped, CPU-bound
// session - ninth in the profile, above every Lua entry. It is eighty-one
// instructions and does no arithmetic worth the name. It is a comparator.
//
// The client sorts render batches with it, from several inlined sort routines:
// sub_82BC20 calls it three times, sub_82E840 five or more, and there are others.
// So the same batch is compared over and over, and each comparison re-derives
// the same key from scratch.
//
// Deriving it costs five dependent loads per operand:
//
//     v3    = *(*a1 + 720)
//     table = *v3                              ; and base = v3[2]
//     idx16 = *(u16*)(table + 24 * a1[1] + 4)
//     desc  = base + 48 * idx16
//     key   = *(u16*)(desc + 16)
//
// Nothing can start until the one before it returns, and the profile's weight
// sits at 0x824B7E, which is the second link - `mov edx, [eax+720]`. That is not
// a busy function, it is a function waiting on memory.
//
// So the derived descriptor is remembered per (object, submesh index) and the
// key read straight out of it. Three of the five links go away.
// ---------------------------------------------------------------------------
// What the comparator actually says
//
// Written as the client's branches resolve, it is a plain lexicographic
// less-than over four keys:
//
//     key (u16 at desc+16), then *(obj+720), then *(obj+44), then the index
//
// all unsigned. Worth writing out because the disassembly expresses it as nested
// `>=` tests with fallthrough, which reads as though the equal cases go
// somewhere else, and they do not.
// ---------------------------------------------------------------------------
// Staleness, which is the whole risk
//
// A wrong key does not crash; it changes the sort order, and a wrong render
// order shows up as transparency drawn in the wrong sequence. That is the kind
// of defect nobody reports precisely, so the cache is bounded by something that
// cannot drift: a generation number bumped on the frame boundary, from
// WowOpt_OnFrameBoundary, which both present paths reach exactly once per
// presented frame. An entry from an earlier frame is not stale, it is invisible.
//
// Within a single frame the material a submesh points at does not move - the
// animation pass has finished before the render sort begins - so a hit inside
// the generation is answering with data derived this frame.
// ---------------------------------------------------------------------------
// Verification
//
// The comparator is pure. It reads memory and returns a bool, and it writes
// nothing at all - so unlike almost everything else replaced in this project,
// both versions can simply be run and their answers compared, with no saving
// and restoring and no predicting. The first calls do exactly that, and one in
// four thousand keeps doing it afterwards.
// ============================================================================

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstdint>

#include "m2_sort_key_cache.h"
#include "MinHook.h"
#include "version.h"
#include "config.h"
#include "sampling_profiler.h"
#include "ab_test.h"
#include "session_verdict.h"
#include "high_tables.h"
#include "self_bench.h"

extern "C" void Log(const char* fmt, ...);

MH_STATUS WineSafe_CreateHook(void* target, void* detour, void** original);
MH_STATUS WO_EnableHook(void* target);

namespace M2SortKey {

namespace {

constexpr uintptr_t kCompare = 0x00824B70;

// Offsets, all read off sub_824B70 itself.
constexpr unsigned kO_batchPtr = 720;   // object + 720: the batch block, or null
constexpr unsigned kO_altRoot  = 44;    // object + 44 when it is null
constexpr unsigned kO_desc     = 16;    // the u16 sort key inside the descriptor
constexpr unsigned kSubStride  = 24;    // submesh table stride
constexpr unsigned kSubIdxOff  = 4;     // u16 index within a submesh entry
constexpr unsigned kDescStride = 48;

typedef int (__stdcall* compare_fn)(void* a, void* b);
compare_fn orig_Compare = nullptr;

bool g_installed = false;
bool g_armed     = false;
// Set at init when the A/B harness names this module, so the hot path
// tests a plain bool instead of calling out on every invocation.
bool g_abSubject = false;
bool g_dead      = false;

// Plain 32-bit on a comparator's path. Lower bounds, and the report says so.
unsigned long g_calls    = 0;
unsigned long g_verified = 0;
// Comparisons where the two answers agree in AL and differ above it. That is
// the shape of a bool-returning client leaving its scratch in the register, and
// counting it is what keeps "only the low byte is the answer" a measurement
// rather than an assumption: if this stays at zero over a session, the high
// bits were never garbage and the reading is wrong.
unsigned long g_highBitsDiffered = 0;
unsigned long g_hits     = 0;
unsigned long g_misses   = 0;

constexpr unsigned long kVerifyFirst  = 20000;
constexpr unsigned long kResampleMask = 4095;

// Geometry, from the field rather than from a round number.
//
// This was 256 slots, direct-mapped, indexed by `((obj >> 4) ^ idx) & 255`, and
// it hits 62.5% - 575885710 lookups a session take the five dependent loads
// this module exists to avoid, and the profile's weight is on the second of
// them.
//
// Neither half of that index was doing much. A raw xor of a pointer field with
// a small submesh index preserves whatever structure the allocator gave the
// pointers, and `obj >> 4` under an 8-bit mask keeps only bits 4 to 11, so
// objects a few kilobytes apart contribute nothing to distinguish themselves.
//
// The size was the larger problem. A session runs 2056719638 key lookups at a
// 7.65 ms median frame, which is about 6500 a frame, and the generation counter
// empties the cache every frame - so the table has to hold one frame's distinct
// (object, submesh) pairs and nothing longer. A comparator revisits the same
// pairs many times over a sort, so the distinct count is well under the lookup
// count, but 256 is under it too, and that is what 62.5% means.
//
// A slot is sixteen bytes, so this is the cheapest table in the project to get
// wrong and the cheapest to fix: 8192 sets of 2 ways is 16384 entries and
// 256 KB. Two ways sit in the same cache line, so probing the second costs
// nothing a miss would not have cost anyway. It moves out of this DLL's image
// at the same time - 256 KB of static array would otherwise sit in the low half
// of the address space, which is the half the client allocates from.
constexpr unsigned kSets  = 8192;  // power of two
constexpr unsigned kWays  = 2;
constexpr unsigned kSlots = kSets * kWays;
struct Slot {
    uint32_t obj;
    uint32_t idx;
    uint32_t gen;
    uint32_t desc;
};
Slot*    g_slot = nullptr;
uint32_t g_gen = 1;                // never 0, so a zeroed slot cannot match
// Inserts that landed on a slot still live in this frame. Read against the
// misses, this says whether the geometry holds a frame's working set.
unsigned long g_conflict = 0;

// The client's derivation, verbatim. Returns 0 if it cannot be followed.
inline uint32_t DeriveDesc(uint32_t obj, uint32_t idx) {
    uint32_t batch = *(const uint32_t*)(obj + kO_batchPtr);
    uint32_t base, table;
    if (batch) {
        base  = *(const uint32_t*)(batch + 8);
        table = *(const uint32_t*)batch;
    } else {
        uint32_t root = *(const uint32_t*)(obj + kO_altRoot);
        base  = *(const uint32_t*)(root + 396);
        table = *(const uint32_t*)(*(const uint32_t*)(root + 368) + 40);
    }
    uint32_t sub = *(const uint16_t*)(table + kSubStride * idx + kSubIdxOff);
    return base + kDescStride * sub;
}

// The miss, kept out of line on purpose.
//
// The comparator runs 3175843467 times a session and asks for two descriptors
// each time, so whether the lookup inlines decides whether six billion calls
// happen. Growing this function for the set-associative table is exactly what
// stops the compiler inlining it - and the growth is all in the half that
// almost never runs once the cache is sized properly.
//
// So the hit stays in the caller and the miss becomes a call, the same split
// lua_getstr_inline uses for its chain walk. A call on the miss path costs
// nothing next to the five dependent loads it is about to do.
__declspec(noinline) uint32_t DescForMiss(uint32_t obj, uint32_t idx, Slot* set) {
    const uint32_t d = DeriveDesc(obj, idx);
    g_misses++;

    // Take a way that this frame has not claimed; only when both are live does
    // anything get displaced, and that is the number the report prints.
    unsigned pick = 0;
    for (unsigned w = 0; w < kWays; ++w) {
        if (set[w].gen != g_gen) { pick = w; break; }
        if (w == kWays - 1) { pick = (obj >> 2) & (kWays - 1); ++g_conflict; }
    }
    Slot& s = set[pick];
    s.obj = obj; s.idx = idx; s.desc = d; s.gen = g_gen;
    return d;
}

// __forceinline, not inline: plain inline is a request and MSVC declined it,
// leaving two calls per comparison on a path taken 3175843467 times a
// session. Verified in the object after changing it - the comparator now
// contains no call to this, only one to the miss.
__forceinline uint32_t DescFor(uint32_t obj, uint32_t idx) {
    // Knuth's constant on each half and a shift down, so neither the pointer's
    // low bits nor a small index decides the set on its own.
    uint32_t h = (obj >> 4) * 2654435761u;
    h ^= idx * 2246822519u;
    h ^= h >> 15;
    const unsigned base = (unsigned)((h >> 8) & (kSets - 1)) * kWays;

    Slot* set = &g_slot[base];
    for (unsigned w = 0; w < kWays; ++w) {
        Slot& s = set[w];
        if (s.gen == g_gen && s.obj == obj && s.idx == idx) {
            g_hits++;
            return s.desc;
        }
    }
    return DescForMiss(obj, idx, set);
}

// Lexicographic less-than over the four keys, in the client's order.
inline int Compare(void* a, void* b) {
    const uint32_t* pa = (const uint32_t*)a;
    const uint32_t* pb = (const uint32_t*)b;
    uint32_t oa = pa[0], ia = pa[1];
    uint32_t ob = pb[0], ib = pb[1];

    uint16_t ka = *(const uint16_t*)(DescFor(oa, ia) + kO_desc);
    uint16_t kb = *(const uint16_t*)(DescFor(ob, ib) + kO_desc);
    if (ka != kb) return ka < kb;

    uint32_t ba = *(const uint32_t*)(oa + kO_batchPtr);
    uint32_t bb = *(const uint32_t*)(ob + kO_batchPtr);
    if (ba != bb) return ba < bb;

    uint32_t ra = *(const uint32_t*)(oa + kO_altRoot);
    uint32_t rb = *(const uint32_t*)(ob + kO_altRoot);
    if (ra != rb) return ra < rb;

    return ia < ib;
}

}  // namespace

// Faults the guard caught, while verifying or after. The fallback is the
// client's own comparator, which reads the same two objects and the same
// descriptor chain, so a fault here is one the fallback would take too; the
// armed path runs without an exception frame once kVerifyFirst comparisons
// have run guarded and this is still zero. If it ever is not, every comparison
// stays guarded.
static unsigned long g_caught = 0;

static __declspec(noinline) int CompareGuarded(void* a, void* b) {
    __try {
        return Compare(a, b);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        ++g_caught;
        return orig_Compare(a, b);
    }
}

// Timed here and nowhere else, because this is the one path that runs both
// halves on the same input. This module replaces a comparator a profile
// measured at 2.44% of executing time, and in a later profile the detour itself
// is 2.88% - which says nothing about whether the cache is faster than the
// chain it removes, since the two numbers come from different machines. The
// pair below answers that directly, and costs two rdtsc on a path that was
// already doing the work twice.
int g_benchSlot = -1;

static __declspec(noinline) int CompareVerify(void* a, void* b) {
    {
        // Alternated, so neither half always meets the cold caches and the other the warm ones.
        static unsigned s_benchOrder = 0;
        int mine, theirs;
        if (++s_benchOrder & 1) {
            const unsigned long long tA = SelfBench::Now();
            theirs = orig_Compare(a, b);
            const unsigned long long tB = SelfBench::Now();
            __try {
                mine = Compare(a, b);
            } __except (EXCEPTION_EXECUTE_HANDLER) {
                ++g_caught;
                return theirs;
            }
            SelfBench::Pair(g_benchSlot, SelfBench::Now() - tB, tB - tA);
        } else {
            const unsigned long long tA = SelfBench::Now();
            __try {
                mine = Compare(a, b);
            } __except (EXCEPTION_EXECUTE_HANDLER) {
                ++g_caught;
                return orig_Compare(a, b);
            }
            const unsigned long long tB = SelfBench::Now();
            theirs = orig_Compare(a, b);
            SelfBench::Pair(g_benchSlot, tB - tA, SelfBench::Now() - tB);
        }
        g_verified++;

        // The client's comparator is a bool-returning predicate. Both of its
        // exits write one byte and neither clears the rest:
        //
        //     0x00824C26  pop edi ; pop esi ; mov al, 1  ; pop ebx ; pop ebp
        //     0x00824C56  pop edi ; pop esi ; xor al, al ; pop ebx ; pop ebp
        //
        // so the upper three bytes of EAX are whatever the compare chain last
        // put there. On the false exit that is a pointer - the paths into it
        // arrive from mov eax,[edi+2D0h] and mov eax,[edi+2Ch] - with its low
        // byte cleared by the xor.
        //
        // Reading all thirty-two bits reads that. Two field sessions retired
        // this module on its very first comparison, against 0x33C60000 and
        // 0x32A00000; the low byte of each is zero, so the client answered
        // false both times and so did this.
        if (mine != theirs && ((mine & 0xFF) != 0) == ((theirs & 0xFF) != 0))
            ++g_highBitsDiffered;

        if (((mine & 0xFF) != 0) != ((theirs & 0xFF) != 0)) {
            g_dead = true;
            Verdict::Add(Verdict::Bad,
                         "M2SortKey disagreed with the client and retired itself for "
                         "this session");
            Log("[M2SortKey] DISAGREED with the client after %lu comparisons - "
                "retired for this session, every comparison now goes to the "
                "client's own code. It answered 0x%08X and this answered "
                "0x%08X; only the low byte of each is the answer, the rest is "
                "whatever the client left in the register.",
                g_verified, (unsigned)theirs, (unsigned)mine);
            return theirs;
        }
        if (!g_armed && g_verified >= kVerifyFirst) {
            g_armed = true;
            Log("[M2SortKey] armed: %lu comparisons agreed with the client. Now "
                "answering directly, rechecking one in %lu, with %lu cache hits "
                "and %lu derivations so far.",
                g_verified, kResampleMask + 1, g_hits, g_misses);
        }
        return theirs;
    }
}

int __stdcall Hooked_CompareBody(void* a, void* b) {
    g_calls++;
    if (g_dead || !a || !b) return orig_Compare(a, b);
    if (!g_armed || (g_calls & kResampleMask) == 0) return CompareVerify(a, b);
    if (g_caught) return CompareGuarded(a, b);
    return Compare(a, b);
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
int __stdcall Hooked_Compare(void* a, void* b) {
    if (!g_abSubject) return Hooked_CompareBody(a, b);
    unsigned long long t = AbTest::TickIn();
    int r = AbTest::StandAside() ? orig_Compare(a, b)
                                       : Hooked_CompareBody(a, b);
    AbTest::TickOut(t);
    return r;
}

bool Init() {
    if (!Config::g_settings.OptM2SortKey) return true;

    g_slot = (Slot*)HighTables::Reserve("m2_sort_key", sizeof(Slot) * kSlots);
    if (!g_slot) {
        Log("[M2SortKey] no table - not installing. Without it every comparison "
            "takes the five dependent loads this module exists to avoid, which "
            "is slower than leaving the client alone.");
        return false;
    }

    if (IsBadReadPtr((void*)kCompare, 16)) {
        Log("[M2SortKey] 0x%08X unreadable - not installing", (unsigned)kCompare);
        return false;
    }
    static const unsigned char kExp_Compare[8] = { 0x55, 0x8B, 0xEC, 0x8B, 0x4D, 0x08, 0x8B, 0x01 };
    if (memcmp((const void*)kCompare, kExp_Compare, 8) != 0) {
        Log("[M2SortKey] 0x%08X bad prologue - not installing", (unsigned)kCompare);
        return false;
    }
    if (WineSafe_CreateHook((void*)kCompare, (void*)Hooked_Compare,
                            (void**)&orig_Compare) != MH_OK) {
        Log("[M2SortKey] hook NOT created");
        return false;
    }
    if (WO_EnableHook((void*)kCompare) != MH_OK) {
        Log("[M2SortKey] hook created but could not be enabled");
        return false;
    }

    g_abSubject = AbTest::IsSubject("M2SortKey", &g_abSubject);
    if (g_abSubject) {
        Log("[M2SortKey] under A/B test: it alternates on and off in stints "
            "and AbTest reports the frame times either way. The correctness "
            "checks are unaffected and still retire it on a disagreement.");
    }

    g_installed = true;
    SamplingProfiler::RegisterSelfSymbol("M2SortKey_Compare", (const void*)&Hooked_Compare);
    g_benchSlot = SelfBench::Register("M2SortKey");
    Log("[M2SortKey] ACTIVE on the render batch comparator (sub_824B70 @ "
        "0x%08X), 2.44%% of executing time in an uncapped tester session - ninth "
        "in the profile, above every Lua entry. It does no arithmetic; it derives "
        "one 16-bit key through five dependent loads and compares it, and the "
        "profile's weight sits on the second of those loads. The derived "
        "descriptor is remembered per object and submesh for the length of one "
        "frame, which removes three of the five. The comparator is pure, so both "
        "answers are simply compared for the first %lu calls and one in %lu after "
        "that - no saving, restoring or predicting needed.",
        (unsigned)kCompare, kVerifyFirst, kResampleMask + 1);
    return true;
}

void NewFrame() {
    if (!g_installed) return;
    // Bumping is the whole invalidation. An entry from an earlier frame does not
    // match and is overwritten in place, so nothing has to be cleared.
    if (++g_gen == 0) g_gen = 1;
}

void LogStats() {
    if (!Config::g_settings.OptM2SortKey) return;
    if (!g_installed) { Log("[M2SortKey] not installed - nothing measured"); return; }
    if (g_calls == 0) { Log("[M2SortKey] installed but never called"); return; }
    Log("[M2SortKey]   exception guard: %lu fault(s) caught; the armed path runs %s.",
        g_caught, !g_armed ? "guarded, still verifying"
                  : (g_caught ? "guarded, because the guard has caught something"
                              : "without an exception frame"));

    unsigned long looked = g_hits + g_misses;
    Log("[M2SortKey] %lu comparisons%s, %lu verified against the client. Key "
        "lookups: %lu from cache (%.1f%%), %lu derived through the full chain. "
        "Counts are lower bounds.",
        g_calls,
        g_dead ? " - RETIRED on a disagreement"
               : (g_armed ? "" : " - still verifying, the client still answers every one"),
        g_verified, g_hits,
        looked ? 100.0 * (double)g_hits / (double)looked : 0.0,
        g_misses);
    // Printed whether or not it fired. The cache is emptied every frame by the
    // generation counter, so a miss is either a pair this frame has not asked
    // for yet - unavoidable - or one displaced by another pair landing on the
    // same set. Only the second is something the geometry can fix, and a figure
    // near zero says the table now holds a frame and the remaining misses are
    // the floor.
    Log("[M2SortKey]   %lu of those misses displaced a pair this frame was still "
        "using, across %u sets of %u ways. Near zero means the table holds a "
        "frame's working set and what is left is the first look at each pair.",
        g_conflict, kSets, kWays);

    if (g_verified > 0) {
        if (g_highBitsDiffered > 0)
            Log("[M2SortKey]   %lu of those %lu agreed in the low byte and "
                "differed above it. That is the client leaving its scratch in "
                "the register above a one-byte answer, and reading the whole "
                "register is what retired this module in the field.",
                g_highBitsDiffered, g_verified);
        else
            Log("[Wrong] [M2SortKey] %lu comparisons and not one had garbage "
                "above the low byte. This module only compares the low byte "
                "because the client's answer was read as one; if that never "
                "happens the reading needs checking again.", g_verified);
    }
}

void Shutdown() {
    if (g_installed) MH_DisableHook((void*)kCompare);
}

}  // namespace M2SortKey
