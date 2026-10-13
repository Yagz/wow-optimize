// ============================================================================
// Module: ribbon_edge_loop.cpp
//
// The edge loop inside the ribbon emitter update, sub_980090, which the
// instance update sub_828A00 calls once for every ribbon emitter of every
// model it updates. Weapon trails, spell trails and the like. For every edge
// of the ring the client ages it, lifts two of its coordinates, rebuilds the
// emitter's bounding box and rewrites four texture coordinates, and the
// bounding box is twelve fcomp / fnstsw / test / branch groups on values it has
// just stored. The number of edges is the emitter's lifetime over the frame
// time, so the loop grows with the frame rate.
//
// The loop is 0x00980341 to 0x009804EA. Its entry sits at 0x0098032F, where
// the client tests whether the ring is empty and leaves for 0x009804F2 if so.
// Those twelve bytes are replaced with a jump to a thunk that makes the same
// test, hands the object and the one value the client holds on its x87 stack
// (the time step, a double) to RibbonEdgeModel::Run, and rejoins the client at
// 0x009804F2 with that value still on the stack, which is where the client's
// own loop leaves it. Run does the same arithmetic in packed single compares and
// scalar double, and declines, touching nothing, in the cases listed in
// ribbon_edge_model.h; the thunk then falls into the client's loop, which is
// still in place behind the patch.
//
// What was checked, and how:
//
//   - the client's 439 loop bytes were copied instruction for instruction into
//     tools/ribbon_edge_harness and compared with the model over six million
//     random emitters (about a hundred million edges), with NaN, infinities,
//     denormals and signed zeros in every field. The one difference found was
//     the sign of a zero when the time step was 1e-300, which is why Run
//     declines outside 1e-30..1e30. Otherwise no difference, apart from which
//     of two NaNs an addition keeps;
//   - the same harness prints a checksum of one fixed emitter for both, and the
//     module compares its own at startup (kFixedChecksum);
//   - at startup the client's own function is run on a fixed emitter built
//     here, once before the entry is written and once through the patch, and
//     the two images compared. That exercises the thunk, the model and the
//     rejoin from the real prologue to the real return;
//   - the x87 stack is balanced on every harness call;
//   - the 12 entry bytes, the 6 rejoin bytes and a hash of the 439 loop bytes
//     are compared with what this was read from before anything is written.
//
// The function's return value is whatever the last compare left in eax, and
// its only caller (0x00828E8A) reloads eax straight after. ecx is reloaded at
// the rejoin and edx is dead there, so the thunk does not preserve them.
//
// The x87 control word and MXCSR are read on every call: the arithmetic above
// is the client's only at 53-bit precision, round to nearest, with denormals
// kept.
// ============================================================================

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstdint>
#include <cstring>
#include <xmmintrin.h>

#include "ribbon_edge_loop.h"
#include "ribbon_edge_model.h"
#include "version.h"
#include "config.h"
#include "sampling_profiler.h"

extern "C" void Log(const char* fmt, ...);

namespace RibbonEdgeLoop {

namespace {

constexpr uintptr_t kEntry   = 0x0098032F;
constexpr uintptr_t kLoopTop = 0x0098033B;   // fld flt_A4040C, the client's loop
constexpr uintptr_t kRejoin  = 0x009804F2;   // mov ecx,[esi+160h]
constexpr unsigned  kEntryLen = 12;

// mov edx,[esi+18h] / cmp edx,[esi+14h] / jz 0x009804F2
const unsigned char kEntryWant[kEntryLen] = {
    0x8B, 0x56, 0x18, 0x3B, 0x56, 0x14, 0x0F, 0x84, 0xB7, 0x01, 0x00, 0x00
};
// mov ecx,[esi+160h]
const unsigned char kRejoinWant[6] = { 0x8B, 0x8E, 0x60, 0x01, 0x00, 0x00 };

// FNV-1a over 0x0098033B..0x009804F2 in the client this was read from.
constexpr uint32_t kLoopHash = 0xB9D60CCCu;

// RibbonEdgeModel::FixedScenario for the client's own sequence, from the harness.
constexpr uint32_t kFixedChecksum = 0x882309FBu;

bool g_patched = false;
unsigned char g_saved[kEntryLen];
uintptr_t g_loopTop = kLoopTop;
uintptr_t g_rejoin  = kRejoin;

// Plain counters, once per emitter per update. Lower bounds.
unsigned long g_calls    = 0;
unsigned long g_edges    = 0;
unsigned long g_declined = 0;
unsigned long g_badMode  = 0;

uint32_t Fnv(const unsigned char* p, size_t n) {
    uint32_t h = 0x811C9DC5u;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x01000193u;
    return h;
}

bool BytesMatch(uintptr_t addr, const unsigned char* want, size_t n) {
    __try {
        return memcmp((const void*)addr, want, n) == 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

bool LoopHashMatches() {
    __try {
        return Fnv((const unsigned char*)kLoopTop, kRejoin - kLoopTop) == kLoopHash;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// 1 when the arithmetic below is the client's: x87 at 53 bits and round to
// nearest, SSE round to nearest with no flush to zero and no denormals-are-zero.
__forceinline bool ModeIsTheClients() {
    unsigned short cw = 0;
    __asm fnstcw cw
    if ((cw & 0x0F00u) != 0x0200u) return false;
    return (_mm_getcsr() & 0xE040u) == 0;
}

// Called from the thunk with the object in esi and the time step off the x87
// stack. Returns RibbonEdgeModel::Result; kDeclined sends the client's loop on.
RibbonEdgeModel::Result __cdecl RunEdges(unsigned char* obj, double dt) {
    ++g_calls;
    if (!ModeIsTheClients()) {
        ++g_badMode;
        return RibbonEdgeModel::kDeclined;
    }
    const RibbonEdgeModel::Result r = RibbonEdgeModel::Run(obj, dt);
    if (r == RibbonEdgeModel::kDeclined) {
        ++g_declined;
    } else {
        const uint32_t cap  = *reinterpret_cast<const uint32_t*>(obj + RibbonEdgeModel::kCapacity);
        const uint32_t tail = *reinterpret_cast<const uint32_t*>(obj + RibbonEdgeModel::kTail);
        // The loop walked from the head round the ring to the tail and changes
        // neither.
        const uint32_t head = *reinterpret_cast<const uint32_t*>(obj + RibbonEdgeModel::kHead);
        g_edges += (tail >= head) ? (tail - head) : (tail + cap - head);
    }
    return r;
}

// Entered by a jump from 0x0098032F. esi is the emitter and the x87 stack holds
// the time step. The rejoin wants one value on the x87 stack, the one the
// client's own loop leaves there, so the time step is stored, not popped.
__declspec(naked) void Thunk() {
    __asm {
        mov   edx, [esi+18h]
        cmp   edx, [esi+14h]
        jz    Lrejoin
        sub   esp, 8
        fst   qword ptr [esp]
        push  esi
        call  RunEdges
        add   esp, 12
        test  eax, eax
        jnz   Lrejoin
        mov   edx, [esi+18h]
        jmp   dword ptr [g_loopTop]
      Lrejoin:
        jmp   dword ptr [g_rejoin]
    }
}

void __cdecl RunForSelfTest(unsigned char* obj, double dt) {
    RibbonEdgeModel::Run(obj, dt);
}

bool WriteEntry(const unsigned char* bytes) {
    DWORD old = 0;
    if (!VirtualProtect((void*)kEntry, kEntryLen, PAGE_EXECUTE_READWRITE, &old)) return false;
    memcpy((void*)kEntry, bytes, kEntryLen);
    DWORD ignored = 0;
    VirtualProtect((void*)kEntry, kEntryLen, old, &ignored);
    FlushInstructionCache(GetCurrentProcess(), (void*)kEntry, kEntryLen);
    return true;
}

constexpr uintptr_t kClientFunction = 0x00980090;
typedef void(__thiscall* ClientFunction)(void* self, float dt, int skipBuild);

// Runs the client's own function, whatever is at its entry now, on a fixed
// emitter built here and reports a checksum of what it wrote. With the entry
// unpatched that is the client's loop; with it patched it is the thunk, the
// model and the rejoin, from the real prologue to the real return, so a wrong
// register, a wrong stack depth or a wrong jump target shows as a fault or a
// different checksum before any game object is involved.
//
// The function is entered with the second argument non-zero, which skips the
// one branch that calls out (a lookup of the emitter's texture animation and a
// rebuild of its first edge), and with flag 0x10 set in the emitter, which
// takes the time step from the argument instead of working it out from the
// emitter's own fields. Nothing in the fixed emitter is older than its maximum
// age (+110h), so the loop that retires old edges first ends at once.
bool RunClientOnFixed(uint32_t* checksum) {
    alignas(16) unsigned char obj[RibbonEdgeModel::kObjSpan];
    float ages[8];
    float seg[8 * 12];
    RibbonEdgeModel::BuildFixed(obj, ages, seg);
    obj[0x160] = 0x10;
    *reinterpret_cast<float*>(obj + 0x110) = 1.0f;

    unsigned short saved = 0, want = 0x027F;
    __asm {
        fnstcw saved
        fldcw  want
    }
    bool ok = true;
    __try {
        reinterpret_cast<ClientFunction>(kClientFunction)(obj, 0.0166667f, 1);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        ok = false;
    }
    unsigned char env[28];
    __asm {
        fnstenv env
        fldenv  env
        fldcw   saved
    }
    // Tag word: every register empty, which is how the client's function must
    // leave the stack.
    if (*reinterpret_cast<unsigned short*>(env + 8) != 0xFFFFu) ok = false;
    *checksum = RibbonEdgeModel::ImageChecksum(obj, ages, seg);
    return ok;
}

}  // namespace

bool Init() {
    if (!Config::g_settings.OptRibbonEdgeLoop) return true;

    if (!BytesMatch(kEntry, kEntryWant, kEntryLen)) {
        Log("[RibbonEdgeLoop] NOT patched: the bytes at 0x%08X are not the ring "
            "test this was read from.", (unsigned)kEntry);
        return false;
    }
    if (!BytesMatch(kRejoin, kRejoinWant, sizeof(kRejoinWant))) {
        Log("[RibbonEdgeLoop] NOT patched: the entry at 0x%08X matches but the "
            "code at 0x%08X does not.", (unsigned)kEntry, (unsigned)kRejoin);
        return false;
    }
    if (!LoopHashMatches()) {
        Log("[RibbonEdgeLoop] NOT patched: the loop between 0x%08X and 0x%08X is "
            "not the one the replacement was checked against.",
            (unsigned)kLoopTop, (unsigned)kRejoin);
        return false;
    }
    const uint32_t got = RibbonEdgeModel::FixedScenario(RunForSelfTest);
    if (got != kFixedChecksum) {
        Log("[RibbonEdgeLoop] NOT patched: the fixed emitter checksums to %08X "
            "here and %08X in the client's own sequence.", got, kFixedChecksum);
        return false;
    }
    if (!WowOpt_ClientPatchAllowed((const void*)kEntry)) {
        Log("[RibbonEdgeLoop] NOT patched: No Client Patches is on, and this writes "
            "twelve bytes into wow.exe.");
        return false;
    }

    uint32_t clientSum = 0;
    if (!RunClientOnFixed(&clientSum)) {
        Log("[RibbonEdgeLoop] NOT patched: the client's own function faulted or "
            "left the x87 stack unbalanced on the fixed emitter before anything "
            "was written, so it cannot be used as a reference.");
        return false;
    }

    memcpy(g_saved, (const void*)kEntry, kEntryLen);
    unsigned char patch[kEntryLen];
    patch[0] = 0xE9;
    *(int32_t*)(patch + 1) = (int32_t)((uintptr_t)&Thunk - (kEntry + 5));
    memset(patch + 5, 0x90, kEntryLen - 5);
    if (!WriteEntry(patch)) {
        Log("[RibbonEdgeLoop] NOT patched: could not make 0x%08X writable",
            (unsigned)kEntry);
        return false;
    }

    uint32_t patchedSum = 0;
    const bool ran = RunClientOnFixed(&patchedSum);
    const bool reached = g_calls == 1 && g_declined == 0 && g_badMode == 0;
    if (!ran || !reached || patchedSum != clientSum) {
        WriteEntry(g_saved);
        Log("[RibbonEdgeLoop] NOT patched: through the patch the client's function "
            "%s (checksum %08X against %08X unpatched, %lu call(s) reached the "
            "model, %lu declined, %lu for the FPU mode). The entry was put back.",
            ran ? "returned" : "faulted or left the x87 stack unbalanced",
            patchedSum, clientSum, g_calls, g_declined, g_badMode);
        g_calls = g_declined = g_badMode = g_edges = 0;
        return false;
    }
    g_calls = g_declined = g_badMode = g_edges = 0;

    g_patched = true;
    SamplingProfiler::RegisterSelfSymbol("RibbonEdgeLoop", (const void*)&Thunk);
    Log("[RibbonEdgeLoop] ACTIVE on the edge loop of the ribbon emitter update "
        "(sub_980090 @ 0x%08X). The client runs twelve x87 compare-and-branch "
        "groups an edge to keep the emitter's bounding box; this keeps it in two "
        "packed minimums and two packed maximums and does the arithmetic in SSE2 "
        "doubles at the client's rounding. 3.7 times faster an edge in the "
        "harness. Checked offline against the client's own instruction sequence "
        "over six million random emitters with no difference. At startup the "
        "client's own function was run on a fixed emitter once unpatched and "
        "once through the patch and both left checksum %08X. It does not compare "
        "with the client in the game, because the loop it replaces is not run; "
        "it declines when the time step or the FPU mode is outside what was "
        "checked.",
        (unsigned)kEntry, clientSum);
    return true;
}

void Shutdown() {
    if (!g_patched) return;
    WriteEntry(g_saved);
    g_patched = false;
}

void LogStats() {
    if (!Config::g_settings.OptRibbonEdgeLoop) return;
    if (!g_patched) {
        Log("[RibbonEdgeLoop] not patched - the reason is at the top of this log");
        return;
    }
    if (g_calls == 0) {
        Log("[RibbonEdgeLoop] patched, and no ribbon emitter has been updated yet. "
            "That is a measurement: no model with a ribbon has been in range since "
            "it went in.");
        return;
    }
    Log("[RibbonEdgeLoop] %lu emitter update(s) with edges reached the loop here: "
        "%lu answered (%lu edge(s)), %lu handed to the client's loop (%lu of those "
        "for the FPU mode, the rest for a time step or buffers outside what was "
        "checked). Plain counters, lower bounds.",
        g_calls, g_calls - g_declined - g_badMode, g_edges,
        g_declined + g_badMode, g_badMode);
}

}  // namespace RibbonEdgeLoop
