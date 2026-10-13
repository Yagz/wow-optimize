// ============================================================================
// Module: alloc_tag_census
//
// See alloc_tag_census.h for what this is for. How it works:
//
// The client's allocation wrapper (sub_76E540) and its free wrapper (sub_76E5A0)
// are already replaced by crt_free_hook.cpp, which calls OnAlloc and OnFree here
// while g_on is set. The realloc wrapper (sub_76E5E0) calls the CRT directly,
// so it would drift out of step; it is hooked here and moves the record from the
// old block to the new one.
//
// A table keyed by block address holds {address, size, tag}. Small blocks are
// recorded one in eight, chosen by a hash of the address so that the free of a
// block finds the record its allocation made; blocks of 1 KB or more are all
// recorded. The report scales the sampled bytes back by eight and says so.
//
// What this does not see, and the report says: blocks allocated before this
// hook went in (the first three seconds), blocks the client takes from the CRT
// without the wrapper (54 call sites for malloc against 1764 for the wrapper),
// and blocks it frees the same way, which leave a stale record until the
// address is allocated again.
//
// Cost, since this is a diagnostic and not a feature: one lock around a table
// operation per recorded block, and a periodic heap walk in the report that
// holds each heap's lock while it runs. Off by default.
// ============================================================================

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "MinHook.h"
#include "version.h"
#include "alloc_tag_census.h"
#include "config.h"
#include "win_mutex.h"
#include "high_tables.h"
#include "sampling_profiler.h"

extern "C" void Log(const char* fmt, ...);

namespace AllocTagCensus {

volatile long g_on = 0;

namespace {

// The client's realloc wrapper:
//   void* __stdcall sub_76E5E0(void* block, unsigned size, int tag, unsigned line, char flags)
// A null block goes through the allocation wrapper, which records it; a flags
// value of 0xB00BEEE5 makes it return null without touching anything.
constexpr uintptr_t kReallocWrapper = 0x0076E5E0;
typedef void* (__stdcall* RealloFn)(void* block, unsigned size, int tag, unsigned line, char flags);
RealloFn g_origRealloc = nullptr;

constexpr uint32_t kTableBits  = 22;
constexpr uint32_t kCap        = 1u << kTableBits;
constexpr uint32_t kMask       = kCap - 1;
constexpr uint32_t kTagCap     = 16384;
constexpr uint32_t kTagIdxCap  = kTagCap * 2;
constexpr unsigned kBigBytes   = 1024;     // recorded always
constexpr unsigned kSampleMask = 7;        // 1 in 8 below that
constexpr unsigned kSampleRate = kSampleMask + 1;

struct Entry { uint32_t ptr; uint32_t size; uint32_t tag; };

struct Tag {
    const char* file;
    uint32_t    line;
    uint32_t    countBig;
    uint32_t    countSmall;
    uint32_t    pad;
    uint64_t    bytesBig;
    uint64_t    bytesSmall;
};

Entry*    g_table  = nullptr;
Tag*      g_tags   = nullptr;
uint32_t* g_tagIdx = nullptr;
uint32_t  g_tagCount = 0;
bool      g_installed = false;
WinMutex  g_lock;

// Plain counters, lower bounds where they are written outside the lock.
unsigned long g_allocSeen   = 0;
unsigned long g_recorded    = 0;
unsigned long g_freeSeen    = 0;
unsigned long g_freeMatched = 0;
unsigned long g_overwrites  = 0;
unsigned long g_dropsFull   = 0;
unsigned long g_tagOverflow = 0;
unsigned long g_reallocs    = 0;
uint32_t      g_used        = 0;

inline uint32_t Slot(uint32_t p) { return ((p >> 3) * 2654435761u) >> (32 - kTableBits); }
inline bool Sampled(uint32_t p)  { return ((((p >> 5) * 0x85EBCA6Bu) >> 16) & kSampleMask) == 0; }

// Returns an index into g_tags, or the overflow bucket at kTagCap - 1 when the
// registry is full. Caller holds the lock.
uint32_t TagFor(const char* file, uint32_t line) {
    uint32_t h = (((uint32_t)(uintptr_t)file >> 2) * 2654435761u) ^ (line * 40503u);
    uint32_t i = (h >> 7) & (kTagIdxCap - 1);
    for (uint32_t n = 0; n < kTagIdxCap; ++n) {
        const uint32_t v = g_tagIdx[i];
        if (v == 0) {
            if (g_tagCount >= kTagCap - 1) { ++g_tagOverflow; return kTagCap - 1; }
            const uint32_t id = g_tagCount++;
            g_tags[id].file = file;
            g_tags[id].line = line;
            g_tagIdx[i] = id + 1;
            return id;
        }
        if (g_tags[v - 1].file == file && g_tags[v - 1].line == line) return v - 1;
        i = (i + 1) & (kTagIdxCap - 1);
    }
    return kTagCap - 1;
}

inline void Account(const Entry& e, int sign) {
    Tag& t = g_tags[e.tag];
    if (e.size >= kBigBytes) {
        t.bytesBig  += (uint64_t)sign * e.size;
        t.countBig  += (uint32_t)sign;
    } else {
        t.bytesSmall += (uint64_t)sign * e.size;
        t.countSmall += (uint32_t)sign;
    }
}

// Backward-shift deletion, so the table never accumulates tombstones under the
// churn of an allocator that frees as fast as it allocates.
void EraseAt(uint32_t i) {
    uint32_t j = i;
    for (;;) {
        j = (j + 1) & kMask;
        if (g_table[j].ptr == 0) break;
        const uint32_t k = Slot(g_table[j].ptr);
        const bool inGap = (i <= j) ? (i < k && k <= j) : (i < k || k <= j);
        if (inGap) continue;
        g_table[i] = g_table[j];
        i = j;
    }
    g_table[i].ptr = 0;
    --g_used;
}

void RecordLocked(uint32_t p, unsigned size, const char* file, unsigned line) {
    if (g_used * 10u > kCap * 7u) { ++g_dropsFull; return; }
    const uint32_t tag = TagFor(file, line);
    uint32_t i = Slot(p);
    while (g_table[i].ptr != 0) {
        if (g_table[i].ptr == p) {           // freed by a path this does not see, then reused
            Account(g_table[i], -1);
            ++g_overwrites;
            g_table[i].size = size;
            g_table[i].tag = tag;
            Account(g_table[i], +1);
            return;
        }
        i = (i + 1) & kMask;
    }
    g_table[i].ptr = p;
    g_table[i].size = size;
    g_table[i].tag = tag;
    ++g_used;
    ++g_recorded;
    Account(g_table[i], +1);
}

bool ForgetLocked(uint32_t p) {
    uint32_t i = Slot(p);
    while (g_table[i].ptr != 0) {
        if (g_table[i].ptr == p) {
            Account(g_table[i], -1);
            EraseAt(i);
            return true;
        }
        i = (i + 1) & kMask;
    }
    return false;
}

void* __stdcall Hooked_Realloc(void* block, unsigned size, int tag, unsigned line, char flags) {
    void* r = g_origRealloc(block, size, tag, line, flags);
    if (g_on && block) {
        const uint32_t oldp = (uint32_t)(uintptr_t)block;
        if (r) {
            WinLockGuard g(g_lock);
            ++g_reallocs;
            ForgetLocked(oldp);
            if (size) {
                const unsigned rounded = (size + 7u) & 0xFFFFFFF8u;
                const uint32_t np = (uint32_t)(uintptr_t)r;
                if (rounded >= kBigBytes || Sampled(np)) RecordLocked(np, rounded, (const char*)(uintptr_t)tag, line);
            }
        } else if (size == 0) {
            WinLockGuard g(g_lock);
            ForgetLocked(oldp);
        }
    }
    return r;
}

// A tag string is normally ".\src\Name.cpp" or an RTTI name; read it without
// trusting it, because a few call sites pass something else.
void TagText(const char* s, char* out, size_t cap) {
    out[0] = 0;
    if (!s) { _snprintf_s(out, cap, _TRUNCATE, "(none)"); return; }
    bool ok = true;
    size_t n = 0;
    __try {
        for (; n + 1 < cap && s[n]; ++n) {
            const unsigned char c = (unsigned char)s[n];
            if (c < 0x20 || c > 0x7E) { ok = false; break; }
            out[n] = (char)c;
        }
        out[n] = 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        ok = false;
    }
    if (!ok || n == 0) {
        _snprintf_s(out, cap, _TRUNCATE, "(tag at 0x%08X)", (unsigned)(uintptr_t)s);
        return;
    }
    const char* start = out;
    if (strncmp(start, ".\\src\\", 6) == 0) start += 6;
    else if (strncmp(start, ".\\", 2) == 0) start += 2;
    if (start != out) memmove(out, start, strlen(start) + 1);
}

struct HeapSummaryStruct { DWORD cb; SIZE_T cbAllocated; SIZE_T cbCommitted; SIZE_T cbReserved; SIZE_T cbMaxReserve; };
typedef BOOL (WINAPI* HeapSummary_fn)(HANDLE, DWORD, HeapSummaryStruct*);

void ReportHeaps() {
    HeapSummary_fn hs = (HeapSummary_fn)GetProcAddress(GetModuleHandleA("kernel32.dll"), "HeapSummary");
    if (!hs) {
        Log("[AllocTagCensus] heaps: not measured, kernel32 has no HeapSummary here.");
        return;
    }
    HANDLE heaps[64];
    const DWORD n = GetProcessHeaps(64, heaps);
    HANDLE crt = nullptr;
    __try { crt = *(HANDLE*)0x00B31684; } __except (EXCEPTION_EXECUTE_HANDLER) { crt = nullptr; }
    HANDLE proc = GetProcessHeap();

    LARGE_INTEGER f, a, b;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&a);
    double totA = 0, totC = 0, totR = 0;
    Log("[AllocTagCensus] heaps (allocated is what is handed out, committed what is backed by memory; the "
        "difference is slack and fragmentation):");
    for (DWORD i = 0; i < n && i < 64; ++i) {
        HeapSummaryStruct s = {};
        s.cb = sizeof(s);
        if (!hs(heaps[i], 0, &s)) continue;
        const char* who = (heaps[i] == crt) ? "the client's CRT heap" : (heaps[i] == proc) ? "process heap" : "other heap";
        Log("[AllocTagCensus]   %-22s 0x%08X  allocated %8.1f MB  committed %8.1f MB  reserved %8.1f MB",
            who, (unsigned)(uintptr_t)heaps[i], s.cbAllocated / 1048576.0, s.cbCommitted / 1048576.0,
            s.cbReserved / 1048576.0);
        totA += s.cbAllocated; totC += s.cbCommitted; totR += s.cbReserved;
    }
    QueryPerformanceCounter(&b);
    Log("[AllocTagCensus]   all %lu heaps: allocated %.1f MB, committed %.1f MB, reserved %.1f MB (the walk took %.0f ms)",
        (unsigned long)n, totA / 1048576.0, totC / 1048576.0, totR / 1048576.0,
        (double)(b.QuadPart - a.QuadPart) * 1000.0 / (double)f.QuadPart);
}

}  // namespace

void OnAlloc(void* vp, unsigned size, const char* tag, unsigned line) {
    ++g_allocSeen;
    const uint32_t p = (uint32_t)(uintptr_t)vp;
    if (size < kBigBytes && !Sampled(p)) return;
    WinLockGuard g(g_lock);
    RecordLocked(p, size, tag, line);
}

void OnFree(void* vp) {
    ++g_freeSeen;
    WinLockGuard g(g_lock);
    if (ForgetLocked((uint32_t)(uintptr_t)vp)) ++g_freeMatched;
}

void Init() {
    if (!Config::g_settings.OptAllocTagCensus) return;
    if (g_installed) return;

    static const unsigned char kOpening[14] = { 0x55, 0x8B, 0xEC, 0x51, 0x53, 0x8B, 0x5D, 0x18, 0x81, 0xFB, 0xE5, 0xEE, 0x0B, 0xB0 };
    if (!WowOpt_ClientBytesAre(kReallocWrapper, kOpening, sizeof(kOpening))) {
        char hex[64];
        WowOpt_HexBytes(kReallocWrapper, hex, sizeof(hex));
        Log("[AllocTagCensus] NOT active: the bytes at 0x%08X are not the client's realloc wrapper: %s",
            (unsigned)kReallocWrapper, hex);
        return;
    }

    const size_t tableBytes = (size_t)kCap * sizeof(Entry);
    const size_t tagBytes   = (size_t)kTagCap * sizeof(Tag);
    const size_t idxBytes   = (size_t)kTagIdxCap * sizeof(uint32_t);
    uint8_t* block = (uint8_t*)HighTables::Reserve("alloc_tag_census", tableBytes + tagBytes + idxBytes);
    if (!block) {
        Log("[AllocTagCensus] NOT active: %.0f MB for the table could not be reserved.",
            (tableBytes + tagBytes + idxBytes) / 1048576.0);
        return;
    }
    g_table  = (Entry*)block;
    g_tags   = (Tag*)(block + tableBytes);
    g_tagIdx = (uint32_t*)(block + tableBytes + tagBytes);
    g_tagCount = 0;
    g_used = 0;

    if (WineSafe_CreateHook((void*)kReallocWrapper, (void*)Hooked_Realloc, (void**)&g_origRealloc) != MH_OK ||
        MH_EnableHook((void*)kReallocWrapper) != MH_OK) {
        Log("[AllocTagCensus] NOT active: the realloc wrapper at 0x%08X could not be hooked.", (unsigned)kReallocWrapper);
        return;
    }
    SamplingProfiler::RegisterSelfSymbol("alloc_tag_realloc", (const void*)&Hooked_Realloc);
    g_installed = true;
    InterlockedExchange(&g_on, 1);
    Log("[AllocTagCensus] ACTIVE: every block the client allocates through its wrapper is paired with the source "
        "file and line it passed, one in %u below %u bytes and all above. Uses %.0f MB for the table. Costs a lock "
        "per recorded block, and a pause in each report while the heaps are summarised. Measurement only.",
        kSampleRate, kBigBytes, (tableBytes + tagBytes + idxBytes) / 1048576.0);
}

static void ReportImpl(bool terminating) {
    if (!Config::g_settings.OptAllocTagCensus) return;
    if (!g_installed) {
        if (!terminating) Log("[AllocTagCensus] not measured: not installed (the reason is logged at start).");
        return;
    }

    // Copy what the report needs out from under the lock, then print.
    struct Row { const char* file; uint32_t line; double mb; unsigned long count; };
    static Row top[30];
    static struct FileRow { const char* file; double mb; unsigned long count; } files[24];
    int nTop = 0, nFiles = 0;
    double trackedMB = 0.0;
    unsigned long trackedCount = 0;
    unsigned long used, recorded, overwrites, dropsFull, tagCount, overflow, seen, freed, matched, reallocs;

    // At exit the thread that crashed may be the one holding this lock, so it is tried and not waited for.
    if (terminating) {
        if (!g_lock.try_lock()) {
            Log("[AllocTagCensus] not reported at exit: the table was being written when the process went down.");
            return;
        }
    } else {
        g_lock.lock();
    }
    {
        used = g_used; recorded = g_recorded; overwrites = g_overwrites; dropsFull = g_dropsFull;
        tagCount = g_tagCount; overflow = g_tagOverflow; seen = g_allocSeen; freed = g_freeSeen;
        matched = g_freeMatched; reallocs = g_reallocs;
        memset(top, 0, sizeof(top));
        memset(files, 0, sizeof(files));
        // Per file the tags are folded together by file pointer; there are at most a
        // few hundred distinct files, so a linear search is fine for a report.
        static const char* seenFile[512];
        static double      seenMB[512];
        static unsigned long seenCount[512];
        int nSeen = 0;
        for (uint32_t i = 0; i < g_tagCount; ++i) {
            const Tag& t = g_tags[i];
            const double mb = ((double)t.bytesBig + (double)t.bytesSmall * kSampleRate) / 1048576.0;
            const unsigned long cnt = t.countBig + t.countSmall * kSampleRate;
            if (mb <= 0.0 && cnt == 0) continue;
            trackedMB += mb;
            trackedCount += cnt;
            // top by tag
            int pos = nTop;
            while (pos > 0 && top[pos - 1].mb < mb) --pos;
            if (pos < 30) {
                const int last = (nTop < 30) ? nTop : 29;
                for (int k = last; k > pos; --k) top[k] = top[k - 1];
                top[pos].file = t.file; top[pos].line = t.line; top[pos].mb = mb; top[pos].count = cnt;
                if (nTop < 30) ++nTop;
            }
            // by file
            int s = 0;
            for (; s < nSeen; ++s) if (seenFile[s] == t.file) break;
            if (s == nSeen && nSeen < 512) { seenFile[nSeen] = t.file; seenMB[nSeen] = 0; seenCount[nSeen] = 0; ++nSeen; }
            if (s < nSeen) { seenMB[s] += mb; seenCount[s] += cnt; }
        }
        for (int s = 0; s < nSeen; ++s) {
            int pos = nFiles;
            while (pos > 0 && files[pos - 1].mb < seenMB[s]) --pos;
            if (pos < 24) {
                const int last = (nFiles < 24) ? nFiles : 23;
                for (int k = last; k > pos; --k) files[k] = files[k - 1];
                files[pos].file = seenFile[s]; files[pos].mb = seenMB[s]; files[pos].count = seenCount[s];
                if (nFiles < 24) ++nFiles;
            }
        }
    }
    g_lock.unlock();

    PROCESS_MEMORY_COUNTERS_EX pmc = {};
    pmc.cb = sizeof(pmc);
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));

    Log("[AllocTagCensus] === live blocks by who asked (scaled: small blocks counted one in %u) ===", kSampleRate);
    Log("[AllocTagCensus] tracked live: %.1f MB in %lu blocks, against %.1f MB private for the whole process. "
        "The difference is the CRT's direct callers, other modules, the driver and DXVK, memory allocated in the "
        "first seconds before this was hooked, and stale records.",
        trackedMB, trackedCount, pmc.PrivateUsage / 1048576.0);
    Log("[AllocTagCensus] %lu allocations seen, %lu recorded; %lu frees seen, %lu matched a record (%lu realloc, "
        "%lu overwritten by a reuse, %lu dropped on a full table, %lu tag(s) over the registry). Lower bounds.",
        seen, recorded, freed, matched, reallocs, overwrites, dropsFull, overflow);
    Log("[AllocTagCensus] by file (%lu tag(s) seen):", tagCount);
    char name[64];
    for (int i = 0; i < nFiles; ++i) {
        TagText(files[i].file, name, sizeof(name));
        Log("[AllocTagCensus]   %9.1f MB %9lu blocks  %s", files[i].mb, files[i].count, name);
    }
    Log("[AllocTagCensus] by file and line:");
    for (int i = 0; i < nTop; ++i) {
        TagText(top[i].file, name, sizeof(name));
        const double avg = top[i].count ? top[i].mb * 1048576.0 / (double)top[i].count : 0.0;
        Log("[AllocTagCensus]   %9.1f MB %9lu blocks %9.0f B avg  %s:%d", top[i].mb, top[i].count, avg, name,
            (int)top[i].line);
    }
    // Summarising a heap takes its lock. At exit that lock may be held by the very thread that is
    // dying inside the heap, and waiting for it would turn a crash report into a hang.
    if (!terminating) ReportHeaps();
    (void)used;
}

void LogStats() { ReportImpl(false); }
void LogAtTerminate() { ReportImpl(true); }

}  // namespace AllocTagCensus
