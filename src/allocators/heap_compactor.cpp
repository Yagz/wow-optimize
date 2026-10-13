#include "version.h"

#if TEST_DISABLE_HEAP_COMPACTOR == 0

#include <windows.h>
#include <psapi.h>
#include <cstdint>
#include <atomic>

#pragma comment(lib, "psapi.lib")

// Configuration
static constexpr DWORD MONITOR_INTERVAL_MS = 10000;     // Check every 10s (was 3 - reduce CPU overhead)
static constexpr DWORD LOADING_INTERVAL_MS = 3000;      // Faster checks during loading screens
// The one threshold, and it is read against the largest free run below 2GB.
// A WARNING_THRESHOLD of 32MB used to sit beside it, checked against the whole
// address space; it is gone with the branch that read it. An unused constant
// with that name is an invitation to wire it back to the wrong figure.
static constexpr SIZE_T CRITICAL_THRESHOLD = 16 * 1024 * 1024;  // 16MB

// Statistics
static std::atomic<uint64_t> g_checksPerformed{0};
static std::atomic<uint64_t> g_compactionsTriggered{0};
static std::atomic<SIZE_T>   g_lastLargestBlock{0};
static std::atomic<SIZE_T>   g_minLargestBlock{SIZE_MAX};
static std::atomic<SIZE_T>   g_maxLargestBlock{0};

// Monitor thread handle
static HANDLE g_monitorThread = nullptr;
static volatile bool g_shutdown = false;

// The monitor thread only ever *requests* compaction; the actual HeapCompact()/
// mi_collect() work runs on the main thread via HeapCompactor_RunPendingWork(),
// called from the existing per-frame maintenance tick. HeapCompact() walks and
// mutates every process heap, and mi_collect(true) can decommit/unmap memory —
// doing that from an unsynchronized background thread raced against whatever
// heap WoW's networking code (or Winsock internally) was using mid-connect,
// which could yank a buffer out from under an in-flight WSA operation and
// surface as "unable to connect" (GitHub issue #39).
static std::atomic<int> g_pendingWork{0}; // 0=none, 1=proactive mi_collect, 2=full ForceHeapCompaction

// Brake and self-assessment for the low-half trigger. A compaction that does
// not give address space back is pure stall, and doing it every ten seconds for
// eight hours would be worse than the problem.
static DWORD g_lastCompactTick     = 0;
static DWORD g_compactIntervalMs   = 60000;   // grows when a pass achieves little
static bool  g_compactionGaveUp    = false;
static int   g_uselessInARow       = 0;
static unsigned long long g_reclaimedTotalKb = 0;

// Forward declarations
extern "C" void Log(const char* fmt, ...);
extern "C" void mi_collect(bool force);
#include <mimalloc.h>
#include "crash_dumper.h"
#include "perf_diagnostics.h"
#include "mimalloc_high_arena.h"
#include "high_placement.h"

// Largest free virtual address range, measured twice: over everything, and over
// the low 2 GB into `lowHalfOut` when it is wanted. On a large-address-aware
// client the full-range answer is dominated by the barely-touched half above
// 2 GB and stays in the gigabytes while the low half is down to a megabyte, so
// the two are different questions and a caller has to say which it is asking.
//
// The walk is timed and the region count reported with it, because a
// VirtualQuery over a fragmented 3 GB address space is tens of thousands of
// calls and is the prime suspect for a stall of our own making.
// The monitor thread's most recent walk, for the report to read instead of
// walking again on the main thread. Written by one thread and read by another;
// they are plain SIZE_T on 32-bit x86, so a read cannot tear, and a report that
// caught a value mid-update would be one monitor interval out of date rather than
// wrong.
static volatile SIZE_T g_lastWalkFree = 0;
static volatile SIZE_T g_lastWalkLow  = 0;
static volatile SIZE_T g_lastWalkLowTotal = 0;
static volatile DWORD  g_lastWalkTick = 0;

static double   g_walkMsWorst = 0.0;
static double   g_walkMsTotal = 0.0;

// Which thread paid for each walk, counted separately. Pooled, the report reads
// as main-thread work: 811 runs and 7.7 seconds looks like a stall inside a
// frame until the arithmetic says 811 walks over 7854 seconds is one every 9.7,
// which is the monitor interval.
static DWORD    g_monitorTid    = 0;
static unsigned g_walkCountMain = 0;
static double   g_walkMsMain    = 0.0;
static double   g_walkMsWorstMain = 0.0;
static unsigned g_walkCount   = 0;
static unsigned g_walkRegionsWorst = 0;

// Returns the largest free run over all of user address space. Optionally also
// the largest free run below 2GB and the sum of every free region there - the
// two figures the periodic report used to get by walking the address space a
// second time, on the main thread, inside a frame.
static SIZE_T GetLargestFreeBlock(SIZE_T* lowHalfOut = nullptr,
                                  SIZE_T* lowTotalOut = nullptr) {
    static const uintptr_t LOW_HALF_END = 0x80000000u;
    static LARGE_INTEGER freq = {};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER wa;
    QueryPerformanceCounter(&wa);
    unsigned regions = 0;

    MEMORY_BASIC_INFORMATION mbi;
    SIZE_T largestFree = 0;
    SIZE_T currentFree = 0;
    SIZE_T largestLow  = 0;
    SIZE_T currentLow  = 0;
    SIZE_T lowTotal    = 0;
    uintptr_t addr = 0;

    while (VirtualQuery((LPCVOID)addr, &mbi, sizeof(mbi))) {
        uintptr_t base = (uintptr_t)mbi.BaseAddress;

        if (mbi.State == MEM_FREE) {
            currentFree += mbi.RegionSize;

            // Only the part of this region that lies below the boundary counts
            // towards the low-half figure, and a run that crosses it stops
            // there rather than carrying the high side back down.
            if (base < LOW_HALF_END) {
                SIZE_T lowPart = (base + mbi.RegionSize > LOW_HALF_END)
                               ? (SIZE_T)(LOW_HALF_END - base)
                               : mbi.RegionSize;
                currentLow += lowPart;
                lowTotal   += lowPart;
                if (lowPart != mbi.RegionSize) {
                    if (currentLow > largestLow) largestLow = currentLow;
                    currentLow = 0;
                }
            }
        } else {
            if (currentFree > largestFree) largestFree = currentFree;
            currentFree = 0;
            if (base < LOW_HALF_END) {
                if (currentLow > largestLow) largestLow = currentLow;
                currentLow = 0;
            }
        }

        ++regions;
        addr = base + mbi.RegionSize;
        if (addr < base) break; // Overflow
    }

    if (currentFree > largestFree) largestFree = currentFree;
    if (currentLow  > largestLow)  largestLow  = currentLow;

    if (freq.QuadPart) {
        LARGE_INTEGER wb;
        QueryPerformanceCounter(&wb);
        double ms = (double)(wb.QuadPart - wa.QuadPart) * 1000.0 / (double)freq.QuadPart;
        g_walkMsTotal += ms;
        ++g_walkCount;
        if (ms > g_walkMsWorst) { g_walkMsWorst = ms; g_walkRegionsWorst = regions; }
        if (GetCurrentThreadId() != g_monitorTid) {
            ++g_walkCountMain;
            g_walkMsMain += ms;
            if (ms > g_walkMsWorstMain) g_walkMsWorstMain = ms;
        }
    }

    if (lowHalfOut)  *lowHalfOut  = largestLow;
    if (lowTotalOut) *lowTotalOut = lowTotal;
    return largestFree;
}

// Force heap compaction via Windows heap APIs
static void ForceHeapCompaction() {
    // Get all process heaps
    HANDLE heaps[64];
    DWORD heapCount = GetProcessHeaps(64, heaps);
    
    for (DWORD i = 0; i < heapCount; i++) {
        // HeapCompact is available on Vista+
        // This consolidates free blocks within the heap
        HeapCompact(heaps[i], 0);
    }
    
    // Also trigger mimalloc collection
    // (mimalloc is our global allocator replacement)
    {
        StallProbe probe("ForceHeapCompaction mi_collect", 4.0);
        mi_collect(true);
    }
}

// Forward declaration for loading mode check
namespace LuaOpt { bool IsLoadingMode(); }

// Monitor thread - checks VA state periodically
static DWORD WINAPI MonitorThread(LPVOID) {
    Log("[HeapCompactor] Monitor thread started (interval=%dms, loading=%dms, critical=%dMB)",
        MONITOR_INTERVAL_MS, LOADING_INTERVAL_MS, CRITICAL_THRESHOLD / (1024*1024));
    
    while (!g_shutdown) {
        // Use faster interval during loading screens, slower during gameplay
        DWORD interval = LuaOpt::IsLoadingMode() ? LOADING_INTERVAL_MS : MONITOR_INTERVAL_MS;
        Sleep(interval);
        
        SIZE_T largestLow = 0, lowTotal = 0;
        SIZE_T largestFree = GetLargestFreeBlock(&largestLow, &lowTotal);
        g_checksPerformed++;
        HighPlacement::NotifyLowHalfFree(largestLow, lowTotal);

        // Published for the report, which used to walk the address space again on
        // the main thread to get the same two numbers. This thread has just paid
        // for that walk off the main thread; there is no reason to pay again
        // inside a frame.
        g_lastWalkFree = largestFree;
        g_lastWalkLow  = largestLow;
        g_lastWalkLowTotal = lowTotal;
        g_lastWalkTick = GetTickCount();

        // Top the high arena up before the allocator has to go to the OS for
        // more, which it does bottom-up and into the half the client needs.
        // Here rather than on the main thread because it reserves address
        // space, and this thread already exists to do that kind of work.
        MimallocHighArena::Grow();

        // Keep the census's module list current for DLLs loaded since the last
        // pass - DXVK's d3d9.dll arrives after install. It takes the loader lock,
        // which the allocation hook itself must never do.
        HighPlacement::RefreshModules();

        // The low half is what actually runs out, and it is what the trigger
        // reads now.
        //
        // Triggering on the full-range figure instead misses it entirely: an
        // eight-hour session sat with the low half at 10-14 MB, under the 16 MB
        // threshold the whole time, while the full range read 1857 MB and no
        // compaction ran.
        //
        // Acting on it needs a brake: a client parked below the threshold would
        // otherwise compact on every tick, and a full mi_collect plus a
        // HeapCompact of every process
        // heap is exactly the kind of stall this project keeps chasing. So the
        // request is rate limited, and the module measures whether its own work
        // achieves anything - see the recovery check in RunPendingWork.
        if (largestLow < CRITICAL_THRESHOLD) {
            DWORD nowTick2 = GetTickCount();

            // Name what is holding the low half, once when it first runs out
            // and then rarely. Every log that has ever shown this state has
            // shown the number and not the owner, so the same question gets
            // asked of every tester and answered by none of them. This is a
            // full VirtualQuery walk, which is why it is here on the monitor
            // thread and rate limited rather than in the periodic report.
            static DWORD lastOccupancyTick = 0;
            if (lastOccupancyTick == 0 ||
                (nowTick2 - lastOccupancyTick) >= 300000) {
                lastOccupancyTick = nowTick2;
                PerfDiagnostics::LogLowHalfOccupancy(
                    "largest free block below 2GB went under 16MB");
            }
            bool due = (g_lastCompactTick == 0) ||
                       (nowTick2 - g_lastCompactTick) >= g_compactIntervalMs;
            if (due && !g_compactionGaveUp) {
                g_pendingWork.store(2, std::memory_order_release);
                g_lastCompactTick = nowTick2;
            }
            static DWORD lastSplitTick = 0;
            if (lastSplitTick == 0 || nowTick2 - lastSplitTick > 60000) {
                Log("[HeapCompactor] %uMB free below 2GB (largest block), %uMB "
                    "across all user address space. The low half is what the "
                    "client allocates from, so that is what this acts on.%s",
                    (unsigned)(largestLow / (1024*1024)),
                    (unsigned)(largestFree / (1024*1024)),
                    g_compactionGaveUp ? " Compaction has stopped: it was not "
                                         "recovering anything." : "");
                lastSplitTick = nowTick2;
            }
        }
        
        // Update statistics
        g_lastLargestBlock = largestFree;
        
        SIZE_T minVal = g_minLargestBlock.load();
        while (largestFree < minVal && 
               !g_minLargestBlock.compare_exchange_weak(minVal, largestFree));
        
        SIZE_T maxVal = g_maxLargestBlock.load();
        while (largestFree > maxVal && 
               !g_maxLargestBlock.compare_exchange_weak(maxVal, largestFree));
        
        // The thresholds that used to be checked here are gone.
        //
        // They tested largestFree - the largest run anywhere in user address
        // space - against 16MB and 32MB, and requested the same compaction the
        // low-half check above requests. largestLow can never exceed largestFree,
        // so largestFree < 16MB implies largestLow < 16MB and the block above has
        // already fired: strictly redundant, and removing them changes nothing
        // for any input.
        //
        // What they were not was harmless. A second threshold on the wrong figure
        // is what let two other modules sit calm through a session with one
        // megabyte left below 2GB, and their log lines said "LargestFreeBlock"
        // with no range on it while the line above named its own.
    }

    Log("[HeapCompactor] Monitor thread shutting down");
    return 0;
}

// Called once per frame from the main thread's existing periodic maintenance
// tick. Performs whatever compaction the monitor thread requested — keeps all
// heap-mutating work off the background thread.
extern "C" void HeapCompactor_RunPendingWork() {
    int work = g_pendingWork.exchange(0, std::memory_order_acquire);
    if (work == 0) return;

    if (work == 2) {
        // The "before" figure comes from the monitor thread's last walk, not
        // from a fresh one on this thread.
        //
        // This walk is VirtualQuery over all of user address space. A field
        // session logs 40 of them at 598.5 ms in total, and seven of those -
        // 92.3 ms, worst 15.8 ms - ran on a thread other than the monitor,
        // which means inside a frame. Six of the seven are this function: two
        // walks per compaction, before and after, and that session's three
        // compactions recovered 0 KB each.
        //
        // The before walk is the one that can go. The monitor is what decided
        // this pass was needed, and it decided on the strength of its own walk
        // moments earlier, so its published figure is the same measurement this
        // one would repeat. Only when the monitor has never run is a walk done
        // here, and then it is the first and only one.
        SIZE_T beforeLow = g_lastWalkTick ? g_lastWalkLow : 0;
        if (!g_lastWalkTick) GetLargestFreeBlock(&beforeLow);
        const unsigned long beforeAgeMs =
            g_lastWalkTick ? (unsigned long)(GetTickCount() - g_lastWalkTick) : 0;

        // mimalloc runs with purge_decommits off, so a purge resets pages but
        // leaves them committed - physical RAM comes back, address space does not.
        // That is the right trade in normal play (the comment at the option's
        // setting explains the driver-fault reasoning), and exactly the wrong one
        // here: this branch only runs because the process has no address space
        // left, which is the one resource resetting cannot return.
        //
        // A tester's session climbed to 3533 MB of mimalloc commit with 223 MB of
        // VA free and a 1 MB largest block, and every compaction pass achieved
        // nothing. Decommit for the duration of this pass only, then restore.
        {
            StallProbe probe("critical mi_collect (decommit)", 4.0);
            mi_option_set(mi_option_purge_decommits, 1);
            mi_collect(true);
            mi_option_set(mi_option_purge_decommits, 0);
        }

        ForceHeapCompaction();
        g_compactionsTriggered++;

        // Judge the pass by the half it was run for. The full-range figure is
        // what made this module blind in the first place, so measuring recovery
        // with it would repeat the mistake one level down.
        SIZE_T afterLow = 0;
        SIZE_T after = GetLargestFreeBlock(&afterLow);
        long long gainedKb = ((long long)afterLow - (long long)beforeLow) / 1024;
        if (gainedKb > 0) g_reclaimedTotalKb += (unsigned long long)gainedKb;

        Log("[HeapCompactor] After compaction: %uMB largest below 2GB (%+lldKB "
            "against a before figure %lu ms old), %uMB across all address space",
            (unsigned)(afterLow / (1024*1024)), gainedKb, beforeAgeMs,
            (unsigned)(after / (1024*1024)));

        // A pass that returns almost nothing is pure stall. Back off, and stop
        // entirely if it keeps happening: on a client whose low half is
        // genuinely full rather than merely fragmented there is nothing to
        // recover, and grinding every heap in the process to find that out again
        // costs a frame each time.
        //
        // A session that has not got 256 KB back in total from six passes - three
        // at the base interval and three at twice it - is not going to. A field
        // session of 2026-10-13 ran fifteen passes at 60 s doubling to 960 s,
        // every one "0KB recovered", and the main thread paid 10 to 35 ms for
        // each; the maintenance stall report totals 557 ms for the session.
        if (gainedKb < 256) {
            if (++g_uselessInARow >= 3) {
                const bool neverPaid = g_reclaimedTotalKb < 256;
                if (g_compactIntervalMs < 600000 &&
                    !(neverPaid && g_compactIntervalMs >= 120000)) {
                    g_compactIntervalMs *= 2;
                    Log("[HeapCompactor] Three passes in a row recovered almost "
                        "nothing; backing off to one attempt per %u seconds.",
                        g_compactIntervalMs / 1000);
                } else {
                    g_compactionGaveUp = true;
                    Log("[HeapCompactor] Compaction is not recovering address "
                        "space on this client, so it stops. The low half is full "
                        "rather than fragmented, and grinding every heap to "
                        "rediscover that costs a frame each time.");
                }
                g_uselessInARow = 0;
            }
        } else {
            g_uselessInARow = 0;
        }
    } else {
        {
            StallProbe probe("compaction mi_collect", 4.0);
            mi_collect(true);
        }
        g_compactionsTriggered++;
    }
}

bool HeapCompactor_Init() {
    if (g_monitorThread) {
        Log("[HeapCompactor] Already initialized");
        return true;
    }
    
    g_shutdown = false;
    DWORD monTid = 0;
    g_monitorThread = CreateThread(NULL, 0, MonitorThread, NULL, 0, &monTid);
    g_monitorTid = monTid;
    
    if (!g_monitorThread) {
        Log("[HeapCompactor] Failed to create monitor thread");
        return false;
    }
    
    // Log initial state
    SIZE_T initialFree = GetLargestFreeBlock();
    Log("[HeapCompactor] ACTIVE (initial LargestFreeBlock=%uMB)",
        (unsigned)(initialFree / (1024*1024)));
    
    return true;
}

void HeapCompactor_Shutdown() {
    if (!g_monitorThread) return;
    
    g_shutdown = true;
    WaitForSingleObject(g_monitorThread, 3000);
    CloseHandle(g_monitorThread);
    g_monitorThread = nullptr;
    
    Log("[HeapCompactor] Shutdown complete (checks=%llu, compactions=%llu, min=%uMB, max=%uMB)",
        (unsigned long long)g_checksPerformed.load(),
        (unsigned long long)g_compactionsTriggered.load(),
        (unsigned)(g_minLargestBlock.load() / (1024*1024)),
        (unsigned)(g_maxLargestBlock.load() / (1024*1024)));
}

// Query current state (for diagnostics)
// Printed from the periodic report, not from Shutdown: this DLL exits through
// TerminateProcess and a teardown-only counter never reaches a log.
extern "C" void HeapCompactor_LogStats() {
    // Read the monitor thread's last walk rather than walking again here.
    //
    // This function runs on the main thread from the periodic report, and the
    // walk it used to do is VirtualQuery over the whole address space, one call
    // per region. Tester logs carry "periodic maintenance took 42.6 ms" against a
    // frame median near ten, and the session that recorded it had the low half
    // fragmented to a 19 MB largest block - tens of thousands of regions.
    //
    // The monitor thread walks every ten seconds anyway and now publishes what it
    // found. A report reading that is at most one interval out of date, which for
    // a figure printed every five minutes is nothing, and the age is printed so
    // nobody has to assume it.
    //
    // Before the monitor has run once there is nothing to read, and the walk
    // happens here - once, at the first report.
    SIZE_T low = 0, all = 0;
    DWORD  ageMs = 0;
    if (g_lastWalkTick) {
        all   = g_lastWalkFree;
        low   = g_lastWalkLow;
        ageMs = GetTickCount() - g_lastWalkTick;
    } else {
        all = GetLargestFreeBlock(&low);
    }

    // Printed before the numbers it produced, and split by which thread paid.
    // Only the second figure is a stall a player can feel; the first is off the
    // frame entirely and is here so the two are not confused, which is exactly
    // what happened while they shared one line.
    if (g_walkCount) {
        Log("[HeapCompactor] the address-space walk that produces the figures "
            "below: %u run(s), %.1f ms in total, worst %.1f ms over %u regions. "
            "%u of those (%.1f ms, worst %.1f ms) were on a thread other than "
            "the monitor - those are the ones that land inside a frame. The rest "
            "are the monitor thread and cost the player nothing.",
            g_walkCount, g_walkMsTotal, g_walkMsWorst, g_walkRegionsWorst,
            g_walkCountMain, g_walkMsMain, g_walkMsWorstMain);
    }
    Log("[HeapCompactor] the figures below are %lu ms old - taken by the monitor "
        "thread, so this report does not walk again for them.",
        (unsigned long)ageMs);
    Log("[HeapCompactor] %uMB largest free below 2GB, %uMB across all address "
        "space; %llu compactions, %lluKB recovered, next attempt no sooner than "
        "%us%s",
        (unsigned)(low / (1024*1024)), (unsigned)(all / (1024*1024)),
        (unsigned long long)g_compactionsTriggered.load(), g_reclaimedTotalKb,
        g_compactIntervalMs / 1000,
        g_compactionGaveUp ? " - stopped, it was not recovering anything" : "");
}

// Every query below hands back the monitor thread's last walk.
//
// Two that walked on the caller's thread used to stand here, exported from the
// header with no thread rule attached and called by nothing. The walk is
// VirtualQuery over all of user address space and the field data has 6183 free
// regions in the low half alone; a main-thread copy of it, run once a second,
// is the one stall this project has measured against itself. With them gone
// there is no way to reach the walk except from the monitor thread that owns it.

// The low-half figure without the walk, for callers on the main thread.
//
// Returns the monitor thread's last result and how old it is, or 0 with an age of
// 0 when the monitor has not run yet - which is a different fact from "no memory
// left" and the caller has to be able to tell them apart. The walk is
// VirtualQuery over the whole address space and costs tens of milliseconds on a
// fragmented one; that is not a price a diagnostic should charge inside a frame.
extern "C" SIZE_T HeapCompactor_GetLastLowHalf(unsigned long* ageMsOut) {
    if (!g_lastWalkTick) { if (ageMsOut) *ageMsOut = 0; return 0; }
    if (ageMsOut) *ageMsOut = (unsigned long)(GetTickCount() - g_lastWalkTick);
    return g_lastWalkLow;
}


// Both low-half figures at once, for the periodic report.
//
// Returns false when the monitor has not walked yet, so the caller can say "not
// measured" rather than print two zeros. Kept separate from the accessor above
// because that one has a caller which wants only the largest block.
extern "C" bool HeapCompactor_GetLowHalfSnapshot(SIZE_T* largestOut,
                                                 SIZE_T* totalOut,
                                                 unsigned long* ageMsOut) {
    if (!g_lastWalkTick) return false;
    if (largestOut) *largestOut = g_lastWalkLow;
    if (totalOut)   *totalOut   = g_lastWalkLowTotal;
    if (ageMsOut)   *ageMsOut   = (unsigned long)(GetTickCount() - g_lastWalkTick);
    return true;
}


// The largest free run over all of user address space, from the same cached
// walk. Two other modules had their own copy of this walk and ran it on the
// main thread - one of them up to once a second, for a number an addon
// displays. False when the monitor has not walked yet.
extern "C" bool HeapCompactor_GetLastLargestFree(SIZE_T* largestOut,
                                                 unsigned long* ageMsOut) {
    if (!g_lastWalkTick) return false;
    if (largestOut) *largestOut = g_lastWalkFree;
    if (ageMsOut)   *ageMsOut   = (unsigned long)(GetTickCount() - g_lastWalkTick);
    return true;
}

// Cheap cached read (no VirtualQuery walk) for per-frame consumers like the GC
// step. Returns the last value the monitor sampled; 0 means "not sampled yet".
extern "C" SIZE_T HeapCompactor_GetCachedLargestBlock() {
    return g_lastLargestBlock.load();
}

// The same reading for the low half, which is the one every consumer of the
// line above actually wanted.
//
// txtsd's 2026-09-02 session: "VA Space (below 2GB): Free=16MB
// LargestBlock=1MB WARNING: fragmented" four reports running, while this
// module reported 1237MB free across all of user address space in the same
// second. The client allocates from below 2GB. Every consumer read the second
// number, so the pressure governor sat in GREEN and the Lua collector never
// stepped harder, for a whole session, while the client could not get 2MB.
// ElvUI.lua was written to disk that session as ")_.lua".
//
// Zero means the monitor has not walked yet, which callers must treat as "not
// measured" rather than "nothing free" - the same contract as above.
extern "C" SIZE_T HeapCompactor_GetCachedLowHalf() {
    if (!g_lastWalkTick) return 0;
    return g_lastWalkLow;
}

extern "C" void HeapCompactor_GetStats(uint64_t* checks, uint64_t* compactions, 
                                        SIZE_T* lastBlock, SIZE_T* minBlock, SIZE_T* maxBlock) {
    if (checks) *checks = g_checksPerformed.load();
    if (compactions) *compactions = g_compactionsTriggered.load();
    if (lastBlock) *lastBlock = g_lastLargestBlock.load();
    if (minBlock) *minBlock = g_minLargestBlock.load();
    if (maxBlock) *maxBlock = g_maxLargestBlock.load();
}

#else  // TEST_DISABLE_HEAP_COMPACTOR

// Compactor disabled: report "no data" so VA-pressure consumers stay inert.
#include <windows.h>
extern "C" SIZE_T HeapCompactor_GetCachedLargestBlock() { return 0; }
extern "C" SIZE_T HeapCompactor_GetCachedLowHalf() { return 0; }

#endif // TEST_DISABLE_HEAP_COMPACTOR
