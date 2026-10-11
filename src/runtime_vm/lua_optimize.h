#pragma once

// Lua VM optimizer
#ifndef LUA_OPTIMIZE_H
#define LUA_OPTIMIZE_H

#include <windows.h>
#include <cstdint>
#include <atomic>
#include "loading_defrag.h"

// The raw flags behind LuaOpt::IsReloading() and LuaOpt::IsSwapping(),
// exposed so the inline guard below can read them without calling into
// another module. They live at file scope in lua_optimize.cpp, not inside
// the namespace, and are declared here where they actually are.
extern std::atomic<bool> g_isReloading;
extern std::atomic<bool> g_isSwapping;

namespace LuaOpt {

// Call from worker thread. Validates addresses.
bool PrepareFromWorkerThread();

// Call from hooked_Sleep on main thread.
// First call: initializes GC and writes the LUABOOST_DLL_* markers.
// Subsequent: incremental GC step.
void OnMainThreadSleep(DWORD mainThreadId, double frameMs = 0.0);

// Call on DLL unload.
void Shutdown();

// Combat mode (reduces GC during combat)
void SetCombatMode(bool inCombat);
// Returns true if client is in loading screen
bool IsLoadingMode();

struct Stats {
    bool   initialized;
    bool   gcOptimized;
    bool   allocatorReplaced;
    double luaMemoryKB;
    int    gcStepsTotal;
    int    gcPause;
    int    gcStepMul;
    // What the client itself had before this DLL wrote over them. Needed to
    // compare the two pacings rather than assert that ours is better.
    int    gcPauseOrig;
    int    gcStepMulOrig;
};

Stats GetStats();

// Printed from the periodic report. Lua memory used to appear once at install
// and never again.
void LogStats();

// Thread-safe swap/reload state queries for worker threads
bool IsReloading();
bool IsSwapping();

// The same question the two above answer, for callers hot enough that the calls
// themselves cost more than the answer.
//
// `IsReloading() || IsSwapping()` reads three flags, and it reads them through
// four cross-module call and return pairs: one into each of those two, and one
// more from inside each into LoadingDefrag::IsLoadingActive. A field session
// runs lua_getstr_inline 11522349178 times and every one of them paid all four
// before doing any work, to learn that nothing is reloading.
//
// This reads the same three variables in the same order with no calls at all.
// It cannot drift from the functions above, because it is not a copy of their
// state - it is their state.
//
// One difference, in the safe direction. IsLoadingActive() also expires a stuck
// loading flag after a watchdog interval, and this does not, so in the one case
// where a loading-end event is lost this reports "busy" for longer. A caller
// that reports busy defers to the client, which is the outcome that was already
// correct.
inline bool GuardActive() {
    return ::g_isReloading.load(std::memory_order_acquire)
        || ::g_isSwapping.load(std::memory_order_acquire)
        || LoadingDefrag::g_loadingActive.load(std::memory_order_acquire);
}
DWORD GetLastSwapTick();

// Restore original Lua allocator (safe to call during ExitProcess teardown)
void RestoreAllocator();

} // namespace LuaOpt

// Called on UI reload to clear luaH_getstr and lua_getfield caches
extern "C" void ClearLuaOptCaches();

#endif