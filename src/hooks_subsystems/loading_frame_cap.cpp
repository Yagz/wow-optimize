// ============================================================================
// Module: loading_frame_cap.cpp
//
// The client's frame limiter, sub_6836D0, keeps frames to the smaller of the maxfps and maxfpsbk
// settings by sleeping the rest of each frame period through its Sleep wrapper (sub_86B280), at
// 0x00683742. It does this in a loading screen as it does in the world. A frame there that has
// nothing left to do sleeps a full period, 15 ms at the default of 64, and the work the next frame
// starts is the load's.
//
// In a Frostmourne Rebuffed session of four loads of 17 to 24 s each (nobus, 2026-10-10) the
// main thread slept 7.4 s of 74 s at that site inside the loading screens. In the same four loads
// without a second reader thread it slept 1.9 s of 77 s there and 4.5 s in the pump's 1 ms wait,
// against 1.8 s in the pump with it: the wait that one reader left is replaced by frames that
// finish early and sleep out their period. Each millisecond of it is one in which the main thread
// does nothing and the load does not move. This skips that one sleep while LoadingState says a
// loading screen is up. It does not touch the cap outside loading screens, the limiter's own
// bookkeeping (it still stores the next frame's time) or any other sleep.
//
// What is not known: how much of a load is carried by frames. The 7.4 s is the ceiling of the
// gain; if the load's work does not wait for the next frame the gain is smaller, and none of the
// rig can show it because it never enters a world. The "Load took" lines with the switch on and
// off are the measurement.
//
// Two guards. A loading window longer than 150 s stops skipping, so a missed PLAYER_ENTERING_WORLD
// cannot leave the cap off for a session. And the hook sits on kernel32!Sleep, outside wow.exe, so
// it writes nothing into the client and runs under No Client Patches.
//
// Off by default (General/LoadingFrameCap), experimental.
// ============================================================================
#include "loading_frame_cap.h"
#include "config.h"
#include "loading_state.h"

extern "C" void Log(const char* fmt, ...);

namespace LoadingFrameCap {
namespace {

constexpr DWORD kMaxWindowMs = 150000;

// Plain counters, main thread only.
unsigned long g_asked = 0;          // frame limiter sleeps seen, in or out of a loading screen
unsigned long g_skipped = 0;
unsigned long g_skippedMs = 0;      // what the limiter asked for, not what a sleep would have taken
unsigned long g_stuck = 0;          // sleeps kept because a window ran past the guard
unsigned long g_windows = 0;
bool g_inWindow = false;
DWORD g_windowStart = 0;

}  // namespace

bool SkipSleep(DWORD ms) {
    if (!Config::g_settings.OptLoadingFrameCap) return false;
    ++g_asked;
    if (!LoadingState::IsLoading()) {
        g_inWindow = false;
        return false;
    }
    const DWORD now = GetTickCount();
    if (!g_inWindow) {
        g_inWindow = true;
        g_windowStart = now;
        ++g_windows;
    }
    if (now - g_windowStart > kMaxWindowMs) {
        ++g_stuck;
        return false;
    }
    ++g_skipped;
    g_skippedMs += ms;
    return true;
}

void Init() {
    if (!Config::g_settings.OptLoadingFrameCap) return;
    Log("[LoadingFrameCap] ACTIVE: the client's frame limiter (sub_6836D0) does not sleep while a loading screen is up, "
        "for at most %lu s of one loading screen. The cap outside loading screens is untouched. It only has work on a "
        "client whose frame rate is capped: with maxFPS 0 and the window active the limiter returns before it sleeps. "
        "The ms skipped below is the most a load can have gained.", (unsigned long)(kMaxWindowMs / 1000));
}

void LogStats() {
    if (!Config::g_settings.OptLoadingFrameCap) return;
    if (g_asked == 0) {
        // sub_6836D0 returns before its sleep when maxFPS is 0 and the window is
        // active, which is every uncapped client in the foreground. A tester ran it
        // through three raid sessions that way and it never had a sleep to skip.
        Log("[LoadingFrameCap] the client's frame limiter never slept, so there was nothing to skip: this is what an "
            "uncapped client (maxFPS 0) in the foreground does, and the switch does nothing there.");
        return;
    }
    Log("[LoadingFrameCap] frame limiter sleeps seen %lu; skipped %lu (%lu ms asked for) in %lu loading screen(s); kept %lu "
        "because a loading screen ran past %lu s. Plain counters.",
        g_asked, g_skipped, g_skippedMs, g_windows, g_stuck, (unsigned long)(kMaxWindowMs / 1000));
}

}  // namespace LoadingFrameCap
