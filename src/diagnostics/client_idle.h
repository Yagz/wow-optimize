#pragma once

#include <windows.h>
#include <cstdint>

// The client's "nothing to draw" sleep.
//
// sub_4A8720 draws the world when two device flags (CGxDevice+0xF58 and +0xF60)
// are both set. When either is clear it draws nothing and sleeps sub_86B2B0()
// milliseconds, 25 at a time, through its wrapper sub_86B280, calling from
// 0x004A8B22 and 0x004A8B45. A frame that contains one of these sleeps is the
// client idling - a minimised window or a device that is not ready - and lasts
// about 250 ms. It is not a stall. In one tester's log 391 of the 469 frames the
// freeze catcher wrote up were this, and in an A/B session a run of 22 of them
// landed in one stint and was read as that half's long frames.
//
// hooked_Sleep counts every such sleep on the main thread. Each instrument keeps
// its own copy of the count from its previous frame and compares; nothing is ever
// reset, so the order the frame-boundary instruments run in does not matter.
namespace ClientIdle {

__declspec(selectany) volatile LONG g_sleeps = 0;   // plain increment, main thread only

inline bool IsIdleSleepCaller(uintptr_t ret) {
    return ret == 0x004A8B27 || ret == 0x004A8B4A;
}

// True when at least one idle sleep happened since `seen` was last updated.
inline bool SinceLast(LONG& seen) {
    const LONG now = g_sleeps;
    const bool idle = now != seen;
    seen = now;
    return idle;
}

}  // namespace ClientIdle
