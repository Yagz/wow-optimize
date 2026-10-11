#pragma once

#include <windows.h>

namespace LoadingFrameCap {
    // Called from the Sleep hook, on the main thread, with the client's frame limiter as the asker
    // (the sleep that sub_6836D0 makes at 0x00683742). True when this sleep is to be skipped.
    bool SkipSleep(DWORD ms);
    void Init();
    void LogStats();
}
