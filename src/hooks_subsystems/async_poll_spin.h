#pragma once

#include <cstdint>

namespace AsyncPollSpin {
    // While the main thread is inside the client's sleep wrapper through this
    // module's detour: the client's return address into the wrapper's caller.
    // 0 otherwise. The Sleep hook uses it to file the sleep under the right caller.
    uintptr_t MainThreadWrapperCaller();

    void Init();
    void Shutdown();
    void LogStats();
}
