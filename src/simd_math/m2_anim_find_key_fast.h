#pragma once

namespace M2AnimFindKey {

bool Init();
void LogStats();

// True once the search has been compared with the client's for a full
// learning phase and has not disagreed since. The track evaluators call
// anim_find_key_core.h directly while it is.
extern bool g_usable;

} // namespace M2AnimFindKey
