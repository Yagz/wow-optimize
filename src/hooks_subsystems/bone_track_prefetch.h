#pragma once

// Follows the bone loop's track chains ahead of the evaluators.
// See bone_track_prefetch.cpp.
namespace BoneTrackPrefetch {
bool Init();
void Shutdown();
void LogStats();
}
