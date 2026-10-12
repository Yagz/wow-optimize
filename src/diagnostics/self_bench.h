// ============================================================================
// Description: Times a replacement against the client on the same input, using
//              the pairs the verification phase already produces.
// Safety & Threading: Main thread, inside a verification path.
// ============================================================================

#pragma once

#include <cstdint>

namespace SelfBench {

// Which side of a pair runs first. The second run finds the inputs the first one
// brought into the cache, so a fixed order leans the ratio one way by up to a
// cache miss a call - the size of the gap that moved three modules to the Lost
// tab. A site that can swap its two calls alternates them. One that cannot,
// because a call changes state the other reads, says which order it keeps, and
// the report names the lean and gives no verdict on it.
enum Order { kAlternating = 0, kOursFirst = 1, kTheirsFirst = 2 };

// Once, at init. Returns a slot id, or -1 when there is no room.
int Register(const char* name, Order order = kAlternating);

// One pair: the same input through both implementations, in cycles.
void Pair(int id, uint64_t oursCycles, uint64_t theirsCycles);

// The clock both halves are timed with.
uint64_t Now();

void LogStats();

} // namespace SelfBench
