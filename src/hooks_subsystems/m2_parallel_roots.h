// ============================================================================
// Module: m2_parallel_roots
//
// Animates the root models of an M2 scene on several threads instead of two.
// See m2_parallel_roots.cpp for what the client already does and why the pass
// can be split.
//
// Off by default (Graphics_Sound/M2ParallelRoots), experimental.
// ============================================================================

#pragma once

namespace M2ParallelRoots {

void Init();
void LogStats();

}  // namespace M2ParallelRoots
