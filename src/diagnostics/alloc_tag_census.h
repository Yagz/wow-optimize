// ============================================================================
// Module: alloc_tag_census
//
// Names who holds the client's heap. Every allocation the client makes through
// its wrapper (sub_76E540, 1764 call sites) carries two arguments the original
// keeps for its out-of-memory message: a string - a source file such as
// ".\src\TextureCache.cpp", or an RTTI type name such as ".?AUFRAMENODE@..." -
// and a line number. This pairs each live block with that tag, so a log can say
// "2.1 GB of the heap is Texture.cpp:2057" instead of "private".
//
// Measurement only. Off unless General/AllocTagCensus is 1.
// ============================================================================

#pragma once

namespace AllocTagCensus {

// Nonzero while recording. The allocator hooks test this and call OnAlloc /
// OnFree only then, so with the switch off they cost one load and a branch.
extern volatile long g_on;

void Init();

// size is what the wrapper actually allocated (rounded to 8). tag and line are
// the wrapper's a2 and exitCode arguments, passed through unread.
void OnAlloc(void* p, unsigned size, const char* tag, unsigned line);

// A pointer that was never recorded is ignored: sampled out, allocated before
// the hook went in, or taken from the CRT directly.
void OnFree(void* p);

void LogStats();

// For the fatal-error report: the same tables, without the heap summaries and without waiting
// for a lock the dying thread may hold.
void LogAtTerminate();

}  // namespace AllocTagCensus
