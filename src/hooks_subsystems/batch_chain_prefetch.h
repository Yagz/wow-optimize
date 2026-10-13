#pragma once

// Prefetches the objects of upcoming M2 batches. See batch_chain_prefetch.cpp.
namespace BatchChainPrefetch {
bool Init();
void Shutdown();
void LogStats();
}
