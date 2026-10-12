// ============================================================================
// Module: gx_render_thread
//
// Runs the client's IDirect3DDevice9 calls on a thread of our own. The main
// thread writes each state change and draw into a lock-free ring and carries on
// with the next frame's work; the render thread executes them, in order, against
// the real device. See gx_ring.h for the ring and gx_render_thread.cpp for the
// rules about which calls are queued and which wait for the ring to drain.
//
// Off by default (D3d9RenderThread in [Graphics_Sound]). Forced off under
// Wine/Rosetta, with NoClientPatches, and with the draw merger.
// ============================================================================

#pragma once

namespace GxRT {

// Call once, early: hooks the client's D3D loader so the device is created with
// D3DCREATE_MULTITHREADED and handed to us. Returns false when the feature is off
// or cannot run; the reason is logged.
bool Init();

bool IsActive();

// True when fn is one of the thunks this module wrote into the device vtable.
// Another module that hooks the same vtable asks before writing over a slot: a
// hook placed above a thunk, whose original is that thunk, closes a loop with the
// hook this module captured below it.
bool IsThunk(const void* fn);

// True on the render thread itself. Present reaches the state manager's hook on
// that thread while this module is active, and that hook must not run the work
// that belongs to the main thread's frame.
bool OnRenderThread();

// The render thread's id, or 0 when none was started. For the freeze watchdog,
// which samples it when the main thread stalls inside the D3D runtime.
unsigned long RenderThreadId();

// Logs the ring and frame counters in the form a freeze report wants them:
// whether the render thread is mid-command or idle, and how far behind it is.
// Safe from any thread; the reads are unlocked and may be one step stale.
void LogStallState();

// Main thread, from the pump. The render thread is normally only reachable
// through a device the client creates through our hook, and this module loads
// after the client made its first one. Once the client has settled this asks it
// for a device restart, exactly as typing the console command would, so the
// device is made again through the hook.
void OnMainThreadTick();

void LogStats();
void Shutdown();

}  // namespace GxRT
