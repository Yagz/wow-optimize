// Shared between gx_render_thread.cpp and gx_buffers.cpp. Not for other modules;
// they use gx_render_thread.h.
#pragma once

#include "gx_ring.h"

namespace GxRT {

constexpr int kSlots = 119;                         // IDirect3DDevice9 methods

extern volatile LONG     g_active;                  // 1: thunks queue, 0: everything goes direct
extern volatile LONG     g_queueOff;                // 1: an A/B OFF stint; the device stays ours, calls go direct
extern long              g_emulatedLocks;           // buffer locks answered from the arena or a shadow, not yet unlocked
extern DWORD             g_mainTid;                 // the thread that created the device
extern void*             g_orig[kSlots];            // the device's own methods, by vtable slot
extern IDirect3DDevice9* g_dev;
extern unsigned long     g_syncCalls[kSlots];       // calls that had to drain the ring first
extern uint64_t          g_ticksDrain;
extern unsigned long     g_drains;

inline bool OnMain() { return __readfsdword(0x24) == g_mainTid; }   // TEB thread id
inline bool Live()   { return g_active && !g_queueOff && OnMain(); }
// Whether this module owns the device on this thread, queueing or not. Buffers
// are tracked on this, so one created during an A/B OFF stint is still zero-copy
// in the next ON stint.
inline bool Owned()  { return g_active && OnMain(); }

template<class F> inline F O(int slot) { return (F)g_orig[slot]; }

// Wait until the render thread has executed everything queued so far.
void Drain();

// Buffers (gx_buffers.cpp).
typedef HRESULT (__stdcall *F_CreateVB)(IDirect3DDevice9*, UINT, DWORD, DWORD, D3DPOOL,
                                        IDirect3DVertexBuffer9**, HANDLE*);
typedef HRESULT (__stdcall *F_CreateIB)(IDirect3DDevice9*, UINT, DWORD, D3DFORMAT, D3DPOOL,
                                        IDirect3DIndexBuffer9**, HANDLE*);
HRESULT __stdcall T_CreateVertexBuffer(IDirect3DDevice9* d, UINT len, DWORD usage, DWORD fvf,
                                       D3DPOOL pool, IDirect3DVertexBuffer9** pp, HANDLE* shared);
HRESULT __stdcall T_CreateIndexBuffer(IDirect3DDevice9* d, UINT len, DWORD usage, D3DFORMAT fmt,
                                      D3DPOOL pool, IDirect3DIndexBuffer9** pp, HANDLE* shared);
bool BuffersInit();
void BuffersLogStats();

}  // namespace GxRT
