// ============================================================================
// Module: gx_ring.h
//
// The command ring and the per-frame payload arena behind the D3D9 render
// thread. One producer (the client's main thread), one consumer (the render
// thread), no locks.
//
// Memory ordering: this is x86, where a store is never reordered with an older
// store and a load is never reordered with an older load. A compiler barrier
// between "write the command" and "publish the cursor" is therefore the whole
// protocol. The only hardware fence is in Flush(), because publishing the
// cursor and then reading the consumer's sleeping flag is a store followed by a
// load, which x86 does reorder.
//
// The rings live in memory taken from HighTables (top-down, outside this DLL's
// image) once at init. Nothing here allocates while a frame is running.
// ============================================================================

#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d9.h>
#include <intrin.h>
#include <stdint.h>
#include <string.h>

namespace GxRT {

constexpr uint32_t kRingBytes         = 4u << 20;     // power of two
constexpr uint32_t kArenaBytes        = 16u << 20;    // per frame; two of them
constexpr uint32_t kMaxDirectLock     = 512u << 10;   // largest zero-copy buffer lock
constexpr uint32_t kMaxFramesInFlight = 1;            // main thread may run this far ahead
constexpr uint32_t kFlushEvery        = 32;           // commands between publications
constexpr uint32_t kRingMask          = kRingBytes - 1;

#define GX_COMPILER_BARRIER() _ReadWriteBarrier()

enum Op : uint16_t {
    OP_WRAP = 0,
    OP_SET_RS, OP_SET_TSS, OP_SET_SAMPLER, OP_SET_TEXTURE, OP_SET_TRANSFORM,
    OP_SET_VIEWPORT, OP_SET_MATERIAL, OP_SET_LIGHT, OP_LIGHT_ENABLE, OP_SET_CLIP,
    OP_SET_SCISSOR, OP_SET_STREAM, OP_SET_STREAM_FREQ, OP_SET_INDICES,
    OP_SET_VDECL, OP_SET_FVF, OP_SET_VS, OP_SET_PS, OP_SET_VS_CONSTF, OP_SET_PS_CONSTF,
    OP_SET_RT, OP_SET_DS, OP_BEGIN_SCENE, OP_END_SCENE, OP_CLEAR,
    OP_DRAW, OP_DRAW_INDEXED, OP_BUFFER_UPLOAD, OP_PRESENT,
    OP__COUNT
};

struct CmdHdr { uint16_t op; uint16_t pad; uint32_t bytes; };     // 8 bytes, bytes % 8 == 0

// ---- commands ---------------------------------------------------------------
struct CmdRS        { CmdHdr h; DWORD state, value; };
struct CmdTSS       { CmdHdr h; DWORD stage, type, value; };
struct CmdSampler   { CmdHdr h; DWORD sampler, type, value; };
struct CmdTexture   { CmdHdr h; DWORD stage; IDirect3DBaseTexture9* tex; };
struct CmdTransform { CmdHdr h; DWORD state; D3DMATRIX m; };
struct CmdViewport  { CmdHdr h; D3DVIEWPORT9 vp; };
struct CmdMaterial  { CmdHdr h; D3DMATERIAL9 m; };
struct CmdLight     { CmdHdr h; DWORD index; D3DLIGHT9 l; };
struct CmdLightEn   { CmdHdr h; DWORD index; BOOL enable; };
struct CmdClip      { CmdHdr h; DWORD index; float plane[4]; };
struct CmdScissor   { CmdHdr h; RECT r; };
struct CmdStream    { CmdHdr h; UINT n; IDirect3DVertexBuffer9* vb; UINT offset, stride; };
struct CmdStreamFreq{ CmdHdr h; UINT n, divider; };
struct CmdIndices   { CmdHdr h; IDirect3DIndexBuffer9* ib; };
struct CmdObject    { CmdHdr h; IUnknown* obj; };                  // vdecl / vs / ps / depth
struct CmdRT        { CmdHdr h; DWORD index; IDirect3DSurface9* surf; };
struct CmdFVF       { CmdHdr h; DWORD fvf; };
struct CmdConstF    { CmdHdr h; UINT start, count; /* float data[count*4] follows */ };
struct CmdClear     { CmdHdr h; DWORD count, flags; D3DCOLOR color; float z; DWORD stencil; /* D3DRECT[count] follows */ };
struct CmdDraw      { CmdHdr h; DWORD type; UINT start, prims; };
struct CmdDrawIdx   { CmdHdr h; DWORD type; INT baseVertex; UINT minVertex, numVertices, startIndex, prims; };
struct CmdScene     { CmdHdr h; };
struct CmdPresent   { CmdHdr h; DWORD flags; RECT src, dst; HWND wnd; };   // flags: 1 src, 2 dst
struct CmdUpload    {
    CmdHdr h;
    void* buf;                      // the real buffer
    void* lockFn;                   // that buffer's own Lock and Unlock, from its vtable
    void* unlockFn;
    const uint8_t* data;            // in the frame arena
    uint32_t offset, size, lockFlags;
    uint32_t pad;
};

// ---- the ring ---------------------------------------------------------------
struct Ring {
    uint8_t* base;

    // producer side
    alignas(64) uint32_t wr;                        // private write cursor
    uint32_t rdCache;
    uint32_t unflushed;

    // shared
    alignas(64) volatile uint32_t pub;              // producer publishes here
    alignas(64) volatile uint32_t rd;               // consumer finished up to here
    alignas(64) volatile LONG     sleeping;
    HANDLE evData;                                  // wakes the consumer
    HANDLE evFrame;                                 // wakes the producer after a Present
    volatile uint32_t framesProduced;
    volatile uint32_t framesConsumed;

    void Flush() {
        GX_COMPILER_BARRIER();
        pub = wr;
        unflushed = 0;
        _mm_mfence();                               // publish, then look at 'sleeping'
        if (sleeping) SetEvent(evData);
    }
};

extern Ring g_ring;
extern volatile LONG g_dead;                        // render thread gone: everything goes direct
extern uint64_t g_ticksRingFull;                    // cycles the producer spent waiting, by cause
extern unsigned long g_waitsRingFull;

// The whole of RingReserve below, out of line: a command that wraps the ring or
// finds the cached read cursor too close. Defined in gx_render_thread.cpp.
uint8_t* RingReserveSlow(uint32_t n);

// Space for `n` bytes at the write cursor, contiguous; publishes and waits if the
// consumer is behind. `n` is a multiple of 8.
//
// Every queued call comes through here, about 1.4 billion times in a 28-minute raid
// session, and MSVC declined to inline the full version: each command paid a call,
// a frame and a return around three compares. The common case - room before the end
// of the ring and before the cached read cursor - is inlined; anything else takes
// the full routine, which re-derives everything from scratch.
__forceinline uint8_t* RingReserve(uint32_t n) {
    Ring& R = g_ring;
    const uint32_t pos = R.wr & kRingMask;
    if (pos + n <= kRingBytes && R.wr + n - R.rdCache <= kRingBytes) {
        uint8_t* p = R.base + pos;
        R.wr += n;
        return p;
    }
    return RingReserveSlow(n);
}

inline uint8_t* RingReserveFull(uint32_t n) {
    Ring& R = g_ring;
    uint32_t pos = R.wr & kRingMask;
    uint32_t pad = 0;
    if (pos + n > kRingBytes) pad = kRingBytes - pos;   // never split a command
    if (R.wr + pad + n - R.rdCache > kRingBytes) {
        R.rdCache = R.rd;
        if (R.wr + pad + n - R.rdCache > kRingBytes) {
            const uint64_t t0 = __rdtsc();
            unsigned spins = 0;
            for (;;) {
                R.Flush();
                R.rdCache = R.rd;
                if (R.wr + pad + n - R.rdCache <= kRingBytes) break;
                if (g_dead) break;
                if (++spins < 4000) YieldProcessor(); else SwitchToThread();
            }
            g_ticksRingFull += __rdtsc() - t0;
            ++g_waitsRingFull;
        }
    }
    if (pad) {
        CmdHdr* w = (CmdHdr*)(R.base + pos);
        w->op = OP_WRAP; w->pad = 0; w->bytes = pad;
        R.wr += pad;
        pos = 0;
    }
    uint8_t* p = R.base + pos;
    R.wr += n;
    return p;
}

template<class T> __forceinline T* Emit(Op op) {
    constexpr uint32_t n = (sizeof(T) + 7u) & ~7u;
    // Publish before reserving, never after: the caller fills the command in
    // once this returns, so a cursor published past it would show the consumer
    // a half-written command.
    if (g_ring.unflushed >= kFlushEvery) g_ring.Flush();
    T* c = (T*)RingReserve(n);
    ((CmdHdr*)c)->op = op; ((CmdHdr*)c)->pad = 0; ((CmdHdr*)c)->bytes = n;
    ++g_ring.unflushed;
    return c;
}

// A command followed by `extra` bytes of payload.
template<class T> __forceinline T* EmitVar(Op op, uint32_t extra) {
    const uint32_t n = (sizeof(T) + extra + 7u) & ~7u;
    if (g_ring.unflushed >= kFlushEvery) g_ring.Flush();
    T* c = (T*)RingReserve(n);
    ((CmdHdr*)c)->op = op; ((CmdHdr*)c)->pad = 0; ((CmdHdr*)c)->bytes = n;
    ++g_ring.unflushed;
    return c;
}

// ---- per-frame payload arena --------------------------------------------------
// Two linear arenas, one per frame in flight. The producer flips at Present, after
// it has waited for the frame before the last to finish, so the arena it starts
// writing was last read two frames ago and is free without a per-block flag.
struct FrameArena {
    uint8_t* base[2];
    uint32_t used;
    uint32_t cur;

    // 16-byte aligned, or null when this frame's arena is full.
    uint8_t* Alloc(uint32_t n) {
        n = (n + 15u) & ~15u;
        if (n > kArenaBytes - used) return nullptr;
        uint8_t* p = base[cur] + used;
        used += n;
        return p;
    }
    void Flip() { cur ^= 1u; used = 0; }
};
extern FrameArena g_arena;

}  // namespace GxRT
