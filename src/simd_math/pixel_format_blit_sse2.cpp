// ============================================================================
// Module: pixel_format_blit_sse2.cpp
//
// sub_6ABC20, the client's repacker from 32-bit source pixels into a 16-bit
// surface format by per-channel shifts. It is in every profile collected on
// 2026-09-19 to 09-21, at 0.93% to 2.21% of executing main-thread time
// (wow!0x006ABD1F and 0x006ABD21, inside this function).
//
// There is no SSE2 here, despite the file name and the commit that added it,
// which describes "SSE2 128-bit vector repacking using shuffle masks for 4
// pixels simultaneously". The file contains no vector intrinsic and no inline
// assembly. What it does is the client's own shifts in scalar C, with one
// real change: where the output is 16-bit, two pixels are assembled into one
// 32-bit store instead of two 16-bit ones. Integer shifts leave nothing to
// round, so the result is exact by construction and the comparison against
// the client below is a check on the reading of the format, not on precision.
// The name stays because renaming a file costs its history.
// ============================================================================

#include "pixel_format_blit_sse2.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <emmintrin.h>
#include "config.h"
#include "ab_test.h"
#include "self_bench.h"
#include "MinHook.h"
#include "version.h"
#include "sampling_profiler.h"

extern "C" void Log(const char* fmt, ...);
MH_STATUS WineSafe_CreateHook(void* target, void* detour, void** original);
MH_STATUS WO_EnableHook(void* target);

namespace PixelFormatBlit {
namespace {

typedef void* (__cdecl *OrigBlit_fn)();
static void* g_orig = nullptr;

constexpr uintptr_t kTarget = 0x006ABC20;
static const uint8_t kExpectedPrologue[8] = {
    0x55, 0x8B, 0xEC, 0x83, 0xEC, 0x1C, 0x53, 0x56
};

static bool g_dead = false;
static bool g_abSubject = false;
static int  g_benchId = -1;

constexpr uint32_t kLearnCalls = 500;
constexpr uint32_t kResampleMask = 0xFF;

static uint64_t g_calls = 0;
static uint64_t g_fastBlits = 0;
static uint64_t g_pixelsProcessed = 0;
static uint32_t g_verified = 0;
static uint32_t g_mismatches = 0;
static uint64_t g_controlCalls = 0;

static uint8_t g_verifyClientBuf[4096];
static uint8_t g_verifyFastBuf[4096];

static void Retire(const char* reason) {
    g_dead = true;
    ++g_mismatches;
    Log("[PixelFormatBlit] RETIRED: %s. All subsequent calls delegate to client.", reason);
}

static void* CallClient(void* dst, const void* src, const int* dim, int src_stride, int dst_stride,
                        int a6, int a7, int a8, int a9, int a10, int a11, int a12, int a13) {
    void* result;
    void* const origFn = g_orig;
    __asm {
        push a13
        push a12
        push a11
        push a10
        push a9
        push a8
        push a7
        push a6
        push dst_stride
        push src_stride
        push dim
        mov eax, dst
        mov ecx, src
        call origFn
        add esp, 0x2C
        mov result, eax
    }
    return result;
}

static inline void* BlitFast(void* dst, const void* src, const int* dim, int src_stride, int dst_stride,
                             int a6, int a7, int a8, int a9, int a10, int a11, int a12, int a13) {
    const int width = dim[0];
    const int height = dim[1];

    if (width <= 0 || height <= 0) {
        return dst;
    }

    const uint8_t r_src_shift = (uint8_t)a6;
    const uint8_t g_src_shift = (uint8_t)a7;
    const uint8_t b_src_shift = (uint8_t)a8;
    const uint8_t a_src_shift = (uint8_t)a9;

    const uint8_t r_dst_shift = (uint8_t)a10;
    const uint8_t g_dst_shift = (uint8_t)a11;
    const uint8_t b_dst_shift = (uint8_t)a12;
    const uint8_t a_dst_shift = (uint8_t)a13;

    uintptr_t cur_dst = (uintptr_t)dst;
    const uint8_t* cur_src = (const uint8_t*)src;

    if (width < 2) {
        // Narrow 1-pixel column path
        for (int y = 0; y < height; ++y) {
            const uint8_t* s = cur_src;
            uint16_t* d = (uint16_t*)cur_dst;
            for (int x = 0; x < width; ++x) {
                const uint16_t p = (uint16_t)(
                    (((uint32_t)(s[0] >> g_src_shift)) << g_dst_shift) |
                    (((uint32_t)(s[1] >> r_src_shift)) << r_dst_shift) |
                    (((uint32_t)(s[2] >> b_src_shift)) << b_dst_shift) |
                    (((uint32_t)(s[3] >> a_src_shift)) << a_dst_shift)
                );
                *d++ = p;
                s += 4;
            }
            cur_src += src_stride;
            cur_dst += dst_stride;
        }
    } else {
        // 2-pixel packed dword path
        for (int y = 0; y < height; ++y) {
            const uint8_t* s = cur_src;
            uint32_t* d = (uint32_t*)cur_dst;
            int x = 0;

            // Process pairs of pixels
            for (; x <= width - 2; x += 2) {
                // Pixel 0 (low 16 bits of dword)
                const uint32_t p0 =
                    (((uint32_t)(s[0] >> g_src_shift)) << g_dst_shift) |
                    (((uint32_t)(s[1] >> r_src_shift)) << r_dst_shift) |
                    (((uint32_t)(s[2] >> b_src_shift)) << b_dst_shift) |
                    (((uint32_t)(s[3] >> a_src_shift)) << a_dst_shift);

                // Pixel 1 (high 16 bits of dword)
                const uint32_t p1 =
                    (((uint32_t)(s[4] >> g_src_shift)) << (g_dst_shift + 16)) |
                    (((uint32_t)(s[5] >> r_src_shift)) << (r_dst_shift + 16)) |
                    (((uint32_t)(s[6] >> b_src_shift)) << (b_dst_shift + 16)) |
                    (((uint32_t)(s[7] >> a_src_shift)) << (a_dst_shift + 16));

                *d++ = (p1 | p0);
                s += 8;
            }

            // Remainder single pixel
            if (x < width) {
                const uint16_t p = (uint16_t)(
                    (((uint32_t)(s[0] >> g_src_shift)) << g_dst_shift) |
                    (((uint32_t)(s[1] >> r_src_shift)) << r_dst_shift) |
                    (((uint32_t)(s[2] >> b_src_shift)) << b_dst_shift) |
                    (((uint32_t)(s[3] >> a_src_shift)) << a_dst_shift)
                );
                *(uint16_t*)d = p;
            }

            cur_src += src_stride;
            cur_dst += dst_stride;
        }
    }

    g_pixelsProcessed += (uint64_t)width * (uint64_t)height;
    return dst;
}

static __declspec(noinline) void* VerifyWithClient(void* dst, const void* src, const int* dim, int src_stride, int dst_stride,
                                                  int a6, int a7, int a8, int a9, int a10, int a11, int a12, int a13) {
    const size_t byteSize = (size_t)dim[1] * (size_t)dst_stride;
    if (byteSize == 0 || byteSize > sizeof(g_verifyClientBuf)) {
        return BlitFast(dst, src, dim, src_stride, dst_stride, a6, a7, a8, a9, a10, a11, a12, a13);
    }

    const uint64_t t0 = SelfBench::Now();
    CallClient(g_verifyClientBuf, src, dim, src_stride, dst_stride, a6, a7, a8, a9, a10, a11, a12, a13);
    const uint64_t clientCycles = SelfBench::Now() - t0;

    const uint64_t t1 = SelfBench::Now();
    BlitFast(g_verifyFastBuf, src, dim, src_stride, dst_stride, a6, a7, a8, a9, a10, a11, a12, a13);
    const uint64_t fastCycles = SelfBench::Now() - t1;

    if (std::memcmp(g_verifyClientBuf, g_verifyFastBuf, byteSize) != 0) {
        // The retire line below names no format, so a tester's log could say
        // that this module disagreed with the client and nothing about where.
        // Where the first differing byte falls tells a wrong channel mapping
        // (inside the pixels, on every row) from a difference in the stride
        // padding (past the width), and the arguments say which call it was.
        size_t first = 0, differing = 0;
        for (size_t i = 0; i < byteSize; ++i) {
            if (g_verifyClientBuf[i] != g_verifyFastBuf[i]) {
                if (!differing) first = i;
                ++differing;
            }
        }
        const size_t stride = dst_stride > 0 ? (size_t)dst_stride : 1;
        const unsigned row = (unsigned)(first / stride);
        const unsigned col = (unsigned)(first % stride);
        const unsigned px  = col / 2;                      // the destination is 16 bits a pixel
        const bool inPixels = (int)px < dim[0] && src_stride > 0;
        char srcText[16] = "n/a";
        if (inPixels) {
            uint32_t srcPixel = 0;
            std::memcpy(&srcPixel, (const uint8_t*)src + (size_t)row * (size_t)src_stride + (size_t)px * 4u, 4);
            _snprintf(srcText, sizeof(srcText) - 1, "0x%08X", (unsigned)srcPixel);
            srcText[sizeof(srcText) - 1] = 0;
        }
        uint16_t clientWord = 0, oursWord = 0;
        const size_t w = first & ~(size_t)1;
        if (w + 2 <= byteSize) {
            std::memcpy(&clientWord, g_verifyClientBuf + w, 2);
            std::memcpy(&oursWord, g_verifyFastBuf + w, 2);
        }
        Log("[PixelFormatBlit] disagreement on call %llu: %d x %d, src stride %d, dst stride %d, "
            "arguments a6..a13 = %d %d %d %d %d %d %d %d. %u of %u bytes differ; the first is "
            "row %u, byte %u of the row (%s), client word 0x%04X, ours 0x%04X, "
            "source bytes there %s.",
            (unsigned long long)g_calls, dim[0], dim[1], src_stride, dst_stride,
            a6, a7, a8, a9, a10, a11, a12, a13,
            (unsigned)differing, (unsigned)byteSize, row, col,
            inPixels ? "inside the pixels" : "in the stride padding",
            (unsigned)clientWord, (unsigned)oursWord, srcText);
        Retire("pixel buffer mismatch against client blit");
        std::memcpy(dst, g_verifyClientBuf, byteSize);
        return dst;
    }

    ++g_verified;
    if (g_benchId >= 0) {
        SelfBench::Pair(g_benchId, fastCycles, clientCycles);
    }

    std::memcpy(dst, g_verifyFastBuf, byteSize);
    return dst;
}

__declspec(safebuffers)
static void* __cdecl Hook_PixelFormatBlit(void* dst, const void* src, const int* dim, int src_stride, int dst_stride,
                                         int a6, int a7, int a8, int a9, int a10, int a11, int a12, int a13) {
    ++g_calls;

    if (g_dead) {
        return CallClient(dst, src, dim, src_stride, dst_stride, a6, a7, a8, a9, a10, a11, a12, a13);
    }

    if (g_abSubject && AbTest::StandAside()) {
        ++g_controlCalls;
        return CallClient(dst, src, dim, src_stride, dst_stride, a6, a7, a8, a9, a10, a11, a12, a13);
    }

    const bool isLearning = (g_verified < kLearnCalls);
    const bool shouldVerify = isLearning || ((g_calls & kResampleMask) == 0);

    if (shouldVerify) {
        return VerifyWithClient(dst, src, dim, src_stride, dst_stride, a6, a7, a8, a9, a10, a11, a12, a13);
    }

    void* const ret = BlitFast(dst, src, dim, src_stride, dst_stride, a6, a7, a8, a9, a10, a11, a12, a13);
    ++g_fastBlits;
    return ret;
}

static __declspec(naked) void Thunk_PixelFormatBlit() {
    __asm {
        push ebp
        mov ebp, esp

        // Push arguments in __cdecl order (right-to-left):
        // [ebp+0x30] = a13
        // ...
        // [ebp+0x08] = dim
        push dword ptr [ebp+0x30] // a13
        push dword ptr [ebp+0x2C] // a12
        push dword ptr [ebp+0x28] // a11
        push dword ptr [ebp+0x24] // a10
        push dword ptr [ebp+0x20] // a9
        push dword ptr [ebp+0x1C] // a8
        push dword ptr [ebp+0x18] // a7
        push dword ptr [ebp+0x14] // a6
        push dword ptr [ebp+0x10] // dst_stride
        push dword ptr [ebp+0x0C] // src_stride
        push dword ptr [ebp+0x08] // dim
        push ecx                  // src
        push eax                  // dst

        call Hook_PixelFormatBlit
        add esp, 0x34

        mov esp, ebp
        pop ebp
        retn
    }
}

} // anonymous namespace

void Init() {
    if (!Config::g_settings.OptPixelFormatBlit) {
        return;
    }

    void* const target = (void*)kTarget;
    if (!WowOpt_ClientPatchAllowed(target)) {
        Log("[PixelFormatBlit] NOT active: client patches disallowed at 0x%08X", (uintptr_t)target);
        return;
    }

    if (std::memcmp(target, kExpectedPrologue, sizeof(kExpectedPrologue)) != 0) {
        Log("[PixelFormatBlit] NOT active: prologue mismatch at 0x%08X", (uintptr_t)target);
        return;
    }

    const MH_STATUS status = WineSafe_CreateHook(target, (void*)&Thunk_PixelFormatBlit, (void**)&g_orig);
    if (status != MH_OK) {
        Log("[PixelFormatBlit] NOT active: MH_CreateHook failed (%d) at 0x%08X", status, (uintptr_t)target);
        return;
    }

    if (WO_EnableHook(target) != MH_OK) {
        Log("[PixelFormatBlit] NOT active: MH_EnableHook failed at 0x%08X", (uintptr_t)target);
        return;
    }

    g_benchId = SelfBench::Register("PixelFormatBlit");
    SamplingProfiler::RegisterSelfSymbol("PixelFormatBlit", (const void*)kTarget);

    Log("[PixelFormatBlit] ACTIVE on pixel format repacker (sub_6ABC20 @ 0x%08X, %u learn calls, 1/256 sampling). "
        "Scalar shifts with two 16-bit pixels per 32-bit store - not SSE2, whatever the file is called.",
        (uintptr_t)kTarget, kLearnCalls);

    if (AbTest::IsSubject("PixelFormatBlit", &g_abSubject)) {
        Log("[PixelFormatBlit]   under A/B test (subject=%d)", g_abSubject ? 1 : 0);
    }
}

void Shutdown() {
}

void LogStats() {
    if (!Config::g_settings.OptPixelFormatBlit) return;
    Log("[PixelFormatBlit] calls=%llu fast=%llu pixels=%llu verified=%u mismatches=%u ctrl=%llu dead=%d",
        g_calls, g_fastBlits, g_pixelsProcessed, g_verified, g_mismatches, g_controlCalls, g_dead ? 1 : 0);
}

} // namespace PixelFormatBlit
