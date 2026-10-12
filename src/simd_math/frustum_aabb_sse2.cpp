// ============================================================================
// Description: SSE2 rewrite of CFrustum::IsAABBVisible.
// Safety & Threading: Main thread, inside world visibility traversal.
// ============================================================================
// sub_9839E0 tests a box against six frustum planes. It is 0.82% of executing
// time in a tester's uncapped session, reached from the world visibility
// traversal, and it is forty-six instructions of which the arithmetic is the
// smaller half.
//
// The larger half is how it picks which corner of the box to test. For each of
// the three components of each plane it reads the plane's sign bit, uses it to
// index a two-entry table of pointers holding the box minimum and maximum, and
// then loads that corner component through the result:
//
//     mov eax, [ecx+4]                  ; plane.z, as bits
//     shr eax, 1Fh                      ; 0 if positive, 1 if negative
//     mov eax, [ebp+eax*4+var_8]        ; -> max corner, or min corner
//     fld dword ptr [eax+8]             ; corner.z
//
// Eighteen of those per call: a sign test, an indexed load, and a dependent load
// through its result. The selection is a blend, and SSE2 does a blend with the
// sign bits themselves as the mask - no branch, no table, no dependent load.
// ---------------------------------------------------------------------------
// The association, and why it is safe to reproduce
//
// Read from the instruction order rather than the pseudocode, every plane
// accumulates the same way:
//
//     (((plane.z * corner.z) + (plane.y * corner.y)) + (plane.x * corner.x)) + plane.w
//
// The same order for all six, which is what makes a packed-double version
// answer identically: the x87 control word is left at 53-bit precision, exactly
// what a double lane carries, so each multiply and add rounds where the client's
// does. Nothing is stored to float in between, and nothing here needs to be -
// the value is compared, not kept.
// ---------------------------------------------------------------------------
// The comparison passes NaN, and that is deliberate
//
// The client compares with `fcomp` then `test ah, 5` / `jnp`. Working the
// condition codes: C0 is set when the distance is below the threshold and C2
// when the comparison is unordered. Masking both and branching on parity means
// the loop continues when the distance is greater, when it is equal, and when
// either operand is a NaN - only a strictly-below distance exits with zero.
//
// C's `d < threshold` is false for NaN, so `if (d < threshold) return 0;`
// answers identically in all four cases without a special case. Getting this
// backwards would cull geometry whenever a coordinate went bad, which is a
// disappearing-world bug rather than a slow one.
//
// The threshold is the float at 0x00AA2E74, loaded onto the x87 stack once
// before the loop and left there. It is read from the client at Init rather than
// written here as a literal - the decompiler prints it as -0.019444443, and this
// project has already been bitten once by a printed literal that differed from
// the bytes on 29.6% of possible inputs.
// ---------------------------------------------------------------------------
// Verification
//
// The function is pure: it reads a frustum and a box and returns 3 or 0, and
// writes nothing. So both versions are simply run and their answers compared,
// with no saving, restoring or predicting - the same shape that made the render
// batch comparator straightforward.
// ============================================================================

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <emmintrin.h>
#include <cstdint>
#include <cstring>

#include "frustum_aabb_sse2.h"
#include "MinHook.h"
#include "version.h"
#include "config.h"
#include "sampling_profiler.h"
#include "ab_test.h"
#include "self_bench.h"
#include "session_verdict.h"

extern "C" void Log(const char* fmt, ...);

MH_STATUS WineSafe_CreateHook(void* target, void* detour, void** original);
MH_STATUS WO_EnableHook(void* target);

namespace FrustumAabb {

namespace {

constexpr uintptr_t kIsVisible = 0x009839E0;
constexpr uintptr_t kThreshold = 0x00AA2E74;
constexpr uintptr_t kIsInside  = 0x00983A60;
constexpr uintptr_t kThresholdInside = 0x00A3FDB8;
constexpr uintptr_t kIsPointVisible = 0x00983D70;
constexpr uintptr_t kIsSphereVisible = 0x00983D20;
constexpr uintptr_t kTranslate       = 0x00983AE0;

constexpr int kPlanes = 6;

// __thiscall with one stack argument, ending in `retn 4`.
typedef int (__fastcall* isVisible_fn)(void* frustum, void* edx, void* aabb);
isVisible_fn orig_IsVisible = nullptr;

typedef int (__fastcall* isInside_fn)(void* frustum, void* edx, void* aabb);
isInside_fn orig_IsInside = nullptr;

typedef void (__fastcall* isPointVisible_fn)(void* frustum, void* edx, const float* pt, uint8_t* outMask);
isPointVisible_fn orig_IsPointVisible = nullptr;

typedef int (__fastcall* isSphereVisible_fn)(void* frustum, void* edx, const float* sphere);
isSphereVisible_fn orig_IsSphereVisible = nullptr;

typedef const float* (__fastcall* translate_fn)(float* frustum, void* edx, const float* delta);
translate_fn orig_Translate = nullptr;

bool g_installed = false;
bool g_armed     = false;
bool g_insideInstalled = false;
bool g_insideArmed     = false;
bool g_insideDead      = false;
unsigned long g_insideCalls = 0;
unsigned long g_insideVerified = 0;
double g_thresholdInside = 0.0;

bool g_pointInstalled = false;
bool g_pointArmed     = false;
bool g_pointDead      = false;
unsigned long g_pointCalls = 0;
unsigned long g_pointVerified = 0;

bool g_sphereInstalled = false;
bool g_sphereArmed     = false;
bool g_sphereDead      = false;
unsigned long g_sphereCalls    = 0;
unsigned long g_sphereVerified = 0;
unsigned long g_sphereCulled   = 0;
unsigned long g_sphereVisible  = 0;

bool g_transInstalled  = false;
bool g_transArmed      = false;
bool g_transDead       = false;
unsigned long g_transCalls    = 0;
unsigned long g_transVerified = 0;
// Set at init when the A/B harness names this module, so the hot path
// tests a plain bool instead of calling out on every invocation.
bool g_abSubject = false;
bool g_dead      = false;
int  g_benchSlot = -1;

// Plain 32-bit, because this is called hard from the visibility walk and a
// locked increment there has eaten whole optimisations in this project before.
//
// But 32 bits is not enough to divide by. Field sessions reach 1.7 billion calls
// and the counter wraps at 4.29 billion, and it wraps before g_visible does
// because every call increments it while only a visible one increments the
// other. Past that point the report divided a real count by a wrapped one:
//
//     1496607690 visibility tests, 1691870587 came back visible (113.0%)
//      106645051 visibility tests, 1314017835 came back visible (1232.1%)
//
// and once, 73322.0%. More things visible than tests run is not a number that
// can happen, and the share was the headline the module exists to report.
//
// So each counter keeps its wraps and the report recombines them as a double.
// This is the same fix the fifteen counters in matrix_copy_sse2 needed, for the
// same reason, found the same way - by an impossible number rather than by
// reading the code.
unsigned long g_calls      = 0;
unsigned long g_callWraps  = 0;
unsigned long g_verified   = 0;
unsigned long g_visible    = 0;
unsigned long g_visWraps   = 0;

inline void BumpCalls() {
    const unsigned long before = g_calls;
    g_calls = before + 1;
    if (g_calls < before) ++g_callWraps;
}

inline void BumpVisible() {
    const unsigned long before = g_visible;
    g_visible = before + 1;
    if (g_visible < before) ++g_visWraps;
}

// Adds to a counter that is allowed to wrap, keeping the count of wraps. The
// plane total reaches twenty billion in a session, which is five times what
// thirty-two bits hold.
inline void Bump(unsigned long& low, unsigned long& wraps, unsigned long by) {
    const unsigned long before = low;
    low = before + by;
    if (low < before) ++wraps;
}

inline double Total(unsigned long low, unsigned long wraps) {
    return (double)low + (double)wraps * 4294967296.0;
}

constexpr unsigned long kVerifyFirst  = 20000;
constexpr unsigned long kResampleMask = 4095;

double g_threshold = 0.0;

// Which plane culled the object before this one, and for which frustum.
//
// The client tests six planes in a fixed order and returns the moment one of
// them culls, so a box rejected by plane four pays for planes one to three
// first. The result does not depend on the order at all: it is "cull if any
// plane culls", a pure existential over six independent tests, and each test
// computes its own plane's distance from its own plane's numbers. Evaluating
// them in a different order returns the same answer for every input, NaN
// included - a NaN distance fails `<` wherever it is tested, so it never culls
// in any order - and the planes that are skipped have no side effects to skip.
//
// So the order is free to change, and the useful order is the one spatial
// coherence hands over: the objects a visibility walk rejects in sequence tend
// to be rejected by the same plane, because they are near each other and the
// frustum has not moved between them. Remembering the last culling plane and
// starting there turns most of those calls into one plane instead of several.
//
// A field session runs 3370465441 of these with 1226911991 of them culled, so
// that is the population. Nothing is saved on a visible box - all six planes run
// whatever the order - and nothing is lost either, because the rotation
// evaluates the same six.
//
// Two things were measured offline before this was written, because the claim
// has two halves and they take different evidence.
//
// That the order cannot change the answer: 400000 random cases, each evaluated
// in all six rotations, with NaN, both infinities and both zeroes among the
// generated plane and box components. Zero disagreements. This is the same class
// of argument as the collision outcode - not that the error is small but that
// there is none - and it holds because each plane distance is computed from that
// plane's own four numbers and nothing else, while the aggregate over planes is
// an existential.
//
// What the hint is worth: one fixed frustum and 19200 boxes handed over in the
// order a grid walk would hand them, against the same boxes shuffled.
//
//     order of objects          planes per call      culled on the first plane
//     spatial walk              3.486 -> 1.854            85.9%
//     shuffled, no coherence    3.486 -> 2.893            34.0%
//
// Both rows are over a set that culls 90% where the field culls 36.4%, so the
// session figure will be smaller: the field mix with the spatial row puts about
// 5.09 plane evaluations per call at about 4.49, which over 3370465441 calls is
// roughly two billion plane evaluations that do not happen. That last step is
// arithmetic over a model of the traversal order rather than a measurement of
// it, which is why the counter below reports the real figure instead.
//
// Keyed by the frustum pointer because shadow cascades are separate frusta with
// separate geometry, and one remembered index shared between them would be
// wrong for both. A different pointer simply starts at plane zero again; this
// is a hint, and a wrong hint costs nothing but the original order.
const void* g_lastFrustum = nullptr;
int         g_lastCull    = 0;

// What the hint is worth, counted rather than assumed: plane evaluations that
// actually ran, against the calls that ran them.
unsigned long g_planeEvals  = 0;
unsigned long g_planeWraps  = 0;
unsigned long g_hintCulled  = 0;   // the remembered plane culled on its first try

// Pick the corner the client would pick, for x and y at once.
//
// Loads are eight bytes, never sixteen. A box is three floats of minimum
// followed by three of maximum, so a sixteen-byte load at the maximum would read
// a fourth float that belongs to nothing - real memory most of the time and a
// fault at the end of a page, in a function called from the visibility walk on
// every object in the world. The client reads three floats and so does this.
inline void Corner(const float* mn, const float* mx, const float* pl,
                   __m128d* outXY, double* outZ) {
    __m128 vmn  = _mm_castpd_ps(_mm_load_sd((const double*)mn));
    __m128 vmx  = _mm_castpd_ps(_mm_load_sd((const double*)mx));
    __m128 vpl  = _mm_castpd_ps(_mm_load_sd((const double*)pl));
    // The plane's own sign bits are the mask: a negative component takes the
    // minimum corner and a positive one the maximum. That is exactly the
    // client's `shr eax, 31` table index, without the table or the dependent
    // load it feeds.
    __m128 mask = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(vpl), 31));
    __m128 sel  = _mm_or_ps(_mm_and_ps(mask, vmn), _mm_andnot_ps(mask, vmx));
    *outXY = _mm_cvtps_pd(sel);

    uint32_t zbits;
    memcpy(&zbits, pl + 2, sizeof(zbits));
    *outZ = (double)((zbits & 0x80000000u) ? mn[2] : mx[2]);
}

// (((plane.z * corner.z) + (plane.y * corner.y)) + (plane.x * corner.x)) + plane.w
// for each of six planes, in that order, stopping the moment one is below the
// threshold. The order is the client's, read from its instruction sequence.
inline int Evaluate(void* frustum, void* aabb) {
    const float* base = (const float*)frustum;
    const float* mn   = (const float*)aabb;
    const float* mx   = mn + 3;

    // Start at the plane that culled the last box, when this is the same
    // frustum. See the note on g_lastCull for why any order is allowed.
    int start = 0;
    if (frustum == g_lastFrustum) {
        start = g_lastCull;
    } else {
        g_lastFrustum = frustum;
        g_lastCull    = 0;
    }

    for (int n = 0; n < kPlanes; n++) {
        int i = start + n;
        if (i >= kPlanes) i -= kPlanes;
        const float* pl = base + 4 * i;

        __m128d cxy;
        double  cz;
        Corner(mn, mx, pl, &cxy, &cz);

        __m128d pxy  = _mm_cvtps_pd(_mm_castpd_ps(_mm_load_sd((const double*)pl)));
        __m128d prod = _mm_mul_pd(pxy, cxy);

        double xterm, yterm;
        _mm_storel_pd(&xterm, prod);
        _mm_storeh_pd(&yterm, prod);

        double d = (((double)pl[2] * cz + yterm) + xterm) + (double)pl[3];

        // Below the threshold is the only outcome that culls. Equal continues,
        // and so does a NaN, because `<` is false for it - which is what the
        // client's parity test on C0 and C2 works out to.
        if (d < g_threshold) {
            Bump(g_planeEvals, g_planeWraps, (unsigned long)(n + 1));
            if (n == 0) ++g_hintCulled;
            g_lastCull = i;
            return 0;
        }
    }
    Bump(g_planeEvals, g_planeWraps, (unsigned long)kPlanes);
    return 3;
}

inline int EvaluateInside(void* frustum, void* aabb) {
    const float* base = (const float*)frustum;
    const float* mn   = (const float*)aabb;
    const float* mx   = mn + 3;

    for (int i = 0; i < kPlanes; i++) {
        const float* pl = base + 4 * i;

        __m128d cxy;
        double  cz;
        Corner(mn, mx, pl, &cxy, &cz);

        __m128d pxy  = _mm_cvtps_pd(_mm_castpd_ps(_mm_load_sd((const double*)pl)));
        __m128d prod = _mm_mul_pd(pxy, cxy);

        double xterm, yterm;
        _mm_storel_pd(&xterm, prod);
        _mm_storeh_pd(&yterm, prod);

        double d = (((double)pl[2] * cz + yterm) + xterm) + (double)pl[3];

        if (d > g_thresholdInside) {
            return 0;
        }
    }
    return 3;
}

inline void EvaluatePoint(void* frustum, const float* pt, uint8_t* outMask) {
    const float* base = (const float*)frustum;
    __m128d pxy_pt = _mm_cvtps_pd(_mm_castpd_ps(_mm_load_sd((const double*)pt)));
    double  pt_z   = (double)pt[2];

    uint8_t mask = 0;
    for (int i = 0; i < kPlanes; i++) {
        const float* pl = base + 4 * i;

        __m128d pxy  = _mm_cvtps_pd(_mm_castpd_ps(_mm_load_sd((const double*)pl)));
        __m128d prod = _mm_mul_pd(pxy, pxy_pt);

        double xterm, yterm;
        _mm_storel_pd(&xterm, prod);
        _mm_storeh_pd(&yterm, prod);

        double d = (((double)pl[2] * pt_z + yterm) + xterm) + (double)pl[3];

        if (d < g_threshold) {
            mask |= (uint8_t)(1u << i);
        }
    }
    *outMask = mask;
}

inline int EvaluateSphere(void* frustum, const float* sphere) {
    const float* base = (const float*)frustum;
    const double sx = (double)sphere[0];
    const double sy = (double)sphere[1];
    const double sz = (double)sphere[2];
    const double neg_r = -(double)sphere[3];

    for (int i = 0; i < kPlanes; ++i) {
        const float* pl = base + 4 * i;

        // Matches stock client x87 double-precision accumulation order:
        // ((nx * sx + nz * sz) + ny * sy) + d
        const double term_x = (double)pl[0] * sx;
        const double term_z = (double)pl[2] * sz;
        const double sum_xz = term_x + term_z;
        const double term_y = (double)pl[1] * sy;
        const double sum_xzy = sum_xz + term_y;
        const double dist = sum_xzy + (double)pl[3];

        if (dist < neg_r) {
            return 0; // Culled by this plane
        }
    }

    return 3; // Visible across all 6 planes
}

}  // namespace

// The checked path, kept out of line so the hook itself carries no exception
// frame.
//
// A __try region costs a prologue on every call into the function that contains
// one, whether or not that call goes anywhere near it. This hook runs 3370465441
// times in a field session and the branch below is taken 20000 times and then
// one call in 4096 - so almost all of those three billion prologues were pushed
// for a region the call never entered.
//
// The guard moves in here with the work it guards. What is left in the caller is
// the armed path, which now runs Evaluate with no frame at all.
//
// That is safe for the same reason it is in the matrix hooks. Evaluate reads the
// frustum's six planes and the box's two corners; when the guard fired, control
// went to orig_IsVisible with the same two pointers, and the client's routine
// reads the same planes and the same box. It cannot succeed where ours faulted.
// And unlike the matrix hooks this needs no new proving phase - arming already
// means 20000 evaluations ran under the guard and none of them faulted, and the
// resample keeps one call in 4096 running under it for the rest of the session.
__declspec(noinline)
static int VerifyAgainstClient(void* frustum, void* edx, void* aabb) {
        // The verification already runs both halves on the same input. Timing
        // it is the only paired comparison this project gets without asking a
        // tester to configure anything.
        // Alternated, so neither half always meets the cold caches and the other the warm ones.
        static unsigned s_benchOrder = 0;
        int mine, theirs;
        if (++s_benchOrder & 1) {
            const uint64_t tA = SelfBench::Now();
            theirs = orig_IsVisible(frustum, edx, aabb);
            const uint64_t tB = SelfBench::Now();
            __try {
                mine = Evaluate(frustum, aabb);
            } __except (EXCEPTION_EXECUTE_HANDLER) {
                return theirs;
            }
            SelfBench::Pair(g_benchSlot, SelfBench::Now() - tB, tB - tA);
        } else {
            const uint64_t tA = SelfBench::Now();
            __try {
                mine = Evaluate(frustum, aabb);
            } __except (EXCEPTION_EXECUTE_HANDLER) {
                return orig_IsVisible(frustum, edx, aabb);
            }
            const uint64_t tB = SelfBench::Now();
            theirs = orig_IsVisible(frustum, edx, aabb);
            SelfBench::Pair(g_benchSlot, tB - tA, SelfBench::Now() - tB);
        }
        g_verified++;

        if (mine != theirs) {
            g_dead = true;
            Verdict::Add(Verdict::Bad,
                         "FrustumAabb disagreed with the client and retired itself for "
                         "this session");
            Log("[FrustumAabb] DISAGREED with the client after %lu tests - retired "
                "for this session, every test now goes to the client's own code. "
                "It answered %d and this answered %d.", g_verified, theirs, mine);
            return theirs;
        }
        if (!g_armed && g_verified >= kVerifyFirst) {
            g_armed = true;
            Log("[FrustumAabb] armed: %lu tests agreed with the client. Now "
                "answering directly and rechecking one in %lu.",
                g_verified, kResampleMask + 1);
        }
        if (theirs) BumpVisible();
        return theirs;
}

int __fastcall Hooked_IsVisibleBody(void* frustum, void* edx, void* aabb) {
    BumpCalls();
    if (g_dead || !frustum || !aabb) return orig_IsVisible(frustum, edx, aabb);

    if (!g_armed || (g_calls & kResampleMask) == 0)
        return VerifyAgainstClient(frustum, edx, aabb);

    // No exception frame on this path. See the note above VerifyAgainstClient.
    const int r = Evaluate(frustum, aabb);
    if (r) BumpVisible();
    return r;
}

// The detour proper, kept apart from the body above for one reason: the
// A/B harness times the call, and a scope guard that closed the sample on
// every return path cannot be used in a function containing __try - MSVC
// refuses object unwinding alongside SEH. A wrapper has no __try of its own,
// so one pair of reads covers every path the body can take, including the
// ones it takes out of an exception handler.
//
// When no test names this module the whole thing is a branch on a false
// global followed by a direct call.
int __fastcall Hooked_IsVisible(void* frustum, void* edx, void* aabb) {
    if (!g_abSubject) return Hooked_IsVisibleBody(frustum, edx, aabb);
    unsigned long long t = AbTest::TickIn();
    int r = AbTest::StandAside() ? orig_IsVisible(frustum, edx, aabb)
                                       : Hooked_IsVisibleBody(frustum, edx, aabb);
    AbTest::TickOut(t);
    return r;
}

__declspec(noinline)
static int VerifyAgainstClientInside(void* frustum, void* edx, void* aabb) {
    int mine;
    __try {
        mine = EvaluateInside(frustum, aabb);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return orig_IsInside(frustum, edx, aabb);
    }
    int theirs = orig_IsInside(frustum, edx, aabb);
    g_insideVerified++;

    if (mine != theirs) {
        g_insideDead = true;
        Verdict::Add(Verdict::Bad,
                     "FrustumAabb IsInside disagreed with the client and retired itself");
        Log("[FrustumAabb] IsInside DISAGREED with client after %lu tests - retired for session "
            "(client=%d, sse2=%d)", g_insideVerified, theirs, mine);
        return theirs;
    }
    if (!g_insideArmed && g_insideVerified >= kVerifyFirst) {
        g_insideArmed = true;
        Log("[FrustumAabb] IsInside armed: %lu tests agreed with client.", g_insideVerified);
    }
    return theirs;
}

int __fastcall Hooked_IsInside(void* frustum, void* edx, void* aabb) {
    ++g_insideCalls;
    if (g_insideDead || !frustum || !aabb) return orig_IsInside(frustum, edx, aabb);

    if (!g_insideArmed || (g_insideCalls & kResampleMask) == 0)
        return VerifyAgainstClientInside(frustum, edx, aabb);

    return EvaluateInside(frustum, aabb);
}

__declspec(noinline)
static void VerifyAgainstClientPoint(void* frustum, void* edx, const float* pt, uint8_t* outMask) {
    uint8_t mine = 0xFF;
    __try {
        EvaluatePoint(frustum, pt, &mine);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        orig_IsPointVisible(frustum, edx, pt, outMask);
        return;
    }
    uint8_t theirs = 0xFF;
    orig_IsPointVisible(frustum, edx, pt, &theirs);
    *outMask = theirs;
    g_pointVerified++;

    if (mine != theirs) {
        g_pointDead = true;
        Verdict::Add(Verdict::Bad,
                     "FrustumAabb IsPointVisible disagreed with the client and retired itself");
        Log("[FrustumAabb] IsPointVisible DISAGREED with client after %lu tests - retired for session "
            "(client=0x%02X, sse2=0x%02X)", g_pointVerified, (unsigned)theirs, (unsigned)mine);
        return;
    }
    if (!g_pointArmed && g_pointVerified >= kVerifyFirst) {
        g_pointArmed = true;
        Log("[FrustumAabb] IsPointVisible armed: %lu tests agreed with client.", g_pointVerified);
    }
}

void __fastcall Hooked_IsPointVisible(void* frustum, void* edx, const float* pt, uint8_t* outMask) {
    ++g_pointCalls;
    if (g_pointDead || !frustum || !pt || !outMask) {
        orig_IsPointVisible(frustum, edx, pt, outMask);
        return;
    }

    if (!g_pointArmed || (g_pointCalls & kResampleMask) == 0) {
        VerifyAgainstClientPoint(frustum, edx, pt, outMask);
        return;
    }

    EvaluatePoint(frustum, pt, outMask);
}

__declspec(noinline)
static int VerifyAgainstClientSphere(void* frustum, void* edx, const float* sphere) {
    int mine = 0;
    __try {
        mine = EvaluateSphere(frustum, sphere);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return orig_IsSphereVisible(frustum, edx, sphere);
    }

    const int theirs = orig_IsSphereVisible(frustum, edx, sphere);
    g_sphereVerified++;

    if (mine != theirs) {
        g_sphereDead = true;
        Verdict::Add(Verdict::Bad,
                     "FrustumAabb IsSphereVisible disagreed with the client and retired itself");
        Log("[FrustumAabb] IsSphereVisible DISAGREED with client after %lu tests - retired for session "
            "(client=%d, sse2=%d)", g_sphereVerified, theirs, mine);
        return theirs;
    }

    if (!g_sphereArmed && g_sphereVerified >= kVerifyFirst) {
        g_sphereArmed = true;
        Log("[FrustumAabb] IsSphereVisible armed: %lu tests agreed with client.", g_sphereVerified);
    }

    return theirs;
}

int __fastcall Hooked_IsSphereVisible(void* frustum, void* edx, const float* sphere) {
    ++g_sphereCalls;
    if (g_sphereDead || !frustum || !sphere) {
        return orig_IsSphereVisible(frustum, edx, sphere);
    }

    if (!g_sphereArmed || (g_sphereCalls & kResampleMask) == 0) {
        const int r = VerifyAgainstClientSphere(frustum, edx, sphere);
        if (r == 3) ++g_sphereVisible;
        else ++g_sphereCulled;
        return r;
    }

    const int r = EvaluateSphere(frustum, sphere);
    if (r == 3) ++g_sphereVisible;
    else ++g_sphereCulled;
    return r;
}

inline void EvaluateTranslate(float* frustum, const float* delta) {
    const double dx = (double)delta[0];
    const double dy = (double)delta[1];
    const double dz = (double)delta[2];

    // 8 frustum corners (frustum + 24 to frustum + 47)
    for (int i = 0; i < 8; ++i) {
        float* pt = frustum + 24 + i * 3;
        pt[0] = (float)((double)pt[0] + dx);
        pt[1] = (float)((double)pt[1] + dy);
        pt[2] = (float)((double)pt[2] + dz);
    }

    // 6 frustum planes (frustum + 0 to frustum + 23)
    // Client sub_983AE0 exact stock x87 accumulation order:
    // dot = ((Nz * dz) + (Ny * dy)) + (Nx * dx)
    // new_D = D - dot
    for (int p = 0; p < 6; ++p) {
        float* plane = frustum + p * 4;
        const double nz_dz = (double)plane[2] * dz;
        const double ny_dy = (double)plane[1] * dy;
        const double nx_dx = (double)plane[0] * dx;
        const double dot = (nz_dz + ny_dy) + nx_dx;
        plane[3] = (float)((double)plane[3] - dot);
    }

    // 2 center points (frustum + 48 to frustum + 53)
    for (int i = 0; i < 2; ++i) {
        float* pt = frustum + 48 + i * 3;
        pt[0] = (float)((double)pt[0] + dx);
        pt[1] = (float)((double)pt[1] + dy);
        pt[2] = (float)((double)pt[2] + dz);
    }
}

__declspec(noinline)
static const float* VerifyAgainstClientTranslate(float* frustum, void* edx, const float* delta) {
    float theirs[54];
    float mine[54];
    memcpy(theirs, frustum, sizeof(theirs));
    memcpy(mine, frustum, sizeof(mine));

    __try {
        EvaluateTranslate(mine, delta);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return orig_Translate(frustum, edx, delta);
    }

    orig_Translate(theirs, edx, delta);
    g_transVerified++;

    bool same = true;
    for (int i = 0; i < 54; ++i) {
        uint32_t bt, bm;
        memcpy(&bt, &theirs[i], 4);
        memcpy(&bm, &mine[i], 4);
        if (bt != bm) {
            same = false;
            break;
        }
    }

    if (!same) {
        g_transDead = true;
        Verdict::Add(Verdict::Bad,
                     "FrustumAabb Translate disagreed with client and retired itself");
        Log("[FrustumAabb] Translate DISAGREED with client after %lu calls - retired for session",
            g_transVerified);
        memcpy(frustum, theirs, sizeof(theirs));
        return delta;
    }

    if (!g_transArmed && g_transVerified >= kVerifyFirst) {
        g_transArmed = true;
        Log("[FrustumAabb] Translate armed: %lu calls agreed bit-for-bit with client.", g_transVerified);
    }

    memcpy(frustum, mine, sizeof(mine));
    return delta;
}

const float* __fastcall Hooked_Translate(float* frustum, void* edx, const float* delta) {
    ++g_transCalls;
    if (g_transDead || !frustum || !delta) {
        return orig_Translate(frustum, edx, delta);
    }

    if (!g_transArmed || (g_transCalls & kResampleMask) == 0) {
        return VerifyAgainstClientTranslate(frustum, edx, delta);
    }

    EvaluateTranslate(frustum, delta);
    return delta;
}

static inline bool CheckPrologue8(void* addr, const unsigned char expected[8], const char* name) {
    if (IsBadReadPtr(addr, 8) || memcmp(addr, expected, 8) != 0) {
        Log("[FrustumAabb] BAD PROLOGUE for %s at 0x%08X", name, (uintptr_t)addr);
        return false;
    }
    return true;
}

bool Init() {
    if (!Config::g_settings.OptFrustumAabb) return true;

    if (IsBadReadPtr((void*)kThreshold, 4)) {
        Log("[FrustumAabb] 0x%08X unreadable - not installing", (unsigned)kThreshold);
        return false;
    }

    static const unsigned char kExp_IsVisible[8]       = { 0x55, 0x8B, 0xEC, 0x83, 0xEC, 0x08, 0x8B, 0x45 };
    static const unsigned char kExp_IsInside[8]        = { 0x55, 0x8B, 0xEC, 0x83, 0xEC, 0x08, 0x8B, 0x45 };
    static const unsigned char kExp_IsPointVisible[8]  = { 0x55, 0x8B, 0xEC, 0x8B, 0x55, 0x08, 0x56, 0x8B };
    static const unsigned char kExp_IsSphereVisible[8] = { 0x55, 0x8B, 0xEC, 0x56, 0x8B, 0x75, 0x08, 0xD9 };
    static const unsigned char kExp_Translate[8]       = { 0x55, 0x8B, 0xEC, 0x8B, 0x45, 0x08, 0xD9, 0x41 };

    if (!CheckPrologue8((void*)kIsVisible, kExp_IsVisible, "CFrustum::IsAABBVisible")) {
        return false;
    }

    g_threshold = (double)*(const float*)kThreshold;

    if (WineSafe_CreateHook((void*)kIsVisible, (void*)Hooked_IsVisible,
                            (void**)&orig_IsVisible) != MH_OK) {
        Log("[FrustumAabb] hook NOT created");
        return false;
    }
    if (WO_EnableHook((void*)kIsVisible) != MH_OK) {
        Log("[FrustumAabb] hook created but could not be enabled");
        return false;
    }

    if (!IsBadReadPtr((void*)kThresholdInside, 4) &&
        CheckPrologue8((void*)kIsInside, kExp_IsInside, "CFrustum::IsAABBInside")) {
        g_thresholdInside = (double)*(const float*)kThresholdInside;
        if (WineSafe_CreateHook((void*)kIsInside, (void*)Hooked_IsInside,
                                (void**)&orig_IsInside) == MH_OK) {
            if (WO_EnableHook((void*)kIsInside) == MH_OK) {
                g_insideInstalled = true;
                SamplingProfiler::RegisterSelfSymbol("FrustumInside_SSE2", (const void*)&Hooked_IsInside);
                Log("[FrustumAabb] ACTIVE on CFrustum::IsAABBInside (0x%08X)", (unsigned)kIsInside);
            }
        }
    }

    if (CheckPrologue8((void*)kIsPointVisible, kExp_IsPointVisible, "CFrustum::IsPointVisible")) {
        if (WineSafe_CreateHook((void*)kIsPointVisible, (void*)Hooked_IsPointVisible,
                                (void**)&orig_IsPointVisible) == MH_OK) {
            if (WO_EnableHook((void*)kIsPointVisible) == MH_OK) {
                g_pointInstalled = true;
                SamplingProfiler::RegisterSelfSymbol("FrustumPoint_SSE2", (const void*)&Hooked_IsPointVisible);
                Log("[FrustumAabb] ACTIVE on CFrustum::IsPointVisible (0x%08X)", (unsigned)kIsPointVisible);
            }
        }
    }

    if (CheckPrologue8((void*)kIsSphereVisible, kExp_IsSphereVisible, "CFrustum::IsSphereVisible")) {
        if (WineSafe_CreateHook((void*)kIsSphereVisible, (void*)Hooked_IsSphereVisible,
                                (void**)&orig_IsSphereVisible) == MH_OK) {
            if (WO_EnableHook((void*)kIsSphereVisible) == MH_OK) {
                g_sphereInstalled = true;
                SamplingProfiler::RegisterSelfSymbol("FrustumSphere_SSE2", (const void*)&Hooked_IsSphereVisible);
                Log("[FrustumAabb] ACTIVE on CFrustum::IsSphereVisible (0x%08X)", (unsigned)kIsSphereVisible);
            }
        }
    }

    if (CheckPrologue8((void*)kTranslate, kExp_Translate, "CFrustum::Translate")) {
        if (WineSafe_CreateHook((void*)kTranslate, (void*)Hooked_Translate,
                                (void**)&orig_Translate) == MH_OK) {
            if (WO_EnableHook((void*)kTranslate) == MH_OK) {
                g_transInstalled = true;
                SamplingProfiler::RegisterSelfSymbol("FrustumTranslate_SSE2", (const void*)&Hooked_Translate);
                Log("[FrustumAabb] ACTIVE on CFrustum::Translate (0x%08X), 560 bytes, "
                    "translates 10 points and updates 6 plane equations with double-precision SSE2.",
                    (unsigned)kTranslate);
            }
        }
    }

    g_abSubject = AbTest::IsSubject("FrustumAabb", &g_abSubject);
    g_benchSlot = SelfBench::Register("FrustumAabb");
    if (g_abSubject) {
        Log("[FrustumAabb] under A/B test: it alternates on and off in stints "
            "and AbTest reports the frame times either way. The correctness "
            "checks are unaffected and still retire it on a disagreement.");
    }

    g_installed = true;
    SamplingProfiler::RegisterSelfSymbol("FrustumAabb_SSE2", (const void*)&Hooked_IsVisible);
    Log("[FrustumAabb] ACTIVE on CFrustum::IsAABBVisible (0x%08X), 0.82%% of "
        "executing time in an uncapped tester session. Most of that function is "
        "not arithmetic: for each of three components of each of six planes it "
        "reads the plane's sign bit, indexes a two-entry pointer table with it, "
        "and loads a box corner through the result - eighteen sign tests and "
        "eighteen dependent loads per call. The sign bits are the blend mask in "
        "SSE2, so the table and the dependent loads go away. The threshold is "
        "read from the client at 0x%08X (%.17g), not written here as a literal. "
        "The function is pure, so both answers are compared for the first %lu "
        "calls and one in %lu after that.",
        (unsigned)kIsVisible, (unsigned)kThreshold, g_threshold,
        kVerifyFirst, kResampleMask + 1);
    return true;
}

void LogStats() {
    if (!Config::g_settings.OptFrustumAabb) return;
    if (!g_installed) { Log("[FrustumAabb] not installed - nothing measured"); return; }
    if (g_calls == 0) { Log("[FrustumAabb] installed but never called"); return; }

    const double calls   = Total(g_calls, g_callWraps);
    const double visible = Total(g_visible, g_visWraps);
    Log("[FrustumAabb] %.0f visibility tests%s, %.0f came back visible (%.1f%%), "
        "%lu verified against the client. Counts are lower bounds.",
        calls,
        g_dead ? " - RETIRED on a disagreement"
               : (g_armed ? "" : " - still verifying, the client still answers every one"),
        visible, calls > 0.0 ? 100.0 * visible / calls : 0.0, g_verified);
    // What the plane hint is actually worth on this client, rather than on the
    // model in the note above. Six would mean the hint never helps; the floor is
    // 0.636 * 6 + 0.364 * 1, about 4.18, if every cull were caught on the first
    // plane tried.
    const double planes = Total(g_planeEvals, g_planeWraps);
    const double culled = calls - visible;
    Log("[FrustumAabb]   %.2f plane(s) evaluated per test against six in the "
        "client's fixed order, and %lu of the %.0f culls were caught by the "
        "plane that culled the box before them (%.1f%%).",
        calls > 0.0 ? planes / calls : 0.0, g_hintCulled, culled,
        culled > 0.0 ? 100.0 * (double)g_hintCulled / culled : 0.0);
    if (visible > calls)
        Log("[Wrong] [FrustumAabb] more tests came back visible than were run. "
            "That cannot happen, and it means these two counters are no longer "
            "being counted at the same boundary.");

    if (g_insideInstalled && g_insideCalls > 0) {
        Log("[FrustumAabb] %lu inside tests, %lu verified against client%s",
            g_insideCalls, g_insideVerified,
            g_insideDead ? " - RETIRED on a disagreement" : (g_insideArmed ? "" : " - still verifying"));
    }

    if (g_pointInstalled && g_pointCalls > 0) {
        Log("[FrustumAabb] %lu point visibility tests, %lu verified against client%s",
            g_pointCalls, g_pointVerified,
            g_pointDead ? " - RETIRED on a disagreement" : (g_pointArmed ? "" : " - still verifying"));
    }

    if (g_sphereInstalled && g_sphereCalls > 0) {
        const double culled_pct = 100.0 * (double)g_sphereCulled / (double)g_sphereCalls;
        Log("[FrustumAabb] %lu sphere visibility tests, %lu culled (%.1f%%), %lu verified against client%s",
            g_sphereCalls, g_sphereCulled, culled_pct, g_sphereVerified,
            g_sphereDead ? " - RETIRED on a disagreement" : (g_sphereArmed ? "" : " - still verifying"));
    }

    if (g_transInstalled && g_transCalls > 0) {
        Log("[FrustumAabb] %lu translate calls, %lu verified against client%s",
            g_transCalls, g_transVerified,
            g_transDead ? " - RETIRED on a disagreement" : (g_transArmed ? "" : " - still verifying"));
    }
}

void Shutdown() {
    if (g_installed) MH_DisableHook((void*)kIsVisible);
    if (g_insideInstalled) MH_DisableHook((void*)kIsInside);
    if (g_pointInstalled) MH_DisableHook((void*)kIsPointVisible);
    if (g_sphereInstalled) MH_DisableHook((void*)kIsSphereVisible);
    if (g_transInstalled) MH_DisableHook((void*)kTranslate);
}

}  // namespace FrustumAabb
