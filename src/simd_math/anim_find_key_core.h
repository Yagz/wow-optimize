#pragma once
// ============================================================================
// anim_find_key_core.h
//
// The keyframe search of sub_8284D0, as one inline function. m2_anim_find_key
// hooks the client's routine with it, and the three track evaluators that are
// the routine's callers (quaternion, vector, scalar and colour) call it directly
// once the hook has proved it against the client. Calling it directly removes the
// trampoline jump, the detour's prologue and its counters from a call that runs
// about four billion times in a three hour session.
//
// There is one copy of the search so a fix cannot reach one caller and not the
// other.
// ============================================================================

#include <cstdint>

namespace M2AnimFindKeyCore {

struct SubTrack {
    uint32_t nKeys;
    const uint32_t* timestamps;
};

// this_ptr is the model (offset 0x70 holds the global sequence table), timing
// the animation timing sub-struct, track the track descriptor. hint is read as
// the starting guess and written with the key it settled on.
__forceinline void* Run(const void* this_ptr, const void* timing, const void* track,
                        uint32_t* hint, uint32_t* second, float* frac)
{
    const uint32_t time = *(const uint32_t*)timing;
    uint16_t anim_id = *(const uint16_t*)((const char*)timing + 4);

    const uint32_t nAnimations = *(const uint32_t*)((const char*)track + 4);
    if (nAnimations == 0) {
        *second = 0;
        *hint = 0;
        *frac = 0.0f;
        return frac;
    }

    const uint16_t global_seq = *(const uint16_t*)((const char*)track + 2);
    uint32_t anim_time = time;
    if (global_seq != 0xFFFF) {
        const uint32_t* const pGlobalSeqs = *(const uint32_t* const*)((const char*)this_ptr + 0x70);
        if (pGlobalSeqs) {
            anim_time = pGlobalSeqs[global_seq];
        }
        anim_id = 0;
    } else {
        if (anim_id >= nAnimations) {
            anim_id = 0;
        }
    }

    const SubTrack* sub_tracks = *(const SubTrack* const*)((const char*)track + 8);
    if (!sub_tracks) {
        *second = 0;
        *hint = 0;
        *frac = 0.0f;
        return frac;
    }

    const uint32_t nKeys = sub_tracks[anim_id].nKeys;
    if (nKeys <= 1) {
        *second = 0;
        *hint = 0;
        *frac = 0.0f;
        return frac;
    }

    const uint32_t* const timestamps = sub_tracks[anim_id].timestamps;
    if (!timestamps) {
        *second = 0;
        *hint = 0;
        *frac = 0.0f;
        return frac;
    }

    uint32_t guess = *hint;
    if (guess >= nKeys) {
        guess = 0;
    }

    const uint32_t t_guess = timestamps[guess];
    const uint32_t delta = anim_time - t_guess;

    if (delta < 500) {
        // Path 1: small positive delta (normal forward animation step)
        const uint32_t max_key = nKeys - 1;
        while (guess < max_key) {
            if (timestamps[guess + 1] > anim_time) {
                break;
            }
            ++guess;
        }
    } else if (delta >= 0xFFFFFE0Cu) {
        // Path 2: small negative delta (animation loop or small backward step)
        while (guess > 0) {
            if (timestamps[guess] <= anim_time) {
                break;
            }
            --guess;
        }
    } else {
        // Path 3: large jump
        if (anim_time < 500) {
            guess = 0;
            const uint32_t max_key = nKeys - 1;
            while (guess < max_key) {
                if (timestamps[guess + 1] > anim_time) {
                    break;
                }
                ++guess;
            }
        } else {
            // Binary search matching assembly logic
            uint32_t low = 0;
            uint32_t high = nKeys;
            guess = 0;
            while (low < high) {
                const uint32_t mid = (low + high) >> 1;
                if (anim_time >= timestamps[mid]) {
                    low = mid + 1;
                    if (low >= nKeys || anim_time < timestamps[mid + 1]) {
                        guess = mid;
                        break;
                    }
                } else {
                    high = (mid > 0) ? (mid - 1) : 0;
                }
                guess = low;
            }
        }
    }

    if (guess + 1 >= nKeys) {
        *hint = guess;
        *second = guess;
        *frac = 0.0f;
    } else {
        *hint = guess;
        *second = guess + 1;
        const uint32_t t0 = timestamps[guess];
        const uint32_t t1 = timestamps[guess + 1];
        const uint32_t dt = t1 - t0;
        if (dt > 0) {
            *frac = (float)((double)(anim_time - t0) / (double)dt);
        } else {
            *frac = 0.0f;
        }
    }

    return frac;
}

}  // namespace M2AnimFindKeyCore
