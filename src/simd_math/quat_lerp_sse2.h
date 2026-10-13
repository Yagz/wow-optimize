#pragma once
#ifndef WOW_OPT_QUAT_LERP_SSE2_H
#define WOW_OPT_QUAT_LERP_SSE2_H

// sub_982630: interpolate two quaternions and renormalise. Called for every
// animated bone of every model, every frame, from the M2 quaternion track. The
// client does it one component at a time on the x87 stack; this does all four
// at once.
namespace QuatLerpSse2 {

bool Init();
void LogStats();

// True once the vector path has been bit-identical to the client's for its
// whole learning phase and has not differed since (and the module is not an
// A/B subject). The quaternion track calls quat_lerp_core.h directly while
// it is.
extern bool g_usable;

} // namespace QuatLerpSse2

#endif
