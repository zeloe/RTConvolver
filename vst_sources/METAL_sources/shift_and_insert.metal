#include <metal_stdlib>
using namespace metal;

kernel void shiftAndInsertKernel(
    device float* delayBuffer [[buffer(0)]],
    device const float* inputBuffer [[buffer(1)]],
    constant uint* SIZES [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]], uint threadid [[thread_position_in_threadgroup]], uint blockDim [[threads_per_threadgroup]], uint blockid[[threadgroup_position_in_grid]])
{
    uint thread_idx  = blockid*blockDim+threadid;
   
    // Insert new elements at the beginning of the delay buffer
    if (thread_idx < SIZES[0]) {
        delayBuffer[thread_idx] = inputBuffer[thread_idx];
    }
    
    // Shift the old values
    delayBuffer[thread_idx + SIZES[0]] = delayBuffer[thread_idx];
}
