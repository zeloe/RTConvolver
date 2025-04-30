// GPUConvUtils.cuh
#ifndef _GPUConvUtils_H_
#define _GPUConvUtils_H_
#include <stdio.h>
#include <stdlib.h> 
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

const int MAX_BUFFER_SIZE = 1024;
const int MAX_BUFFER_SIZE_FLOAT = MAX_BUFFER_SIZE * sizeof(float);

const int MAX_CONVOLUTION_SIZE = int((((48000.f * 6.f) / (float)MAX_BUFFER_SIZE) + 1.f) * MAX_BUFFER_SIZE); // approx 6 seconds of audio
const int MAX_CONVOLUTION_SIZE_FLOAT = MAX_CONVOLUTION_SIZE * sizeof(float);
const int CONV_RES_SIZE = MAX_BUFFER_SIZE * 2;
const int CONV_RES_SIZE_FLOAT = MAX_BUFFER_SIZE * 2 * sizeof(float);


const int SHARED_ARRAY_SIZE = MAX_BUFFER_SIZE * 4;

namespace GPUConv {

	void initSizes();
	void changeSizes(int bs);

	// Error checking helper function
	cudaError_t checkCudaError(cudaError_t error);

	// Utility functions for memory management
	cudaError_t safeCudaFree(void* ptr);

	void safeHostFree(void* ptr);

	// Kernel declarations

	extern "C" __global__ void shiftAndInsertKernel(float* __restrict__ delayBuffer, const float* __restrict__ input);

	extern "C" __global__ void sharedPartitionedConvolution(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp);

}

#endif // _GPUConvUtils_H_