
#ifndef _GPUConvEngine1024_H_

#define _GPUConvEngine1024_H_
 
#include <stdlib.h> 

#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <thread>
#include <atomic>
#include <stdio.h>
#include <omp.h>
#include <immintrin.h> // For SSE intrinsics
#include <cstring>
#include "GPUConvEngine.cuh"
__global__ void shiftAndInsertKernel_1024(float* __restrict__ delayBuffer);
__global__ void shiftAndInsertKernel2_1024(float* __restrict__ delayBuffer);
__global__ void shared_partitioned_convolution_1024(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp);
 
class GPUConvEngine_1024 {
public:
	GPUConvEngine_1024();
	~GPUConvEngine_1024();
	void  process(const float* in, const float* in2, const float* in3, const float* in4, float* out1, float* out2);

	void  prepare(float size);

	void clear();

private:
	dim3 d_Threads;
	dim3 d_Blocks;
 
	
	void cleanup();
	void   launchEngine();
	void checkCudaError(cudaError_t err, const char* errMsg);
	
	int sizeMax = 0;
	const int maxBufferSize = 1024;
	 
	int bs_float = 0;
	int h_numPartitions = 0;
	int h_paddedSize = 0;
	int h_convResSize = 0;
	int h_index = 0;
	int floatSizeRes = 0;
	
	float* d_IR_paddedL = nullptr;
	float* d_TimeDomain_paddedL = nullptr;
	float* d_IR_paddedR = nullptr;
	float* d_TimeDomain_paddedR = nullptr;
	
	float* d_ConvolutionResL = nullptr;
	float* d_ConvolutionResR = nullptr;
	float* h_ConvolutionResL = nullptr;
	float* h_ConvolutionResR = nullptr;
	float* h_OverlapL = nullptr;
	float* h_OverlapR = nullptr;
	int* cpu_sizes = nullptr;
	dim3 dThreads;
	dim3 dBlocks;
	  
	cudaStream_t stream;
};



#endif