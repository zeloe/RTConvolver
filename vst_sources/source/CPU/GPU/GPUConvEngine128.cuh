
#ifndef _GPUConvEngine128_H_

#define _GPUConvEngine128_H_
#include <stdlib.h> 

#include "cuda_runtime.h"
#include <device_launch_parameters.h>

#include <stdio.h>
#include <omp.h>
#include <immintrin.h> // For SSE intrinsics
#include <cstring>

__global__ void shared_partitioned_convolution_128(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp);

__global__ void shared_partitioned_convolution_128_2(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp);

__global__ void shared_partitioned_convolution_128_3(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp);

__global__ void shared_partitioned_convolution_128_4(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp);

__global__ void shared_partitioned_convolution_128_5(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp);

__global__ void shared_partitioned_convolution_128_6(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp);

__global__ void shared_partitioned_convolution_128_7(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp);

__global__ void shared_partitioned_convolution_128_8(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp);


__global__ void  shiftAndInsertKernel_128(float* __restrict__ delayBuffer);
__global__ void  shiftAndInsertKernel2_128(float* __restrict__ delayBuffer);
class GPUConvEngine_128 {
public:
	GPUConvEngine_128();
	~GPUConvEngine_128();
	
	void  process(const float* in, const float* in2, const float* in3, const float* in4, float* out1, float* out2);
	void  prepare(float size);
	void clear();
private:
	
	void cleanup();
	void   launchEngine();
	void checkCudaError(cudaError_t err, const char* errMsg);
	int* cpu_sizes = nullptr;
	int* cpu_offsets = nullptr;
	int sizeMax = 0;
	const int maxBufferSize = 128;
	const int numKernels = 8;
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
	   
	dim3 dThreads;
	dim3 dBlocks;
	dim3 convBlocks;
	cudaStream_t stream;
	 
};



#endif