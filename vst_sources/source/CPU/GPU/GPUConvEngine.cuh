
#ifndef _GPUConvEngine_H_

#define _GPUConvEngine_H_

#include <stdlib.h> 

#include "cuda_runtime.h"
#include <device_launch_parameters.h>

#include <stdio.h>
#include <cstring>

// this is an empty class in case blocksize > 1024 
class GPUConvEngine {
public:
	GPUConvEngine();
	~GPUConvEngine();
	virtual void prepare(float sampleRate);
	void process(const float* in, const float* in2, const float* in3, const float* in4, float* out1, float* out2);
	 
private:
	void clear();
	void cleanup();
	void   launchEngine();
	void checkCudaError(cudaError_t err, const char* errMsg);
	
	int sizeMax = 0;
	const int maxBufferSize = 1024;
	int bs = 0;
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
	dim3 threadsPerBlock;
	dim3 numBlocks;
	size_t SHMEM = 0;
	cudaStream_t stream;
	int threadsPerBlockZero = 0;
	int numBlocksZero = 0;

};



#endif