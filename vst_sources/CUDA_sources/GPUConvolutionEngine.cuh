
#ifndef _GPUConvolutionEngine_H_

#define _GPUConvolutionEngine_H_
#include <cstring>
#include <stdlib.h> 
#include <stdio.h>
#include "cuda_runtime.h"
#include "GPUConvolutionUtils.cuh"
#include <device_launch_parameters.h>

class GPU_ConvolutionEngine {
	public:
	GPU_ConvolutionEngine();
	~GPU_ConvolutionEngine();


	void process(const float* in, const float* in2, const float* in3, const float* in4, float* out1, float* out2);

	void setSize(float size);

	void prepare(int buffersize, float size);

private:
	int h_index = 0;
	int h_numPartitions = 0;
	int bs = 0;
	int bs_float = 0;
	int conv_res_size = 0;
	int conv_res_float = 0;

	float* d_input_A = nullptr; 
	float* d_input_B = nullptr; 

	float* d_TimeDomain_A = nullptr;
	float* d_TimeDomain_B = nullptr;
	float* d_TimeDomain_C = nullptr;
	float* d_TimeDomain_D = nullptr;

	float* d_ConvolutionRes_A = nullptr;
	float* d_ConvolutionRes_B = nullptr;
	float* h_ConvolutionRes_A = nullptr;
	float* h_ConvolutionRes_B = nullptr;
	float* h_Overlap_A = nullptr;
	float* h_Overlap_B = nullptr;
	 
	dim3 dThreads;
	dim3 dBlocks;

	void clear() ;
	void cleanup();
	void launchEngines();
};

#endif // _GPUConvolutionEngine_H_