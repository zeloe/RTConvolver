
#include "GPUConvEngine512.cuh" 

__constant__ int SIZES_512[3];
__constant__ float INPUT_512[512];
__constant__ float INPUT2_512[512];
__shared__ float partArray_512_1[512 * 4];
__shared__ float partArray_512_2[512 * 4];
__constant__ int OFFSETS_512[1];
__global__ void  shared_partitioned_convolution_512(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int partition_idx = blockIdx.x;
	const unsigned int copy_idx = threadIdx.x;
 

	// Declare pointers to the shared memory partitions
	float* arr1 = &partArray_512_1[0];
	float* arr2 = &partArray_512_1[SIZES_512[0]];
	float* tempResult = &partArray_512_1[SIZES_512[0] * 2];
	// Load data into shared memory
	tempResult[copy_idx] = 0.f;
	tempResult[SIZES_512[0] + copy_idx] = 0.f;


	arr1[copy_idx] = Dry[thread_idx];
	arr2[copy_idx] = Imp[thread_idx];

	// Shared memory to accumulate results before writing them to global memory
	// Convolution operation (reduction into shared memory)
#pragma unroll
	for (int i = 0; i < SIZES_512[0]; i++) {
		int inv = (i - copy_idx) % SIZES_512[0];
		tempResult[i + inv] += arr1[i] * arr2[inv];
	}

	__syncthreads();  // Ensure all threads in the block have finished processing


	// Write the accumulated result to global memory (only for the first thread)
	if (copy_idx == 0) {
		// Write the first part of the result (up to SIZES[0] * 2 - 1)
#pragma unroll
		for (int i = 0; i < SIZES_512[1]; i++) {
			atomicAdd(&Result[i], tempResult[i]);
		}
	}

}


__global__ void  shared_partitioned_convolution_512_2(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int partition_idx = blockIdx.x;
	const unsigned int copy_idx = threadIdx.x;


	// Declare pointers to the shared memory partitions
	float* arr1 = &partArray_512_2[0];
	float* arr2 = &partArray_512_2[SIZES_512[0]];
	float* tempResult = &partArray_512_2[SIZES_512[0] * 2];
	// Load data into shared memory
	tempResult[copy_idx] = 0.f;
	tempResult[SIZES_512[0] + copy_idx] = 0.f;


	arr1[copy_idx] = Dry[thread_idx + OFFSETS_512[0]];
	arr2[copy_idx] = Imp[thread_idx + OFFSETS_512[0]];

	// Shared memory to accumulate results before writing them to global memory
	// Convolution operation (reduction into shared memory)
#pragma unroll
	for (int i = 0; i < SIZES_512[0]; i++) {
		int inv = (i - copy_idx) % SIZES_512[0];
		tempResult[i + inv] += arr1[i] * arr2[inv];
	}

	__syncthreads();  // Ensure all threads in the block have finished processing


	// Write the accumulated result to global memory (only for the first thread)
	if (copy_idx == 0) {
		// Write the first part of the result (up to SIZES[0] * 2 - 1)
#pragma unroll
		for (int i = 0; i < SIZES_512[1]; i++) {
			atomicAdd(&Result[i], tempResult[i]);
		}
	}

}
__global__ void  shiftAndInsertKernel_512(float* __restrict__ delayBuffer) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// Insert new elements at the beginning of the delay buffer
	if (tid < SIZES_512[0]) {
		delayBuffer[tid] = INPUT_512[tid];
	}

	delayBuffer[tid + SIZES_512[0]] = delayBuffer[tid];
}


__global__ void  shiftAndInsertKernel2_512(float* __restrict__ delayBuffer) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// Insert new elements at the beginning of the delay buffer
	if (tid < SIZES_512[0]) {
		delayBuffer[tid] = INPUT2_512[tid];
	}

	delayBuffer[tid + SIZES_512[0]] = delayBuffer[tid];


}


GPUConvEngine_512::GPUConvEngine_512() {
	cudaStreamCreate(&stream);
	 
	dThreads.x = maxBufferSize;
	convThreads.x = maxBufferSize;
	sizeMax = (((48000 * 6) / maxBufferSize) + 1) * maxBufferSize;
	h_convResSize = maxBufferSize * 2;
	floatSizeRes = h_convResSize * sizeof(float);
	(cudaMalloc((void**)&d_ConvolutionResL, floatSizeRes));
	(cudaMalloc((void**)&d_ConvolutionResR, floatSizeRes));

	SHMEM = 4 * sizeof(float) * maxBufferSize;

	bs_float = maxBufferSize * sizeof(float);


	h_ConvolutionResL = (float*)calloc(h_convResSize, sizeof(float));
	h_ConvolutionResR = (float*)calloc(h_convResSize, sizeof(float));
	h_OverlapL = (float*)calloc(maxBufferSize, sizeof(float));
	h_OverlapR = (float*)calloc(maxBufferSize, sizeof(float));
	h_index = 0;




	h_numPartitions = sizeMax / maxBufferSize;
	h_paddedSize = h_numPartitions * maxBufferSize;
	dBlocks.x = (h_numPartitions);
	cpu_sizes = (int*)calloc(3, sizeof(int));
	cpu_sizes[0] = maxBufferSize;
	cpu_sizes[1] = h_convResSize;
	cpu_sizes[2] = h_numPartitions;
	cudaMemcpyToSymbol(SIZES_512, cpu_sizes, 3 * sizeof(int));


	(cudaMalloc((void**)&d_IR_paddedL, h_paddedSize * sizeof(float)));
	(cudaMalloc((void**)&d_IR_paddedR, h_paddedSize * sizeof(float)));

	(cudaMalloc((void**)&d_TimeDomain_paddedL, h_paddedSize * sizeof(float)));
	(cudaMalloc((void**)&d_TimeDomain_paddedR, h_paddedSize * sizeof(float)));

	clear();
	 
}
void GPUConvEngine_512::clear() {
	(cudaMemset(d_ConvolutionResL, 0, floatSizeRes));
	(cudaMemset(d_ConvolutionResR, 0, floatSizeRes));
	(cudaMemset(INPUT_512, 0, maxBufferSize * sizeof(float)));
	(cudaMemset(INPUT2_512, 0, maxBufferSize * sizeof(float)));
	(cudaMemset(d_IR_paddedL, 0, h_paddedSize * sizeof(float)));
	(cudaMemset(d_IR_paddedR, 0, h_paddedSize * sizeof(float)));
	(cudaMemset(d_TimeDomain_paddedL, 0, h_paddedSize * sizeof(float)));
	(cudaMemset(d_TimeDomain_paddedR, 0, h_paddedSize * sizeof(float)));
	memset(h_ConvolutionResL, 0.f, floatSizeRes);
	memset(h_ConvolutionResR, 0.f, floatSizeRes);
	memset(h_OverlapL, 0.f, bs_float);
	memset(h_OverlapR, 0.f, bs_float);

	h_index = 0;

}


GPUConvEngine_512::~GPUConvEngine_512() {
	 
	cleanup();
	// Free Stream 
	cudaStreamDestroy(stream);
}

void GPUConvEngine_512::cleanup() {
	
	cudaFree(d_ConvolutionResL);
	cudaFree(d_ConvolutionResR);
	 
	cudaFree(d_IR_paddedL);
	cudaFree(d_IR_paddedR);
	cudaFree(d_TimeDomain_paddedL);
	cudaFree(d_TimeDomain_paddedR);

	// Free CPU memory
	free(h_ConvolutionResL);
	free(h_ConvolutionResR);
	free(h_OverlapL);
	free(h_OverlapR);

 



}


void GPUConvEngine_512::checkCudaError(cudaError_t err, const char* errMsg) {
	if (err != cudaSuccess) {
		printf("CUDA Error (%s): %s\n", errMsg, cudaGetErrorString(err));
	}
}

void GPUConvEngine_512::prepare(float size) {


	cudaStreamSynchronize(stream);
	// Ensure proper padding
	int temp_h_paddedSize = (((size) / maxBufferSize) / numKernels + 1) * maxBufferSize * numKernels;
	int temp_numPartitions = temp_h_paddedSize / maxBufferSize / numKernels;
	int offset = temp_h_paddedSize / numKernels;
	cudaMemcpyToSymbol(OFFSETS_512, &offset, sizeof(int));
 	convBlocks.x = temp_numPartitions;

	h_numPartitions = (((size) / maxBufferSize) + 1);
	// Update dBlocks and other parameters
	dBlocks.x = h_numPartitions;

	// Update sizes array
	cpu_sizes[2] = h_numPartitions;

	// Copy updated sizes to device
	cudaMemcpyToSymbol(SIZES_512, cpu_sizes, sizeof(int) * 3);
	
}

 



void  GPUConvEngine_512::process(const float* in, const float* in2, const float* in3, const float* in4, float* out1, float* out2) {
 
	//copy content and transfer
	int indexBs = h_index * maxBufferSize;
	cudaMemcpyToSymbolAsync(INPUT_512, in, bs_float,0, cudaMemcpyHostToDevice,stream);
	cudaMemcpyToSymbolAsync(INPUT2_512, in2, bs_float,0, cudaMemcpyHostToDevice,stream);
	cudaMemcpyAsync(d_IR_paddedL + indexBs, in3, bs_float, cudaMemcpyHostToDevice , stream);
	cudaMemcpyAsync(d_IR_paddedR + indexBs, in4, bs_float, cudaMemcpyHostToDevice, stream);
	


	//launch the convolution Engine
	launchEngine();

	for (int i = 0; i < maxBufferSize; i += 4) {
		
	 
		// Load 4 floats from h_ConvolutionResL and h_OverlapL
		__m128 resL = _mm_loadu_ps(&h_ConvolutionResL[i]);
		 
		 
		__m128 overlapL = _mm_loadu_ps(&h_OverlapL[i]);

		// Perform (resL + overlapL) 
		__m128 resultL = _mm_add_ps(resL, overlapL);
		_mm_storeu_ps(&out1[i], resultL); // Store the result in out1

		// Load 4 floats from h_ConvolutionResR and h_OverlapR
		__m128 resR = _mm_loadu_ps(&h_ConvolutionResR[i]);
		__m128 overlapR = _mm_loadu_ps(&h_OverlapR[i]);
	 
		 
		// Perform (resR + overlapR)
		__m128 resultR = _mm_add_ps(resR, overlapR);
		_mm_storeu_ps(&out2[i], resultR); // Store the result in out2

	 
	}
	
	// Copy the last `bs` elements as overlap values for the next block
	std::memcpy(h_OverlapL, &h_ConvolutionResL[maxBufferSize - 1], bs_float);
 	std::memcpy(h_OverlapR, &h_ConvolutionResR[maxBufferSize - 1], bs_float);
	 
}




void  GPUConvEngine_512::launchEngine() {
	
	shiftAndInsertKernel_512 << <dBlocks, dThreads,0,stream >> > (d_TimeDomain_paddedL);
	shiftAndInsertKernel2_512 << <dBlocks, dThreads, 0, stream >> > (d_TimeDomain_paddedR);
	shared_partitioned_convolution_512 << <convBlocks,dThreads , 0, stream >> > (d_ConvolutionResL, d_TimeDomain_paddedL, d_IR_paddedL);
	shared_partitioned_convolution_512 << <convBlocks, dThreads, 0, stream >> > (d_ConvolutionResR, d_TimeDomain_paddedR, d_IR_paddedR);
	shared_partitioned_convolution_512_2 << <convBlocks, dThreads, 0, stream >> > (d_ConvolutionResL, d_TimeDomain_paddedL, d_IR_paddedL);
	shared_partitioned_convolution_512_2 << <convBlocks, dThreads, 0, stream >> > (d_ConvolutionResR, d_TimeDomain_paddedR, d_IR_paddedR);
	cudaMemcpyAsync(h_ConvolutionResL, d_ConvolutionResL, floatSizeRes, cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(h_ConvolutionResR, d_ConvolutionResR, floatSizeRes, cudaMemcpyDeviceToHost, stream);
	cudaMemsetAsync(d_ConvolutionResL, 0, floatSizeRes, stream);
	cudaMemsetAsync(d_ConvolutionResR, 0, floatSizeRes, stream);

	cudaStreamSynchronize(stream);
	//update index

	h_index = (h_index + 1) % (h_numPartitions);
	
}
