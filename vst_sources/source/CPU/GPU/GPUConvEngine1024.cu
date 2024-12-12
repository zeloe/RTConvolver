
#include "GPUConvEngine1024.cuh"
// Define the constant memory array
__constant__ int SIZES_1024[3];
__constant__ float INPUT_1024[1024];
__constant__ float INPUT2_1024[1024];
__shared__ float partArray_1024[1024 * 4];
__global__ void   shiftAndInsertKernel_1024(float* __restrict__ delayBuffer) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// Insert new elements at the beginning of the delay buffer
	if (tid < SIZES_1024[0]) {
		delayBuffer[tid] = INPUT_1024[tid];
	}

	delayBuffer[tid + SIZES_1024[0]] = delayBuffer[tid]; 
}


__global__ void   shiftAndInsertKernel2_1024(float* __restrict__ delayBuffer) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// Insert new elements at the beginning of the delay buffer
	if (tid < SIZES_1024[0]) {
		delayBuffer[tid] = INPUT2_1024[tid];
	}

	delayBuffer[tid + SIZES_1024[0]] = delayBuffer[tid];


}









 
__global__ void  shared_partitioned_convolution_1024(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
 
	const unsigned int copy_idx = threadIdx.x;
	

	// Declare pointers to the shared memory partitions
	float* arr1 = &partArray_1024[0];
	float* arr2 = &partArray_1024[SIZES_1024[0]];
	float* tempResult = &partArray_1024[SIZES_1024[0] * 2];
	// Load data into shared memory
	tempResult[copy_idx] = 0.f;
	tempResult[SIZES_1024[0] + copy_idx] = 0.f;

	 
	arr1[copy_idx] = Dry[thread_idx];
	arr2[copy_idx] = Imp[thread_idx];

	// Shared memory to accumulate results before writing them to global memory
	// Convolution operation (reduction into shared memory)
	#pragma unroll
	for (int i = 0; i < SIZES_1024[0]; i++) {
		int inv = (i - copy_idx) % SIZES_1024[0];
		tempResult[i + inv] += arr1[i] * arr2[inv];
	}

	__syncthreads();  // Ensure all threads in the block have finished processing


	// Write the accumulated result to global memory (only for the first thread)
	if (copy_idx == 0) {
		// Write the first part of the result (up to SIZES[0] * 2 - 1)
		#pragma unroll
		for (int i = 0; i < SIZES_1024[1]; i++) {
			atomicAdd(&Result[i], tempResult[i]);
		}
	}

}


GPUConvEngine_1024::GPUConvEngine_1024() {
	cudaStreamCreate(&stream);
 
	dThreads.x = maxBufferSize;
	h_convResSize = maxBufferSize * 2 - 1;
	sizeMax = ((48000 * 6 / maxBufferSize) + 1) * maxBufferSize;
	 
	
	
	floatSizeRes = h_convResSize * sizeof(float);
	(cudaMalloc((void**)&d_ConvolutionResL, floatSizeRes));
	(cudaMalloc((void**)&d_ConvolutionResR, floatSizeRes));

 
 
	bs_float = maxBufferSize * sizeof(float);
	 

	h_ConvolutionResL = (float*)calloc(h_convResSize +1, sizeof(float));
	h_ConvolutionResR = (float*)calloc(h_convResSize + 1, sizeof(float));
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
	cudaMemcpyToSymbol(SIZES_1024, cpu_sizes, 3 * sizeof(int));


	(cudaMalloc((void**)&d_IR_paddedL, h_paddedSize * sizeof(float)));
	(cudaMalloc((void**)&d_IR_paddedR, h_paddedSize * sizeof(float)));

	(cudaMalloc((void**)&d_TimeDomain_paddedL, h_paddedSize * sizeof(float)));
	(cudaMalloc((void**)&d_TimeDomain_paddedR, h_paddedSize * sizeof(float)));
	
	clear(); 
	
}
void GPUConvEngine_1024::clear() { 
	int floatSizeResMax = 2 * maxBufferSize * sizeof(float);
	(cudaMemset(d_ConvolutionResL, 0, floatSizeRes));
	(cudaMemset(d_ConvolutionResR, 0, floatSizeRes));
	(cudaMemset(INPUT_1024, 0, maxBufferSize * sizeof(float)));
	(cudaMemset(INPUT2_1024, 0, maxBufferSize * sizeof(float)));
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


GPUConvEngine_1024::~GPUConvEngine_1024() {
	cleanup();
	// Free Stream 
	cudaStreamDestroy(stream);
}

void GPUConvEngine_1024::cleanup() {
	
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
	free(cpu_sizes);
 



}


void GPUConvEngine_1024::checkCudaError(cudaError_t err, const char* errMsg) {
	if (err != cudaSuccess) {
		printf("CUDA Error (%s): %s\n", errMsg, cudaGetErrorString(err));
	}
}
void GPUConvEngine_1024::prepare(float size) {
	 

	cudaStreamSynchronize(stream);

	// Ensure proper padding
	int temp_h_paddedSize = ((size / maxBufferSize) + 1) * maxBufferSize;
	 
	// Update dBlocks and other parameters
	dBlocks.x = temp_h_paddedSize / maxBufferSize;

	// Update sizes array
	cpu_sizes[2] = temp_h_paddedSize;

	// Copy updated sizes to device
	 cudaMemcpyToSymbol(SIZES_1024, cpu_sizes, sizeof(int) * 3);
}




 

void  GPUConvEngine_1024::process(const float* in, const float* in2, const float* in3, const float* in4, float* out1, float* out2) {
	
	 
	//copy content and transfer
	int indexBs = h_index * maxBufferSize;
	cudaMemcpyToSymbolAsync(INPUT_1024, in, bs_float, 0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(INPUT2_1024,in2, bs_float, 0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_IR_paddedL + indexBs, in3, bs_float, cudaMemcpyHostToDevice, stream);
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

void GPUConvEngine_1024::launchEngine() {
	// Kernel 1: shiftAndInsertKernel_1024
	shiftAndInsertKernel_1024 << <dBlocks, dThreads, 0, stream >> > (d_TimeDomain_paddedL);
	 

	// Kernel 2: shiftAndInsertKernel2_1024
	shiftAndInsertKernel2_1024 << <dBlocks, dThreads, 0, stream >> > (d_TimeDomain_paddedR);
	 

	// Kernel 3: shared_partitioned_convolution_1024 for Left channel
	 
  	shared_partitioned_convolution_1024<<<dBlocks, dThreads, 0, stream >>> (d_ConvolutionResL, d_TimeDomain_paddedL, d_IR_paddedL);

	// Kernel 4: shared_partitioned_convolution_1024 for Right channel
	 
	shared_partitioned_convolution_1024 << <dBlocks, dThreads, 0, stream >> > (d_ConvolutionResR, d_TimeDomain_paddedR, d_IR_paddedR);

	// Copy results back to host for debugging purposes
	cudaMemcpyAsync(h_ConvolutionResL, d_ConvolutionResL, floatSizeRes, cudaMemcpyDeviceToHost, stream);
	 

	cudaMemcpyAsync(h_ConvolutionResR, d_ConvolutionResR, floatSizeRes, cudaMemcpyDeviceToHost, stream);
	 

	 
	 

	// Synchronize the stream to ensure all operations are complete
	cudaStreamSynchronize(stream);
	 
	cudaMemsetAsync(d_ConvolutionResL, 0, floatSizeRes, stream);
	cudaMemsetAsync(d_ConvolutionResR, 0, floatSizeRes, stream);
	// Update the index for overlap handling
	h_index = (h_index + 1) % h_numPartitions;
}
