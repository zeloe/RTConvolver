
#include "GPUConvEngine1024.cuh"
// Define the constant memory array
 __constant__ int SIZES_1024[3];
 __constant__ float INPUT_1024[1024];
 __constant__ float INPUT2_1024[1024];

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
	const unsigned int partition_idx = blockIdx.x;
	const unsigned int copy_idx = threadIdx.x;
	extern __shared__ float partArray[];

	// Declare pointers to the shared memory partitions
	float* arr1 = &partArray[0];
	float* arr2 = &partArray[SIZES_1024[0]];
	float* tempResult = &partArray[SIZES_1024[0] * 2];

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
		#pragma unroll
		for (int i = 0; i < SIZES_1024[1]; i++) {
			atomicAdd(&Result[i], tempResult[i]);
		}


	}

}


GPUConvEngine_1024::GPUConvEngine_1024() {
	cudaStreamCreate(&stream);
	bs = maxBufferSize;
	dThreads.x = maxBufferSize;
	sizeMax = (((48000 * 6) / bs) + 1) * bs;
	h_convResSize = bs * 2 - 1;
	floatSizeRes = h_convResSize * sizeof(float);
	(cudaMalloc((void**)&d_ConvolutionResL, floatSizeRes));
	(cudaMalloc((void**)&d_ConvolutionResR, floatSizeRes));

	SHMEM = 4 * sizeof(float) * bs ;
 
	bs_float = bs * sizeof(float);
	 

	h_ConvolutionResL = (float*)calloc(h_convResSize, sizeof(float));
	h_ConvolutionResR = (float*)calloc(h_convResSize, sizeof(float));
	h_OverlapL = (float*)calloc(bs, sizeof(float));
	h_OverlapR = (float*)calloc(bs, sizeof(float));
	h_index = 0;
	 



	h_numPartitions = sizeMax / bs;
	h_paddedSize = h_numPartitions * bs;
	dBlocks.x = (h_numPartitions);
	cpu_sizes = (int*)calloc(3, sizeof(int));
	cpu_sizes[0] = bs;
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
	int floatSizeResMax = maxBufferSize * sizeof(float);
	(cudaMemset(d_ConvolutionResL, 0, floatSizeResMax));
	(cudaMemset(d_ConvolutionResR, 0, floatSizeResMax));
	(cudaMemset(INPUT_1024, 0, maxBufferSize * sizeof(float)));
	(cudaMemset(INPUT2_1024, 0, maxBufferSize * sizeof(float)));
	(cudaMemset(d_IR_paddedL, 0, h_paddedSize * sizeof(float)));
	(cudaMemset(d_IR_paddedR, 0, h_paddedSize * sizeof(float)));
	(cudaMemset(d_TimeDomain_paddedL, 0, h_paddedSize * sizeof(float)));
	(cudaMemset(d_TimeDomain_paddedR, 0, h_paddedSize * sizeof(float)));
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

void GPUConvEngine_1024::prepare(int sampleRate) {
	
	int temp_h_paddedSize = ((sampleRate * 6 / bs) + 1) * bs; 
	h_numPartitions = temp_h_paddedSize / bs;
	
	dBlocks.x = (h_numPartitions);

	threadsPerBlock.x = bs;
	numBlocks.x = (temp_h_paddedSize + threadsPerBlock.x - 1) / threadsPerBlock.x;
	 
	
	cpu_sizes[0] = bs;
	cpu_sizes[1] = h_convResSize;
	cpu_sizes[2] = h_numPartitions - 1;
	cudaMemcpyToSymbol(SIZES_1024, cpu_sizes, 3 * sizeof(int));

	clear();

}



 

void  GPUConvEngine_1024::process(const float* in, const float* in2, const float* in3, const float* in4, float* out1, float* out2) {
	
	 
	//copy content and transfer
	int indexBs = h_index * bs;
	cudaMemcpyToSymbolAsync(INPUT_1024, in, bs_float, 0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(INPUT2_1024,in2, bs_float, 0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_IR_paddedL + indexBs, in3, bs_float, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_IR_paddedR + indexBs, in4, bs_float, cudaMemcpyHostToDevice, stream);



	//launch the convolution Engine
	launchEngine();



	for (int i = 0; i < bs; i += 4) {
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
	std::memcpy(h_OverlapL, &h_ConvolutionResL[bs - 1], bs_float);
	std::memcpy(h_OverlapR, &h_ConvolutionResR[bs - 1], bs_float);










	 
}


void  GPUConvEngine_1024::launchEngine() {

	shiftAndInsertKernel_1024 << <numBlocks, threadsPerBlock,0,stream >> > (d_TimeDomain_paddedL);
	shiftAndInsertKernel2_1024 << <numBlocks, threadsPerBlock, 0, stream >> > (d_TimeDomain_paddedR);
	shared_partitioned_convolution_1024 << <dBlocks,dThreads , SHMEM, stream >> > (d_ConvolutionResL, d_TimeDomain_paddedL, d_IR_paddedL);
	shared_partitioned_convolution_1024 << <dBlocks, dThreads, SHMEM, stream >> > (d_ConvolutionResR, d_TimeDomain_paddedR, d_IR_paddedR);
	cudaMemcpyAsync(h_ConvolutionResL, d_ConvolutionResL, floatSizeRes, cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(h_ConvolutionResR, d_ConvolutionResR, floatSizeRes, cudaMemcpyDeviceToHost, stream);
	cudaMemsetAsync(d_ConvolutionResL, 0, floatSizeRes, stream);
	cudaMemsetAsync(d_ConvolutionResR, 0, floatSizeRes, stream);

	cudaStreamSynchronize(stream);
	//update index

	h_index = (h_index + 1) % (h_numPartitions);
	
}
