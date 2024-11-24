
#include "GPUConvEngine512.cuh" 

__constant__ int SIZES_512[3];
__constant__ float INPUT_512[512];
__constant__ float INPUT2_512[512];

__global__ void shared_partitioned_convolution_512(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int copy_idx = threadIdx.x;
	extern __shared__ float partArray[];

	// Declare pointers to the shared memory partitions
	float* arr1 = &partArray[0];
	float* arr2 = &partArray[SIZES_512[0]];
	float* tempResult = &partArray[SIZES_512[0] * 2];
	// Load data into shared memory
	tempResult[copy_idx] = 0.f;
	tempResult[copy_idx + SIZES_512[0]] = 0.f;
	arr1[copy_idx] = Dry[thread_idx];
	arr2[copy_idx] = Imp[thread_idx];

	 
	__syncthreads();  // Ensure all threads in the block have finished processing
	#pragma unroll
	for (int i = 0; i < SIZES_512[0]; i++) {
		int inv = (i + SIZES_512[0] - copy_idx) % SIZES_512[0];
		tempResult[i + inv] += arr1[i] * arr2[inv];
	}
	__syncthreads();  // Ensure all threads in the block have finished processing

	// Write the accumulated result to global memory (only for the first thread)
	if (copy_idx == 0) {
		// Write the result into global memory
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
	bs = maxBufferSize;
	sizeMax = (((96000) / maxNumberOfTheads) + 1) * maxNumberOfTheads;
	h_convResSize = bs * 2 - 1;
	floatSizeRes = h_convResSize * sizeof(float);
	(cudaMalloc((void**)&d_ConvolutionResL, floatSizeRes));
	(cudaMalloc((void**)&d_ConvolutionResR, floatSizeRes));

	SHMEM = 4 * maxNumberOfTheads * sizeof(float);
 
	bs_float = bs * sizeof(float);
	cpu_sizes = (int*)calloc(2, sizeof(int));

	h_ConvolutionResL = (float*)calloc(h_convResSize, sizeof(float));
	h_ConvolutionResR = (float*)calloc(h_convResSize, sizeof(float));
	h_OverlapL = (float*)calloc(bs, sizeof(float));
	h_OverlapR = (float*)calloc(bs, sizeof(float));
	h_index = 0; 
	cpu_sizes[0] = bs;
	cpu_sizes[1] = h_convResSize;


	h_numPartitions = sizeMax / bs;
	h_paddedSize = h_numPartitions * bs;

	 
	cudaMemcpyToSymbol(SIZES_512, cpu_sizes, 2 * sizeof(int));


	(cudaMalloc((void**)&d_IR_paddedL, h_paddedSize * sizeof(float)));
	(cudaMalloc((void**)&d_IR_paddedR, h_paddedSize * sizeof(float)));

	(cudaMalloc((void**)&d_TimeDomain_paddedL, h_paddedSize * sizeof(float)));
	(cudaMalloc((void**)&d_TimeDomain_paddedR, h_paddedSize * sizeof(float)));
	dThreads.x = maxNumberOfTheads;
	bs_float = bs * sizeof(float);
	h_convResSize = bs * 2 - 1;
	floatSizeRes = h_convResSize * sizeof(float);
	clear();
	free(cpu_sizes);
	 
}
void GPUConvEngine_512::clear() {
	int floatSizeResMax = maxBufferSize * sizeof(float);
	(cudaMemset(d_ConvolutionResL, 0, floatSizeResMax));
	(cudaMemset(d_ConvolutionResR, 0, floatSizeResMax));
	(cudaMemset(INPUT_512, 0, maxBufferSize * sizeof(float)));
	(cudaMemset(INPUT2_512, 0, maxBufferSize * sizeof(float)));
	(cudaMemset(d_IR_paddedL, 0, h_paddedSize * sizeof(float)));
	(cudaMemset(d_IR_paddedR, 0, h_paddedSize * sizeof(float)));
	(cudaMemset(d_TimeDomain_paddedL, 0, h_paddedSize * sizeof(float)));
	(cudaMemset(d_TimeDomain_paddedR, 0, h_paddedSize * sizeof(float)));

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

void GPUConvEngine_512::prepare(int sampleRate) {

	
	
	h_paddedSize = ((sampleRate / maxNumberOfTheads) + 1) * maxNumberOfTheads;
	h_numPartitions = h_paddedSize / maxNumberOfTheads;
	
	dBlocks.x = (h_numPartitions);

	threadsPerBlock.x = bs;
	numBlocks.x = (h_paddedSize + threadsPerBlock.x - 1) / threadsPerBlock.x;

	threadsPerBlockZero = bs;
	clear();
	
}

 



void  GPUConvEngine_512::process(const float* in, const float* in2, const float* in3, const float* in4, float* out1, float* out2) {
 
	//copy content and transfer
	int indexBs = h_index * bs;
	cudaMemcpyToSymbolAsync(INPUT_512, in, bs_float,0, cudaMemcpyHostToDevice,stream);
	cudaMemcpyToSymbolAsync(INPUT2_512, in2, bs_float,0, cudaMemcpyHostToDevice,stream);
	cudaMemcpyAsync(d_IR_paddedL + indexBs, in3, bs_float, cudaMemcpyHostToDevice , stream);
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
	std::memcpy(h_OverlapL, &h_ConvolutionResL[bs -  1 ], bs_float);
 	std::memcpy(h_OverlapR, &h_ConvolutionResR[bs -  1 ], bs_float);
	 
}




void  GPUConvEngine_512::launchEngine() {
	
	shiftAndInsertKernel_512 << <numBlocks, threadsPerBlock,0,stream >> > (d_TimeDomain_paddedL);
	shiftAndInsertKernel2_512 << <numBlocks, threadsPerBlock, 0, stream >> > (d_TimeDomain_paddedR);
	shared_partitioned_convolution_512 << <dBlocks,dThreads , SHMEM, stream >> > (d_ConvolutionResL, d_TimeDomain_paddedL, d_IR_paddedL);
	shared_partitioned_convolution_512 << <dBlocks, dThreads, SHMEM, stream >> > (d_ConvolutionResR, d_TimeDomain_paddedR, d_IR_paddedR);
	cudaMemcpyAsync(h_ConvolutionResL, d_ConvolutionResL, floatSizeRes, cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(h_ConvolutionResR, d_ConvolutionResR, floatSizeRes, cudaMemcpyDeviceToHost, stream);
	cudaMemsetAsync(d_ConvolutionResL, 0, floatSizeRes, stream);
	cudaMemsetAsync(d_ConvolutionResR, 0, floatSizeRes, stream);

	cudaStreamSynchronize(stream);
	//update index

	h_index = (h_index + 1) % (h_numPartitions);
	
}
