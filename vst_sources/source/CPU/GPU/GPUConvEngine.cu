
#include "GPUConvEngine.cuh"
// Define the constant memory array
__constant__ int SIZES[2];
__constant__ float INPUT[1024];
__constant__ float INPUT2[1024];
__global__ void shared_partitioned_convolution(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int copy_idx = threadIdx.x;
	extern __shared__ float partArray[];
	float* arr1 = &partArray[0];
	float* arr2 = &partArray[SIZES[0]];
	arr1[copy_idx] = Dry[thread_idx];
	arr2[copy_idx] = Imp[thread_idx];


	__syncthreads();
	 
	for (int i = 0; i < SIZES[0]; i++) {
		int inv = (SIZES[0] - copy_idx) % SIZES[0];
		atomicAdd(&Result[i + inv], arr1[i] * arr2[inv]);
	}
}

__global__ void  shiftAndInsertKernel(float* __restrict__ delayBuffer) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// Insert new elements at the beginning of the delay buffer
	if (tid < SIZES[0]) {
		delayBuffer[tid] = INPUT[tid];
	}
	
		delayBuffer[tid + SIZES[0]] = delayBuffer[tid];

	
}



__global__ void  shiftAndInsertKernel2(float* __restrict__ delayBuffer, int* __restrict__ index) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int indexInDelayBuffer = *index * SIZES[0];
	 
	delayBuffer[indexInDelayBuffer + tid] = INPUT2[tid];

}

GPUConvEngine::GPUConvEngine() {

}



GPUConvEngine::~GPUConvEngine() {
	cudaFree(d_IR_padded);
	cudaFree(SIZES);
	cudaFree(d_TimeDomain_padded);
	cudaFree(d_ConvolutionRes);
	cudaFree(d_Input);
	cudaFree(&d_index);
	free(h_ConvolutionRes);
	free(h_Overlap);
	free(h_sizesOfSubPartitions);
}

void GPUConvEngine::checkCudaError(cudaError_t err, const char* errMsg) {
	if (err != cudaSuccess) {
		printf("CUDA Error (%s): %s\n", errMsg, cudaGetErrorString(err));
	}
}

void GPUConvEngine::prepare(int maxBufferSize, int size) {
	h_convResSize = maxBufferSize * 2 - 1;
	int floatSizeRes = h_convResSize * sizeof(float);
	checkCudaError(cudaMalloc((void**)&d_ConvolutionRes, floatSizeRes), "d_ConvolutionRes malloc");
	checkCudaError(cudaMemset(d_ConvolutionRes, 0, floatSizeRes), "d_ConvolutionRes memset");
	SHMEM = 2 * sizeof(float) * maxBufferSize;
	bs = maxBufferSize;
	 
	int* cpu_sizes = (int*)calloc(2, sizeof(int));
	h_result_ptr = (float*)calloc(maxBufferSize, sizeof(float));
	h_ConvolutionRes = (float*)calloc(h_convResSize, sizeof(float));
	h_Overlap = (float*)calloc(bs, sizeof(float));
	h_index = (int*)calloc(1, sizeof(int));
	cudaMalloc((void**)&d_index, sizeof(int));
	cudaMemcpy(&d_index, &h_index, sizeof(int), cudaMemcpyHostToDevice);
	cpu_sizes[0] = bs;
	checkCudaError(cudaMalloc((void**)&d_Input, bs * sizeof(float)), "d_Input malloc");
	checkCudaError(cudaMemset(d_Input, 0, bs * sizeof(float)), "d_Input memset");
	checkCudaError(cudaMemset(INPUT, 0, bs * sizeof(float)), "d_Input memset");
	checkCudaError(cudaMemset(INPUT2, 0, bs * sizeof(float)), "d_Input memset");

	h_numPartitions = size / bs;
	h_paddedSize = h_numPartitions * bs;

	cpu_sizes[1] = h_paddedSize;
	cudaMemcpyToSymbol(SIZES, cpu_sizes, 2 * sizeof(int));


	checkCudaError(cudaMalloc((void**)&d_IR_padded, h_paddedSize * sizeof(float)), "d_IR_padded malloc");
	checkCudaError(cudaMemset(d_IR_padded, 0, h_paddedSize * sizeof(float)), "d_IR_padded memset");

	checkCudaError(cudaMalloc((void**)&d_TimeDomain_padded, h_paddedSize * sizeof(float)), "d_TimeDomain_padded malloc");
	checkCudaError(cudaMemset(d_TimeDomain_padded, 0, h_paddedSize * sizeof(float)), "d_TimeDomain_padded memset");

	dThreads.x = bs;

	dBlocks.x = (h_numPartitions);

	threadsPerBlock.x = bs;
	numBlocks.x = (h_paddedSize + threadsPerBlock.x - 1) / threadsPerBlock.x;
	free(cpu_sizes);
}





void  GPUConvEngine::process(const float* in, const float* in2, float* out) {
	 
	//copy content and transfer
	cudaMemcpyToSymbol(INPUT, in, bs * sizeof(float));
	cudaMemcpy(d_IR_padded + *h_index * bs, in2, bs * sizeof(float), cudaMemcpyHostToDevice);



	//launch the convolution Engine
	launchEngine();


	//perform overlap add
	for (int i = 0; i < bs; i++) {
		out[i] = (h_ConvolutionRes[i] + h_Overlap[i]) * 0.15f;
		h_Overlap[i] = h_ConvolutionRes[i + bs - 1];
	}
	 

}




void  GPUConvEngine::launchEngine() {
	
	shiftAndInsertKernel << <numBlocks, threadsPerBlock >> > (d_TimeDomain_padded);
	shared_partitioned_convolution << <dBlocks,dThreads , SHMEM>> > (d_ConvolutionRes, d_TimeDomain_padded, d_IR_padded);
	cudaDeviceSynchronize();
	 
		 
 
	
			cudaMemcpy(h_ConvolutionRes, d_ConvolutionRes, h_convResSize * sizeof(float), cudaMemcpyDeviceToHost);
	//set the result to 0

	cudaMemset(d_ConvolutionRes, 0, h_convResSize * sizeof(float));
	*h_index = (*h_index + 1) % (h_numPartitions);
	cudaMemcpy(&d_index, &h_index, sizeof(int), cudaMemcpyHostToDevice);

}
