
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









 
__global__ void  shared_partitioned_convolution_1024(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int idx = tid + bid * blockDim.x;


	// Declare pointers to the shared memory partitions

	float* shared_arr1 = &partArray_1024[0];

	float* shared_arr2 = &partArray_1024[SIZES_1024[0]];

	float* tempResult = &partArray_1024[SIZES_1024[0] * 2];

	const int block_size = SIZES_1024[0];

	const int result_size = SIZES_1024[1];

	const int num_Partitions = SIZES_1024[2];
	
	 
	tempResult[tid] = 0.f;
	tempResult[tid + block_size] = 0.f;
	// Load data into shared memory
	shared_arr1[tid] = Dry[idx];
	shared_arr2[tid] = Imp[idx];
	__syncthreads();

	// Perform convolution with loop unrolling and careful index handling
	
	float sum = 0.0f;
	for (int j = 0; j <= tid; j++) {
		sum += shared_arr1[tid - j] * shared_arr2[j];
	}
	tempResult[tid] = sum;

	float sum2 = 0.f;

//	sum2 += shared_arr1[tid] * shared_arr2[block_size - tid];
//	tempResult[block_size] += sum2;
	 

	for (int i = tid + block_size; i < result_size; i += blockDim.x) {
		float sum3 = 0.0f;

		int start_j = max(0, i - block_size + 1);  // Prevent negative indices
		int end_j = block_size;

		for (int j = start_j; j < end_j; j++) {
			int idx1 = i - j;  // Index for shared_arr1
			if (idx1 >= 0 && idx1 < block_size) {  // Validate index
				sum3 += shared_arr1[idx1] * shared_arr2[j];
			}
		}
		tempResult[i] = sum3;

	}

	if (tid == 0) {
		for (int i = 0; i < result_size; i++) {
			atomicAdd(&Result[i], tempResult[i]);
		}
	}
	 

}


	 


GPUConvEngine_1024::GPUConvEngine_1024() {
	 
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
	 

	cudaDeviceSynchronize( );

	// Ensure proper padding
	int temp_h_paddedSize = ((size / maxBufferSize) + 1) * maxBufferSize;
	 
	// Update dBlocks and other parameters
	dBlocks.x = temp_h_paddedSize / maxBufferSize;
	h_numPartitions = dBlocks.x;
	// Update sizes array
	cpu_sizes[2] = dBlocks.x;

	// Copy updated sizes to device
	 cudaMemcpyToSymbol(SIZES_1024, cpu_sizes, sizeof(int) * 3);
}




 

void  GPUConvEngine_1024::process(const float* in, const float* in2, const float* in3, const float* in4, float* out1, float* out2) {
	
	 
	//copy content and transfer
	int indexBs = h_index * maxBufferSize;
	cudaMemcpyToSymbol(INPUT_1024, in, bs_float, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(INPUT2_1024,in2, bs_float, 0, cudaMemcpyHostToDevice);
	cudaMemcpy(d_IR_paddedL + indexBs, in3, bs_float, cudaMemcpyHostToDevice);
	cudaMemcpy(d_IR_paddedR + indexBs, in4, bs_float, cudaMemcpyHostToDevice);



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
	shiftAndInsertKernel_1024 << <dBlocks, dThreads >> > (d_TimeDomain_paddedL);
	 

	// Kernel 2: shiftAndInsertKernel2_1024
	shiftAndInsertKernel2_1024 << <dBlocks, dThreads>> > (d_TimeDomain_paddedR);
	 

	// Kernel 3: shared_partitioned_convolution_1024 for Left channel
	 
  	shared_partitioned_convolution_1024<<<dBlocks, dThreads>>> (d_ConvolutionResL, d_TimeDomain_paddedL, d_IR_paddedL);

	// Kernel 4: shared_partitioned_convolution_1024 for Right channel
	 
	shared_partitioned_convolution_1024 << <dBlocks, dThreads>> > (d_ConvolutionResR, d_TimeDomain_paddedR, d_IR_paddedR);


	cudaMemcpy(h_ConvolutionResL, d_ConvolutionResL, floatSizeRes, cudaMemcpyDeviceToHost);
	 

	cudaMemcpy(h_ConvolutionResR, d_ConvolutionResR, floatSizeRes, cudaMemcpyDeviceToHost);
	 

	 
	 

	// Synchronize to ensure all operations are complete
	cudaDeviceSynchronize();

	cudaMemset(d_ConvolutionResL, 0, floatSizeRes);
	cudaMemset(d_ConvolutionResR, 0, floatSizeRes);
	// Update the index for overlap handling
	h_index = (h_index + 1) % h_numPartitions;
}
