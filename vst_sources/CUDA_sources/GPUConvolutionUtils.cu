

 

#include "GPUConvolutionUtils.cuh"


__shared__ float SUB_CONV[SHARED_ARRAY_SIZE];
__constant__ int D_SIZES[2];
 
namespace GPUConv {
	void initSizes() {
		int cpu_sizes[2];
		cpu_sizes[0] = MAX_BUFFER_SIZE;
		cpu_sizes[1] = CONV_RES_SIZE;
		cudaMemcpyToSymbol(D_SIZES, cpu_sizes, 2 * sizeof(int));
	}




	cudaError_t checkCudaError(cudaError_t error) {
		if (error != cudaSuccess) {
			fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
			return error;
		}
		return cudaSuccess;
	}

	// Utility function to safely free CUDA device memory
	cudaError_t safeCudaFree(void* ptr) {
		cudaError_t error = cudaSuccess;
		if (ptr != nullptr) {
			error = cudaFree(ptr);
			if (error != cudaSuccess) {
				fprintf(stderr, "CUDA error freeing device memory: %s\n", cudaGetErrorString(error));
			}
		}

		return error;
	}

	// Utility function to safely free host memory allocated with malloc/calloc
	void safeHostFree(void* ptr) {
		if (ptr != nullptr) {
			free(ptr);
		}
	}



	extern "C" __global__ void   shiftAndInsertKernel(float* __restrict__ delayBuffer, const float* input) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		// Insert new elements at the beginning of the delay buffer
		if (tid < D_SIZES[0]) {
			delayBuffer[tid] = input[tid];
		}

		delayBuffer[tid + D_SIZES[0]] = delayBuffer[tid];
	}




	extern "C" __global__ void  sharedPartitionedConvolution(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp) {
		const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

		const unsigned int copy_idx = threadIdx.x;


		// Declare pointers to the shared memory partitions
		float* arr1 = &SUB_CONV[0];
		float* arr2 = &SUB_CONV[D_SIZES[0]];
		float* tempResult = &SUB_CONV[D_SIZES[1]];
		// Load data into shared memory
		tempResult[copy_idx] = 0.f;
		tempResult[D_SIZES[0] + copy_idx] = 0.f;


		arr1[copy_idx] = Dry[thread_idx];
		arr2[copy_idx] = Imp[thread_idx];

		// Shared memory to accumulate results before writing them to global memory
		// Convolution operation (reduction into shared memory)
		#pragma unroll
		for (int i = 0; i < D_SIZES[0]; i++) {
			int inv = (i - copy_idx) % D_SIZES[0];
			tempResult[i + inv] += arr1[i] * arr2[inv];
		}

		__syncthreads();  // Ensure all threads in the block have finished processing


		// Write the accumulated result to global memory (only for the first thread)
		if (copy_idx == 0) {
			// Write the first part of the result (up to SIZES[0] * 2 - 1)
			#pragma unroll
			for (int i = 0; i < D_SIZES[1]; i++) {
				atomicAdd(&Result[i], tempResult[i]);
			}
		}

	}

}