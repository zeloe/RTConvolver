 
#include "GPUConvolutionEngine.cuh"



GPU_ConvolutionEngine::GPU_ConvolutionEngine() {

	GPUConv::initSizes();
	dThreads.x = MAX_BUFFER_SIZE;
	 

	h_ConvolutionRes_A = (float*)calloc(CONV_RES_SIZE, sizeof(float));
	h_ConvolutionRes_B = (float*)calloc(CONV_RES_SIZE, sizeof(float));
	h_Overlap_A = (float*)calloc(MAX_BUFFER_SIZE, sizeof(float));
	h_Overlap_B = (float*)calloc(MAX_BUFFER_SIZE, sizeof(float));
	h_index = 0;


	
	(cudaMalloc((void**)&d_ConvolutionRes_A, CONV_RES_SIZE_FLOAT));
	(cudaMalloc((void**)&d_ConvolutionRes_B, CONV_RES_SIZE_FLOAT));

	(cudaMalloc((void**)&d_input_A, MAX_BUFFER_SIZE_FLOAT));
	(cudaMalloc((void**)&d_input_B, MAX_BUFFER_SIZE_FLOAT)); 

	 

	dBlocks.x = int(float(MAX_CONVOLUTION_SIZE) / 2.f);
	 
	
	
	(cudaMalloc((void**)&d_TimeDomain_A, MAX_CONVOLUTION_SIZE_FLOAT));
	(cudaMalloc((void**)&d_TimeDomain_B, MAX_CONVOLUTION_SIZE_FLOAT));
	(cudaMalloc((void**)&d_TimeDomain_C, MAX_CONVOLUTION_SIZE_FLOAT));
	(cudaMalloc((void**)&d_TimeDomain_D, MAX_CONVOLUTION_SIZE_FLOAT));

	clear();

}




GPU_ConvolutionEngine::~GPU_ConvolutionEngine() {

	cleanup();

}






void GPU_ConvolutionEngine::clear() {

		 
	cudaMemset(d_ConvolutionRes_A, 0.f, CONV_RES_SIZE_FLOAT);
	cudaMemset(d_ConvolutionRes_B, 0.f, CONV_RES_SIZE_FLOAT);
	cudaMemset(d_input_A, 0.f, MAX_BUFFER_SIZE_FLOAT);
	cudaMemset(d_input_B, 0.f, MAX_BUFFER_SIZE_FLOAT);
	cudaMemset(d_TimeDomain_A, 0.f, MAX_CONVOLUTION_SIZE_FLOAT);
	cudaMemset(d_TimeDomain_B, 0.f, MAX_CONVOLUTION_SIZE_FLOAT);
	cudaMemset(d_TimeDomain_C, 0.f, MAX_CONVOLUTION_SIZE_FLOAT);
	cudaMemset(d_TimeDomain_D, 0.f, MAX_CONVOLUTION_SIZE_FLOAT);
	memset(h_ConvolutionRes_A, 0.f, CONV_RES_SIZE_FLOAT);
	memset(h_ConvolutionRes_B, 0.f, CONV_RES_SIZE_FLOAT);
	memset(h_Overlap_A, 0.f, MAX_BUFFER_SIZE_FLOAT);
	memset(h_Overlap_B, 0.f, MAX_BUFFER_SIZE_FLOAT);

	h_index = 0;

}



void GPU_ConvolutionEngine::cleanup() {


	GPUConv::safeCudaFree(d_ConvolutionRes_A);
    d_ConvolutionRes_A = nullptr;
    GPUConv::safeCudaFree(d_ConvolutionRes_B);
    d_ConvolutionRes_B = nullptr;

	GPUConv::safeCudaFree(d_input_A);
    d_input_A = nullptr;
    GPUConv::safeCudaFree(d_input_B);
    d_input_B = nullptr;
	 

    GPUConv::safeCudaFree(d_TimeDomain_A);
    d_TimeDomain_A = nullptr;
	GPUConv::safeCudaFree(d_TimeDomain_B);
    d_TimeDomain_B = nullptr;
    
	GPUConv::safeCudaFree(d_TimeDomain_C);
    d_TimeDomain_C = nullptr;
	GPUConv::safeCudaFree(d_TimeDomain_D);
	d_TimeDomain_D = nullptr;

	GPUConv::safeHostFree(h_ConvolutionRes_A);
	GPUConv::safeHostFree(h_ConvolutionRes_B);
	GPUConv::safeHostFree(h_Overlap_A);
	GPUConv::safeHostFree(h_Overlap_B);

	 

}


void GPU_ConvolutionEngine::prepare(int buffersize, float size) {

	bs = buffersize;
	setSize(size);
	GPUConv::changeSizes(bs);
	dThreads.x = bs;
	
	bs_float = bs * sizeof(float);
	conv_res_size = bs * 2;
	conv_res_float = conv_res_size * sizeof(float);
}


void GPU_ConvolutionEngine::setSize(float size) {

	cudaDeviceSynchronize( );

	
	// Ensure proper padding
	h_numPartitions = ((size / float(bs)) + 1.f);

	dBlocks.x = h_numPartitions;

	clear();
}





void GPU_ConvolutionEngine::process(const float* h_input_A, const float* h_input_B, const float* h_input_C, const float* h_input_D, float* h_output_A, float* h_output_B) {

	//copy content and transfer
	int indexBs = h_index * bs;

	cudaMemcpy(d_input_A, h_input_A, bs_float, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input_B, h_input_B, bs_float, cudaMemcpyHostToDevice);

	cudaMemcpy(d_TimeDomain_C + indexBs, h_input_C, bs_float, cudaMemcpyHostToDevice);
	cudaMemcpy(d_TimeDomain_D + indexBs, h_input_D, bs_float, cudaMemcpyHostToDevice);

	launchEngines();

	//Copy to output and perform overlap add
 	for (int i = 0; i < bs; i++) {
		
		h_output_A[i] = h_ConvolutionRes_A[i] + h_Overlap_A[i];
		h_output_B[i] = h_ConvolutionRes_B[i] + h_Overlap_B[i];
	}

	 // Copy the last elements as overlap values for the next block
		memcpy(h_Overlap_A, &h_ConvolutionRes_A[bs - 1], bs_float);
		memcpy(h_Overlap_B, &h_ConvolutionRes_B[bs - 1], bs_float);

}
  

void GPU_ConvolutionEngine::launchEngines() {


	// A
	GPUConv::shiftAndInsertKernel << <dBlocks, dThreads ,0 >> > (d_TimeDomain_A,d_input_A);

	// B
	GPUConv::shiftAndInsertKernel << <dBlocks, dThreads,0 >> > (d_TimeDomain_B,d_input_B);

	// Result A
	GPUConv::sharedPartitionedConvolution << <dBlocks, dThreads,0 >> > (d_ConvolutionRes_A,d_TimeDomain_A, d_TimeDomain_C);

	// Result B 
	GPUConv::sharedPartitionedConvolution << <dBlocks, dThreads,0 >> > (d_ConvolutionRes_B,d_TimeDomain_B, d_TimeDomain_D);

	//Copy A to Host
	cudaMemcpy(h_ConvolutionRes_A, d_ConvolutionRes_A, conv_res_float, cudaMemcpyDeviceToHost);

	//Copy B to Host
	cudaMemcpy(h_ConvolutionRes_B, d_ConvolutionRes_B, conv_res_float, cudaMemcpyDeviceToHost);


	// Synchronize to ensure all operations are complete
	cudaDeviceSynchronize();

	// Reset accumulation buffers
	cudaMemset(d_ConvolutionRes_A, 0.f, CONV_RES_SIZE_FLOAT);
	cudaMemset(d_ConvolutionRes_B, 0.f, CONV_RES_SIZE_FLOAT);


	//Increment Index
	h_index = (h_index + 1) % h_numPartitions;

}