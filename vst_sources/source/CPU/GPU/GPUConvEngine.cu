
#include "GPUConvEngine.cuh"
  

GPUConvEngine::GPUConvEngine() {
  
}



GPUConvEngine::~GPUConvEngine() {
	 
}

void GPUConvEngine::cleanup() {
	 
 



}


void GPUConvEngine::checkCudaError(cudaError_t err, const char* errMsg) {
	if (err != cudaSuccess) {
		printf("CUDA Error (%s): %s\n", errMsg, cudaGetErrorString(err));
	}
}

void GPUConvEngine::prepare(float sampleRate) {

	return;
	 
}




void  GPUConvEngine::launchEngine() {
	return;
	  
}
 
  





void  GPUConvEngine::getPointers(const float* in, const float* in2, const float* in3, const float* in4, float* out1, float* out2)  {
	return;
}


 