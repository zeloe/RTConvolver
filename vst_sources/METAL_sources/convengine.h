#ifndef CONV_ENGINE_H
#define CONV_ENGINE_H

#include <iostream>
#include <cassert>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
class GPU_ConvolutionEngine
{
public:
    GPU_ConvolutionEngine();
    ~GPU_ConvolutionEngine();
    
    void process(const float* inputA,const float* inputB,const float* inputC,const float* inputD, float* outputA,  float* outputB);
    void init();
    void setSize(float size);
    void prepare(int block_size, float size);
private:
    void initDevice();
    
    void allocateOnDevice();
    
    void createDefaultLibrary();
    void createCommandQueue();
    void createComputePipeLine();
    void encodeComputeCommand(MTL::ComputeCommandEncoder* computeEncoder);
    void sendComputeCommandCommand();
    void clear();
    void freeMemory();
    
    MTL::Device* _pDevice;
    
    MTL::Buffer* _timeDomainA;
    MTL::Buffer* _timeDomainB;
    MTL::Buffer* _timeDomainC;
    MTL::Buffer* _timeDomainD;
    MTL::Buffer* _sizes ;
    MTL::Buffer* _dryBufferA;
    MTL::Buffer* _dryBufferB;
    MTL::Buffer* _resultBufferA;
    MTL::Buffer* _resultBufferB;
    
    MTL::CommandBuffer* _CommandBuffer;
    MTL::CommandBuffer* previousCommandBuffer;
    MTL::CommandQueue* _mCommandQueue;
    
    MTL::Library* metalDefaultLibrary;
    MTL::Library* _library;

    MTL::ComputePipelineState* _pipeLine;
    
    //
    MTL::ComputePipelineState* _convolutionPipeline;
    MTL::ComputePipelineState* _shiftAndInsertPipeline;
    //
    int offset = 0;
    int bs = 0;
    int bsFloat = 0;
    float size_param = 0;
    int convResSize = 0;
    int convResSizeFloat = 0;
    int paddedSize = 0;
    int paddedSizeFloat = 0;
    int partitions = 0;
    
    
    float* convResBufferA = nullptr;
    float* overLapBufferA = nullptr;
    
    float* convResBufferB = nullptr;
    float* overLapBufferB = nullptr;
    uint totalSharedMemorySize = 0;
    //
    MTL::Size gridSize;
    MTL::Size numberOfThreads;
    int index = 0;
};



#endif //CONV_ENGINE_H
