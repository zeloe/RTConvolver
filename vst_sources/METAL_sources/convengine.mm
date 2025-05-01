#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "convengine.h"

const char* metalLibPath = METAL_LIBRARY_PATH; // Access the path defined in CMake

GPU_ConvolutionEngine::GPU_ConvolutionEngine() {
    initDevice();
    // Get max buffer size
    bs = 1024;
    
    // Size of BlockSize (bytes)
    bsFloat = bs * sizeof(float);
    // Size of result (int)
    convResSize = bs * 2;
    // Size of result (bytes)
    convResSizeFloat = convResSize * sizeof(float);
    //
    
    partitions = ((48000.0f * 10.f) / float(bs)) + 1;
    paddedSize = partitions * bs;
    paddedSizeFloat = paddedSize * sizeof(float);
    allocateOnDevice();
    
    
    convResBufferA = (float*)calloc(convResSize, sizeof(float));
    overLapBufferA = (float*)calloc(bs, sizeof(float));
    
    convResBufferB = (float*)calloc(convResSize, sizeof(float));
    overLapBufferB = (float*)calloc(bs, sizeof(float));
    
    uint sizes[2] = { static_cast<uint>(bs), static_cast<uint>(convResSize)};
    
    _sizes = _pDevice->newBuffer(sizeof(uint) * 2, MTL::ResourceStorageModeShared);
    memcpy(_sizes->contents(), sizes, sizeof(uint) * 2);

    // Create a command queue
    createCommandQueue();
    
    // get Functions from .metallib
    createDefaultLibrary();
    
    //create Compute Pipelines
    createComputePipeLine();
    paddedSize = ((48000.f) / float(bs)) + 1;
    //set gridsize and the number of threads and total size of shared memory
    gridSize = MTL::Size::Make(paddedSize,1,1);
    numberOfThreads = MTL::Size::Make(bs,1,1);
    totalSharedMemorySize = bs * 4 * sizeof(float);
    index = 0;
}


void GPU_ConvolutionEngine::freeMemory() {
    
    free(overLapBufferA);
    free(convResBufferA);
    free(overLapBufferB);
    free(convResBufferB);
    
   
    _dryBufferA->release();
    _dryBufferB->release();
    
    _timeDomainA->release();
    _timeDomainB->release();
    _timeDomainC->release();
    _timeDomainD->release();
    
    _resultBufferA->release();
    _resultBufferB->release();
    _CommandBuffer->release();
    _mCommandQueue->release();
    _convolutionPipeline->release();
    _shiftAndInsertPipeline->release();
    _sizes->release();
    _library->release();
    _pDevice->release();
}


GPU_ConvolutionEngine::~GPU_ConvolutionEngine()
{
    
}
void GPU_ConvolutionEngine::process(const float* inputA,const float* inputB,const float* inputC,const float* inputD, float* outputA,  float* outputB)
{
    
    // Copy input data to the dry buffer
    int index_bytes = bsFloat * index;
    memcpy(_dryBufferA->contents(), inputA, bsFloat);
    memcpy(_dryBufferB->contents(), inputB, bsFloat);
    memcpy(static_cast<char*>(_timeDomainC->contents()) + index_bytes,inputC,bsFloat);
    memcpy(static_cast<char*>(_timeDomainD->contents()) + index_bytes,inputD,bsFloat);
    //compute
    @autoreleasepool {
        sendComputeCommandCommand();
   }
    //copy results to host
    memcpy(convResBufferA,_resultBufferA->contents(), convResSizeFloat);
    memcpy(convResBufferB,_resultBufferB->contents(), convResSizeFloat);
    
    
    
    for (int i = 0; i < bs; i++) {
        outputA[i] = (convResBufferA[i] + overLapBufferA[i]) * 0.25f;
        overLapBufferA[i] = convResBufferA[bs + i];
    }
    
    for (int i = 0; i < bs; i++) {
        outputB[i] = (convResBufferB[i] + overLapBufferB[i]) * 0.25f;
        overLapBufferB[i] = convResBufferB[bs + i];
    }
    
    //reset result buffer
    memset(_resultBufferA->contents(), 0.f, convResSizeFloat);
    memset(_resultBufferB->contents(), 0.f, convResSizeFloat);
    
    index = index + 1;
    if (index > partitions) {
        index = 0;
    }
}


void GPU_ConvolutionEngine::initDevice() {
    // Get the device (GPU)
    _pDevice = MTL::CreateSystemDefaultDevice();
}


void GPU_ConvolutionEngine::allocateOnDevice() {
    
   
    _dryBufferA = _pDevice->newBuffer(bsFloat, MTL::ResourceStorageModeShared);
    _dryBufferB= _pDevice->newBuffer(bsFloat, MTL::ResourceStorageModeShared);
    _timeDomainA= _pDevice->newBuffer(paddedSizeFloat, MTL::ResourceStorageModeShared);
    _timeDomainB= _pDevice->newBuffer(paddedSizeFloat, MTL::ResourceStorageModeShared);
    _timeDomainC= _pDevice->newBuffer(paddedSizeFloat, MTL::ResourceStorageModeShared);
    _timeDomainD= _pDevice->newBuffer(paddedSizeFloat, MTL::ResourceStorageModeShared);
   
    _resultBufferA = _pDevice->newBuffer(convResSizeFloat, MTL::ResourceStorageModeShared);
    _resultBufferB = _pDevice->newBuffer(convResSizeFloat, MTL::ResourceStorageModeShared);
    

    
    clear();
    
    
}

void GPU_ConvolutionEngine::clear() {
    
    memset(_dryBufferA->contents(), 0.f, bsFloat);
    memset(_dryBufferB->contents(), 0.f, bsFloat);
    memset(_timeDomainA->contents(), 0.f, paddedSizeFloat);
    memset(_timeDomainB->contents(), 0.f, paddedSizeFloat);
    memset(_timeDomainC->contents(), 0.f, paddedSizeFloat);
    memset(_timeDomainD->contents(), 0.f, paddedSizeFloat);
    memset(_resultBufferA->contents(), 0.f, convResSizeFloat);
    memset(_resultBufferB->contents(), 0.f,convResSizeFloat);
    
}

void GPU_ConvolutionEngine::createDefaultLibrary() {
    // Load the default library from the bundle
    NS::Error* pError = nullptr; // To capture any errors
    NS::String* filePath = NS::String::string(metalLibPath, NS::ASCIIStringEncoding);
    _library = _pDevice->newLibrary(filePath, &pError);
    if (!_library) {
        std::cerr << "Failed to load Metal library" ;
    }
}


void GPU_ConvolutionEngine::createCommandQueue() {
    
    _mCommandQueue  = _pDevice->newCommandQueue();
    
}

void GPU_ConvolutionEngine::prepare(int block_Size, float size) {
    size_param = size;
    bs = block_Size;
    setSize(size);
    bsFloat = bs * sizeof(float);
    // Size of result (int)
    convResSize = bs * 2;
    // Size of result (bytes)
    convResSizeFloat = convResSize * sizeof(float);
    //
    uint sizes[2] = { static_cast<uint>(bs), static_cast<uint>(convResSize)};
    memcpy(_sizes->contents(), sizes, sizeof(uint) * 2);
    
    numberOfThreads = MTL::Size::Make(bs,1,1);
}


void GPU_ConvolutionEngine::createComputePipeLine() {
    

    MTL::Function *shift_and_insert = _library->newFunction(NS::String::string("shiftAndInsertKernel",NS::ASCIIStringEncoding));
    
    assert(shift_and_insert);
    
    MTL::Function* convolution = _library->newFunction(NS::String::string("shared_partitioned_convolution", NS::ASCIIStringEncoding));
    
    assert(convolution);
    
    
    NS::Error* shift_error;
    
   
    _shiftAndInsertPipeline = _pDevice->newComputePipelineState(shift_and_insert,&shift_error);
    
    assert(_shiftAndInsertPipeline);
    
    NS::Error* convolution_error;
    _convolutionPipeline = _pDevice->newComputePipelineState(convolution,&convolution_error);
    
    assert(_convolutionPipeline);
    
    
    shift_and_insert->release();
    convolution->release();
}


void GPU_ConvolutionEngine::sendComputeCommandCommand() {
    
    
    _CommandBuffer = _mCommandQueue->commandBuffer();

    {
        auto encoder = _CommandBuffer->computeCommandEncoder();

        // Shift + Insert A
        encoder->setComputePipelineState(_shiftAndInsertPipeline);
        encoder->setBuffer(_timeDomainA, 0, 0);
        encoder->setBuffer(_dryBufferA, 0, 1);
        encoder->setBuffer(_sizes, 0, 2);
        encoder->dispatchThreads(gridSize, numberOfThreads);

        // Shift + Insert B
        encoder->setComputePipelineState(_shiftAndInsertPipeline);
        encoder->setBuffer(_timeDomainB, 0, 0);
        encoder->setBuffer(_dryBufferB, 0, 1);
        encoder->setBuffer(_sizes, 0, 2);
        encoder->dispatchThreads(gridSize, numberOfThreads);

        // Convolution A
        encoder->setComputePipelineState(_convolutionPipeline);
        encoder->setBuffer(_resultBufferA, 0, 0);
        encoder->setBuffer(_timeDomainA, 0, 1);
        encoder->setBuffer(_timeDomainC, 0, 2);
        encoder->setBuffer(_sizes, 0, 3);
        encoder->setThreadgroupMemoryLength(totalSharedMemorySize, 0);
        encoder->dispatchThreads(gridSize, numberOfThreads);

        // Convolution B
        encoder->setComputePipelineState(_convolutionPipeline);
        encoder->setBuffer(_resultBufferB, 0, 0);
        encoder->setBuffer(_timeDomainB, 0, 1);
        encoder->setBuffer(_timeDomainD, 0, 2);
        encoder->setBuffer(_sizes, 0, 3);
        encoder->setThreadgroupMemoryLength(totalSharedMemorySize, 0);
        encoder->dispatchThreads(gridSize, numberOfThreads);

        encoder->endEncoding();
    }

    if (previousCommandBuffer) {
        previousCommandBuffer->waitUntilCompleted();  // Wait only if previous is still running
        previousCommandBuffer->release();
    }
    previousCommandBuffer = _CommandBuffer;
    previousCommandBuffer->retain();
    _CommandBuffer->commit();

    
    
}


void GPU_ConvolutionEngine::setSize(float size) {
    size_param = size;
    partitions = (size_param / float(bs)) + 1.f;
    paddedSize = partitions * bs;
    gridSize = MTL::Size::Make(paddedSize,1,1);
    clear();
    index = 0;
}

