
#pragma once
#ifndef ProcessorSwapper_H
#define ProcessorSwapper_H
#include "JuceHeader.h"
#include "../pluginparamers/PluginParameters.h"
#include "../GPU/GPUConvEngine.cuh"
template<typename T>
class ProcessorSwapper : public juce::Thread {
public:


    ProcessorSwapper() : juce::Thread("GPUThread")
    {

        convEngine = std::make_unique<GPUConvEngine>();
        convEngine2 = std::make_unique<GPUConvEngine>();
    }



    ~ProcessorSwapper() {
        stopThread(2000);
    }



    void prepare(int samplesPerBlock, int size) {


        bs = samplesPerBlock;
         
        bufferToProcess.setSize(4, bs);
        bufferToProcess2.setSize(2, bs);
        bufferToProcess.clear();
        bufferToProcess2.clear();
        convEngine->prepare(samplesPerBlock, size);
        convEngine2->prepare(samplesPerBlock,size);
      
         

    }
     

    void push(juce::AudioBuffer<float>& inputBuffer) {
        // Acquire spin lock to ensure safe access to bufferToProcess
        {
        juce::SpinLock::ScopedLockType lockGuard(lock);
        bufferToProcess.copyFrom(0, 0, inputBuffer, 0, 0, bs);  // Left channel
        bufferToProcess.copyFrom(1, 0, inputBuffer, 1, 0, bs);  // Right channel
        bufferToProcess.copyFrom(2, 0, inputBuffer, 2, 0, bs);  // Left channel
        bufferToProcess.copyFrom(3, 0, inputBuffer, 3, 0, bs);  // Right channel
        startThread(Priority::normal);
        }
    }


    void run() override
    {
            // Lock scope for processing data from bufferToProcess
            {
                juce::SpinLock::ScopedLockType lockGuard(lock);

                // Process the data in bufferA
                const float* leftChannelA = bufferToProcess.getWritePointer(0);
                const float* rightChannelA = bufferToProcess.getWritePointer(1);
                // Process the data in bufferA
                const float* leftChannelB = bufferToProcess.getWritePointer(2);
                const float* rightChannelB = bufferToProcess.getWritePointer(3);
                // Call convolution engine or other processing function
                convEngine->process(leftChannelA, leftChannelB, bufferToProcess2.getWritePointer(0));
                convEngine2->process(rightChannelA, rightChannelB, bufferToProcess2.getWritePointer(1));
                
            }
        return;
    }

    // Check and retrieve the processed buffer for the main thread
    void retrieveProcessedBuffer(juce::AudioBuffer<float>& outputBuffer, int blockSize) {
        {
        juce::SpinLock::ScopedLockType lockGuard(lock);
            // Copy data from bufferB (processed data) to outputBuffer
            outputBuffer.copyFrom(0, 0, bufferToProcess2, 0, 0, blockSize); // Left channel
            outputBuffer.copyFrom(1, 0, bufferToProcess2, 1, 0, blockSize); // Right channel
        }
        return;
    }

     
   
   
private:
    juce::AudioBuffer<float> bufferToProcess;
    juce::AudioBuffer<float> bufferToProcess2;
    juce::SpinLock lock;
    std::unique_ptr<GPUConvEngine> convEngine;
    std::unique_ptr<GPUConvEngine> convEngine2;
    
    
	int bs = 0; 
 };

#endif