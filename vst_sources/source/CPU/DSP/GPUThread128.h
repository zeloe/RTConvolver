#pragma once
#ifndef ProcessorSwapper_H
#define ProcessorSwapper_H

#include "JuceHeader.h"
#include "../pluginparamers/PluginParameters.h"
#include "../GPU/GPUConvEngine1024.cuh"
#include "../GPU/GPUConvEngine512.cuh"
#include "../GPU/GPUConvEngine.cuh"
#include <atomic>
#include <variant>  // For variant

template<typename T>
class ProcessorSwapper : public juce::Thread
{
public:
    ProcessorSwapper() : juce::Thread("GPUThread")
    {
        // Initialize all engines once as unique_ptr
       // Initialize all engines
          // Initialize all engines
        engine_0 = std::make_unique<GPUConvEngine>();
        engine_512 = std::make_unique<GPUConvEngine_512>();
        engine_1024 = std::make_unique<GPUConvEngine_1024>();

        // Set default active engine and function pointer
        activeEngine = engine_0.get();
     


        startThread(Priority::highest);
    }

    ~ProcessorSwapper() {
        stopThread(2000); // Gracefully stop the thread after waiting for 2 seconds.
    }

    void prepare(int samplesPerBlock, int sampleRate)
    {
        processingInBackground.store(false, std::memory_order_release);
        cur_sr = sampleRate;
        cur_bs = samplesPerBlock;

        if (cur_bs != bs) {
            bufferToProcess.setSize(4, cur_bs);
            bufferToProcess2.setSize(2, cur_bs);

            absBuffer.setSize(2, cur_bs);
            absBuffer.clear();
            scaleFactors.setSize(1, 2);
            scaleFactors.clear();
            switchConvolutionEngines(samplesPerBlock);
            bs = cur_bs;
        }
        prepareEngines(sampleRate);
        bufferToProcess.clear();
        bufferToProcess2.clear();
    }

    // Push the buffer to the background thread for processing
    void push(juce::AudioBuffer<float>& inputBuffer, juce::AudioBuffer<float>& outputBuffer)
    {
        // Copy input data to the buffer to be processed
        bufferToProcess.copyFrom(0, 0, inputBuffer, 0, 0, bs); // Left channel
        bufferToProcess.copyFrom(1, 0, inputBuffer, 1, 0, bs); // Right channel
        bufferToProcess.copyFrom(2, 0, inputBuffer, 2, 0, bs); // SideChain Left channel
        bufferToProcess.copyFrom(3, 0, inputBuffer, 3, 0, bs); // SideChain Right channel

        bufferToProcess.applyGain(0.25);

        // Signal background thread to start processing
        processingInBackground.store(true, std::memory_order_release);
        // Wait for the processing to complete before copying output
        outputBuffer.copyFrom(0, 0, bufferToProcess2, 0, 0, outputBuffer.getNumSamples());
        outputBuffer.copyFrom(1, 0, bufferToProcess2, 1, 0, outputBuffer.getNumSamples());

        // Scale channels 1 and 2
        outputBuffer.applyGain(0.25);
    }

    void run() override
    {
        while (!threadShouldExit()) {
            if (processingInBackground.load(std::memory_order_acquire)) {
                processConvolution();
                processingInBackground.store(false, std::memory_order_release);
            }
        }
    }

    

private:
    void prepareEngines(int sampleRate) {
        cur_sr = sampleRate;
        if (cur_sr != sr) {
            // Call prepare for each engine explicitly
            if (engine_512) engine_512->prepare(cur_sr);
            if (engine_1024) engine_1024->prepare(cur_sr);

            sr = cur_sr; // Update the sample rate to the current sample rate
        }
    }
    juce::AudioBuffer<float> bufferToProcess;  // Buffer for storing the input data
    juce::AudioBuffer<float> bufferToProcess2; // Buffer for storing the output
    juce::AudioBuffer<float> scaleFactors;
    juce::AudioBuffer<float> absBuffer;
    int bs = 0; // Number of samples per block
    int cur_bs = 0;
    int cur_sr = 0;
    int sr = 0;
    float pole = 0;
    float new_Gain = 0;
    float scale_factor = 0;
    unsigned int enginesIdx = 0;
    std::atomic<bool> processingInBackground{ false };  // Atomic flag to indicate whether the background thread is processing

    // Declare all the engines using unique pointers
 
    // Engine instances
    std::unique_ptr<GPUConvEngine> engine_0;
    std::unique_ptr<GPUConvEngine_512> engine_512;
    std::unique_ptr<GPUConvEngine_1024> engine_1024;

    
    // Active engine instance pointer for the selected engine
    GPUConvEngine* activeEngine = nullptr;


    void processConvolution()
    {
        const float* leftChannelA = bufferToProcess.getReadPointer(0);
        const float* rightChannelA = bufferToProcess.getReadPointer(1);
        const float* leftChannelB = bufferToProcess.getReadPointer(2);
        const float* rightChannelB = bufferToProcess.getReadPointer(3);

        // Process using the active engine
         
            // Direct call to the stored function pointer for processing
        activeEngine->getPointers(leftChannelA, rightChannelA, leftChannelB, rightChannelB, bufferToProcess.getWritePointer(0), bufferToProcess.getWritePointer(1));
       

        bufferToProcess2.copyFrom(0, 0, bufferToProcess, 0, 0, bs);
        bufferToProcess2.copyFrom(1, 0, bufferToProcess, 1, 0, bs);
    }

    void switchConvolutionEngines(int blockSize)
    {
        // Ensure no processing is happening in the background
        while (processingInBackground.load(std::memory_order_acquire)) {
            juce::Thread::sleep(1); // Prevent busy waiting
        }

        switch (blockSize) {
        case 512: {
            activeEngine = engine_512.get();
          
            break;
        }
        case 1024: {
            activeEngine = engine_1024.get();
         
            break;
        }
        default: {
            activeEngine = engine_0.get();
          

        }
        }
    }


    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ProcessorSwapper)
    //make sturct with pointers
};

#endif
