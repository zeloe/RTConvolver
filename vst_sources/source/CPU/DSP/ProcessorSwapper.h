#pragma once
#ifndef ProcessorSwapper_H
#define ProcessorSwapper_H

#include "JuceHeader.h"
#include "../pluginparamers/PluginParameters.h"
#include "../GPU/GPUConvEngine.cuh"
#include <atomic>

template<typename T>
class ProcessorSwapper : public juce::Thread
{
public:
    ProcessorSwapper() : juce::Thread("GPUThread")
    {
        convEngine = std::make_unique<GPUConvEngine>();
        startThread(Priority::highest);
    }

    ~ProcessorSwapper() {
        stopThread(2000); // Gracefully stop the thread after waiting for 2 seconds.
    }

    void prepare(int samplesPerBlock, int size)
    {
        bs = samplesPerBlock;
        bufferToProcess.setSize(4, bs);
        bufferToProcess2.setSize(2, bs);
        bufferToProcess.clear();
        bufferToProcess2.clear();
        absBuffer.setSize(1, bs);
        absBuffer.clear();
        scaleFactors.setSize(1, 2);
        scaleFactors.clear();
        convEngine->prepare(samplesPerBlock, size);
       
    }

    // Push the buffer to the background thread for processing
    void push(juce::AudioBuffer<float>& inputBuffer, juce::AudioBuffer<float>& outputBuffer)
    {
      


        // Copy input data to the buffer to be processed
        bufferToProcess.copyFrom(0, 0, inputBuffer, 0, 0, bs); // Left channel
        bufferToProcess.copyFrom(1, 0, inputBuffer, 1, 0, bs); // Right channel
        bufferToProcess.copyFrom(2, 0, inputBuffer, 2, 0, bs); // SideChain Left channel
        bufferToProcess.copyFrom(3, 0, inputBuffer, 3, 0, bs); // SideChain Right channel

        //scale channels 1 and 2
        calculateScaleFactor(bufferToProcess);

        // Signal background thread to start processing
        processingInBackground.store(true, std::memory_order_release);
     
        // Copy the result to the output buffer after processing
          outputBuffer.copyFrom(0, 0, bufferToProcess2, 0, 0, outputBuffer.getNumSamples());
          outputBuffer.copyFrom(1, 0, bufferToProcess2, 1, 0, outputBuffer.getNumSamples());
     
    }

    void run() override
    {
        while (!threadShouldExit()) {
            // Only process if the background processing flag is set
            if (processingInBackground.load(std::memory_order_acquire)) {
                processConvolution();

                // Once done processing, signal that the thread is done
                processingInBackground.store(false, std::memory_order_release);
            }
        }
    }

private:
    std::unique_ptr<GPUConvEngine> convEngine;
    juce::AudioBuffer<float> bufferToProcess;  // Buffer for storing the input data
    juce::AudioBuffer<float> bufferToProcess2; // Buffer for storing the output
    juce::AudioBuffer<float> scaleFactors;
    juce::AudioBuffer<float> absBuffer;
    int bs = 0; // Number of samples per block

    std::atomic<bool> processingInBackground{ false };  // Atomic flag to indicate whether the background thread is processing

    // Process the convolution and send the result to the output buffer.
    void processConvolution()
    {
        const float* leftChannelA = bufferToProcess.getReadPointer(0);
        const float* rightChannelA = bufferToProcess.getReadPointer(1);
        const float* leftChannelB = bufferToProcess.getReadPointer(2);
        const float* rightChannelB = bufferToProcess.getReadPointer(3);

        // Call the convolution engine to process the audio
        convEngine->process(leftChannelA, rightChannelA, leftChannelB, rightChannelB,
            bufferToProcess.getWritePointer(0), bufferToProcess.getWritePointer(1));

        // Store the result in bufferToProcess2 (ready for the audio thread to use)
        bufferToProcess2.copyFrom(0, 0, bufferToProcess, 0, 0, bs);  // Left channel
        bufferToProcess2.copyFrom(1, 0, bufferToProcess, 1, 0, bs);  // Right channel
    }

void calculateScaleFactor(juce::AudioBuffer<float>& output)
{
    const int numSamples = output.getNumSamples();
    float* absWrite = absBuffer.getWritePointer(0);
    const float* absRead = absBuffer.getReadPointer(0);
    for (int ch = 0; ch < 2; ++ch)  
    {
        const float* data = output.getReadPointer(ch);



        juce::FloatVectorOperations::abs(absWrite, data, bs);
        // Accumulate the sum of absolute values
        float sum = 0.0f;
        for (int i = 0; i < numSamples; ++i)
        {
            sum += absRead[i];
        }
        float* scaleWrite = scaleFactors.getWritePointer(0);
        // Compute the scale factor
        scaleWrite[ch] = (sum > 0.0f) ? 1.0f / sum : 1.0f;
    }
    const float* scaleRead = scaleFactors.getReadPointer(0);
    output.applyGain(0, 0, numSamples, scaleRead[0]);
    output.applyGain(1, 0, numSamples, scaleRead[1]);
}
};
#endif
