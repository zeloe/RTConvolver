

#ifndef GPUThread256_H
#define GPUThread256_H

#include "JuceHeader.h" 
#include "GPUThread0.h"
#include "../GPU/GPUConvEngine256.cuh" 


class GPUThread_256 :public GPUThread_0, public juce::Thread
{
public:
    GPUThread_256(GPUConvEngine_256* engine) : juce::Thread("GPUThread256"), activeEngine(engine)
    {
        prepare();

    }

    ~GPUThread_256() {
        stopThread(2000); // Gracefully stop the thread after waiting for 2 seconds.
    }
    virtual void start() override {
        activeEngine->clear();
        startThread(Priority::normal);
    }

    virtual void reset() override {
        activeEngine->clear();
        stopThread(200);
    }

    virtual void setSize(float Size) override {
        activeEngine->prepare(Size);
    }
    void prepare()
    {
        processingInBackground.store(false, std::memory_order_release);
        bufferToProcess.setSize(4, bs);
       
        bufferToProcess.clear();
        
    }

    void push(juce::AudioBuffer<float>& inputBuffer, juce::AudioBuffer<float>& outputBuffer) override {

        // Copy input data to the buffer to be processed
        bufferToProcess.copyFrom(0, 0, inputBuffer, 0, 0, bs); // Left channel
        bufferToProcess.copyFrom(1, 0, inputBuffer, 1, 0, bs); // Right channel
        bufferToProcess.copyFrom(2, 0, inputBuffer, 2, 0, bs); // SideChain Left channel
        bufferToProcess.copyFrom(3, 0, inputBuffer, 3, 0, bs); // SideChain Right channel
         

        // Signal background thread to start processing
        processingInBackground.store(true, std::memory_order_release);
        while (processingInBackground.load(std::memory_order_acquire)) {}
        // Wait for the processing to complete before copying output
        outputBuffer.copyFrom(0, 0, bufferToProcess, 0, 0, outputBuffer.getNumSamples());
        outputBuffer.copyFrom(1, 0, bufferToProcess, 1, 0, outputBuffer.getNumSamples());
         






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

    juce::AudioBuffer<float> bufferToProcess;  // Buffer for storing the input data
    juce::AudioBuffer<float> bufferToProcess2; // Buffer for storing the output

    const int bs = 256;
    std::atomic<bool> processingInBackground{ false };  // Atomic flag to indicate whether the background thread is processing





    // Active engine instance pointer for the selected engine
    GPUConvEngine_256* activeEngine = nullptr;


    void processConvolution()
    {
        const float* leftChannelA = bufferToProcess.getReadPointer(0);
        const float* rightChannelA = bufferToProcess.getReadPointer(1);
        const float* leftChannelB = bufferToProcess.getReadPointer(2);
        const float* rightChannelB = bufferToProcess.getReadPointer(3);

        // Process using the active engine
        float* outA = bufferToProcess.getWritePointer(0);
        float* outB = bufferToProcess.getWritePointer(1);

        activeEngine->process(leftChannelA, rightChannelA, leftChannelB, rightChannelB, outA, outB);
         
    }




    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GPUThread_256)

};

#endif
