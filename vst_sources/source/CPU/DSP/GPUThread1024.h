

#ifndef GPUThread1024_H
#define GPUThread1024_H

#include "JuceHeader.h" 
#include "GPUThread0.h"
#include "../GPU/GPUConvEngine1024.cuh" 
 

class GPUThread_1024 :public GPUThread_0, public juce::Thread
{
public:
    GPUThread_1024(GPUConvEngine_1024* engine) : juce::Thread("GPUThread1024"), activeEngine(engine)
    {
        prepare();
       
    }

    ~GPUThread_1024() {
         stopThread(2000); // Gracefully stop the thread after waiting for 2 seconds.
    }
    virtual void start() override {
     
        startThread(Priority::normal);
    }

    virtual void setSize(float Size) override {
        activeEngine->prepare(Size);
    }

    virtual void reset() override {
         activeEngine->clear();
         processingInBackground.store(false, std::memory_order_release);
         stopThread(10);
    }
     void prepare() 
    {
        processingInBackground.store(false, std::memory_order_release);
        bufferToProcess.setSize(4, bs);
        bufferToProcess2.setSize(2, bs);
        bufferToProcess.clear();
        bufferToProcess2.clear();
    }

     void push(juce::AudioBuffer<float>& inputBuffer, juce::AudioBuffer<float>& outputBuffer) override {
       
        // Copy input data to the buffer to be processed
        bufferToProcess.copyFrom(0, 0, inputBuffer, 0, 0, bs); // Left channel
        bufferToProcess.copyFrom(1, 0, inputBuffer, 1, 0, bs); // Right channel
        bufferToProcess.copyFrom(2, 0, inputBuffer, 2, 0, bs); // SideChain Left channel
        bufferToProcess.copyFrom(3, 0, inputBuffer, 3, 0, bs); // SideChain Right channel
         

        // Signal background thread to start processing
       processingInBackground.store(true, std::memory_order_release);
     //  processConvolution();
       while (processingInBackground.load(std::memory_order_acquire)) {}
     
        // Wait for the processing to complete before copying output
        outputBuffer.copyFrom(0, 0, bufferToProcess2, 0, 0, outputBuffer.getNumSamples());
        outputBuffer.copyFrom(1, 0, bufferToProcess2, 1, 0, outputBuffer.getNumSamples());

        






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

    const int bs = 1024;
    std::atomic<bool> processingInBackground{ false };  // Atomic flag to indicate whether the background thread is processing





    // Active engine instance pointer for the selected engine
    GPUConvEngine_1024* activeEngine = nullptr;


    void processConvolution()
    {
        const float* leftChannelA = bufferToProcess.getReadPointer(0);
        const float* rightChannelA = bufferToProcess.getReadPointer(1);
        const float* leftChannelB = bufferToProcess.getReadPointer(2);
        const float* rightChannelB = bufferToProcess.getReadPointer(3);

        // Process using the active engine
        float* outA = bufferToProcess2.getWritePointer(0);
        float* outB = bufferToProcess2.getWritePointer(1);
         
         activeEngine->process(leftChannelA, rightChannelA, leftChannelB, rightChannelB, outA, outB);


        
    }




    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GPUThread_1024)

};

#endif
