#pragma once
#ifndef ProcessorSwapper_H
#define ProcessorSwapper_H

#include "JuceHeader.h"
#include "../pluginparamers/PluginParameters.h"
#include "GPUThread128.h"
#include "GPUThread256.h"
#include "GPUThread512.h"
#include "GPUThread1024.h"
#include "GPUThread0.h"
#include "../GPU/GPUConvEngine1024.cuh"
#include "../GPU/GPUConvEngine512.cuh"
#include "../GPU/GPUConvEngine256.cuh"
#include "../GPU/GPUConvEngine128.cuh"
#include "../GPU/GPUConvEngine.cuh"
#include <atomic>
#include <variant>  // For variant

template<typename T>
class ProcessorSwapper 
{
public:
    ProcessorSwapper() 
    {
<<<<<<< HEAD
 
        
 
        // Initialize all engines once as unique_ptr
       // Initialize all engines
          // Initialize all engines
        engine_0 = std::make_unique<GPUConvEngine>();
        engine_128 = std::make_unique<GPUConvEngine_128>();
        engine_256 = std::make_unique<GPUConvEngine_256>();
        engine_512 = std::make_unique<GPUConvEngine_512>();
        engine_1024 = std::make_unique<GPUConvEngine_1024>();

        thread_0 = std::make_unique<GPUThread_0>(engine_0.get());
        thread_128 = std::make_unique<GPUThread_128>(engine_128.get());
        thread_256 = std::make_unique<GPUThread_256>(engine_256.get());
        thread_512 = std::make_unique<GPUThread_512>(engine_512.get());
        thread_1024 = std::make_unique<GPUThread_1024>(engine_1024.get());



        // Set default active engine and function pointer
        activeThread = thread_0.get();
        activeThread->start();
        cur_size = 0;
     
=======
        convEngine = std::make_unique<GPUConvEngine>();
        startThread(Priority::highest);
>>>>>>> main
    }

    ~ProcessorSwapper() {
     
    }

    void prepare(int samplesPerBlock, int sampleRate)
    {
<<<<<<< HEAD
       
        m_sampleRate = sampleRate;
        cur_bs = samplesPerBlock;
=======
        bs = samplesPerBlock;
        bufferToProcess.setSize(4, bs);
        bufferToProcess2.setSize(2, bs);
        bufferToProcess.clear();
        bufferToProcess2.clear();
        absBuffer.setSize(2, bs);
        absBuffer.clear();
        scaleFactors.setSize(1, 2);
        scaleFactors.clear();
        convEngine->prepare(samplesPerBlock, sampleRate);
       
    }
>>>>>>> main

        if (cur_bs != bs) {
             

             
            switchConvolutionEngines(samplesPerBlock);
            bs = cur_bs;
        }
        
        poles[0] = 0.f;
        poles[1] = 0.f;
        new_Gain = 0;
  
 
    }
    void push(juce::AudioBuffer<float>& inputBuffer, juce::AudioBuffer<float>& outputBuffer)
    {
<<<<<<< HEAD
       
       
        activeThread->push(inputBuffer, outputBuffer);
     //   outputBuffer.applyGain(0.015f);
        normalizeAudioBuffer(outputBuffer);
 
=======
      


        // Copy input data to the buffer to be processed
        bufferToProcess.copyFrom(0, 0, inputBuffer, 0, 0, bs); // Left channel
        bufferToProcess.copyFrom(1, 0, inputBuffer, 1, 0, bs); // Right channel
        bufferToProcess.copyFrom(2, 0, inputBuffer, 2, 0, bs); // SideChain Left channel
        bufferToProcess.copyFrom(3, 0, inputBuffer, 3, 0, bs); // SideChain Right channel

     
       

        // Signal background thread to start processing
        processingInBackground.store(true, std::memory_order_release);
        
        // Copy the result to the output buffer after processing
          outputBuffer.copyFrom(0, 0, bufferToProcess2, 0, 0, outputBuffer.getNumSamples());
          outputBuffer.copyFrom(1, 0, bufferToProcess2, 1, 0, outputBuffer.getNumSamples());
          //scale channels 1 and 2
          calculateScaleFactor(outputBuffer);



>>>>>>> main
    }

    void prepareEngines(float size) {
        cur_size = m_sampleRate * size;
        if (cur_size != m_size) {
            activeThread->reset();
            // Call prepare for each engine explicitly
            thread_128->setSize(cur_size);
            thread_256->setSize(cur_size);
            thread_512->setSize(cur_size);
            thread_1024->setSize(cur_size);
            activeThread->start();
            m_size = cur_size; // Update the size
        }
    }
        
     
     

private:
<<<<<<< HEAD
    
    static float calculateNormalisationFactor(float sumSquaredMagnitude, float targetRMS) {
        if (sumSquaredMagnitude < 1e-8f)
            return 1e-8f;

        return targetRMS / std::sqrt(sumSquaredMagnitude);
    }
    void normalizeAudioBuffer(juce::AudioBuffer<float>& outBuffer) {
        const int numChannels = outBuffer.getNumChannels();
        const int numSamples = outBuffer.getNumSamples();

        

        // Compute the sum of squared magnitudes
        for (int ch = 0; ch < numChannels; ++ch) {
            const float* channelData = outBuffer.getReadPointer(ch);
            float* data = outBuffer.getWritePointer(ch);
            
            float sumSquaredMagnitude = 0.0f;
            for (int i = 0; i < numSamples; ++i) {
                sumSquaredMagnitude += channelData[i] * channelData[i];
            }
            // Calculate the normalization factor
            float rms = outBuffer.getRMSLevel(ch, 0, numSamples);
            float normalizationFactor = calculateNormalisationFactor(sumSquaredMagnitude, rms);
            //outBuffer.applyGain(ch, 0, numSamples, normalizationFactor);
           // Smoothly apply the normalization factor
            smoothingGain(data, normalizationFactor, numSamples, poles[ch]);
        }

        
    }

    void smoothingGain(float* data, float targetGain, int numSamples, float& pole) {
        const float smoothingFactor = 0.99f; // Adjust for desired smoothing
        float currentGain = pole; // Start with the previous gain

        for (int sample = 0; sample < numSamples; ++sample) {
            // Smoothly transition to the target gain
            currentGain += (targetGain - currentGain) * (1.0f - smoothingFactor);
            data[sample] *= currentGain;
        }

        // Update the pole with the final gain value
        pole = currentGain;
    }

    float poles[2] = { 0 };
    float new_Gain = 0;
=======
    std::unique_ptr<GPUConvEngine> convEngine;
    juce::AudioBuffer<float> bufferToProcess;  // Buffer for storing the input data
    juce::AudioBuffer<float> bufferToProcess2; // Buffer for storing the output
    juce::AudioBuffer<float> scaleFactors;
    juce::AudioBuffer<float> absBuffer;
>>>>>>> main
    int bs = 0; // Number of samples per block
    int cur_bs = 0;
    int cur_size = 0;
    int m_size = 0;
    int m_sampleRate = 44100;
    float scale_factor = 0;
    unsigned int enginesIdx = 0;
    std::atomic<bool> processingInBackground{ false };  // Atomic flag to indicate whether the background thread is processing

    // Declare all the engines using unique pointers
 
    // Engine instances
    std::unique_ptr<GPUConvEngine> engine_0;
    std::unique_ptr<GPUConvEngine_128> engine_128;
    std::unique_ptr<GPUConvEngine_256> engine_256;
    std::unique_ptr<GPUConvEngine_512> engine_512;
    std::unique_ptr<GPUConvEngine_1024> engine_1024;

    // Thread instances
    std::unique_ptr<GPUThread_0> thread_0;
    std::unique_ptr<GPUThread_128> thread_128;
    std::unique_ptr<GPUThread_256> thread_256;
    std::unique_ptr<GPUThread_512> thread_512;
    std::unique_ptr<GPUThread_1024> thread_1024;
    // Active engine instance pointer for the selected engine
    GPUThread_0* activeThread = nullptr;
    

       

 
 

    

    void switchConvolutionEngines(int blockSize)
    {
        activeThread->reset();

        switch (blockSize) {
        /*
        case 128: {
            activeThread = thread_128.get();

            break;
        }


        case 256: {
             activeThread = thread_256.get();
             
            break;
        }
        */
        case 512: {
            activeThread = thread_512.get();
          
            break;
        }
        case 1024: {
            activeThread = thread_1024.get();
         
            break;
        }
        default: {
            activeThread = thread_0.get();
          

        }
        
        }
        activeThread->start();
    }
<<<<<<< HEAD


    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ProcessorSwapper)
 
};
=======
>>>>>>> main

void calculateScaleFactor(juce::AudioBuffer<float>& output)
{
    const int numSamples = output.getNumSamples();
    float* absWrite = absBuffer.getWritePointer(0);
    const float* absRead = absBuffer.getReadPointer(0);
    for (int ch = 0; ch < 2; ++ch)
    {
         
         
        float rms = output.getRMSLevel(ch, 0, numSamples);
        
    
        float scaleFactor = 0.20f /(rms +1e-6f);



        output.applyGain(ch, 0, numSamples, scaleFactor);

        

    }
  
   
}

};
#endif
