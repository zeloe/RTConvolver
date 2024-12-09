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
 
        
       
 
    }

    ~ProcessorSwapper() {
     
    }

    void prepare(int samplesPerBlock, int sampleRate)
    {
 
       
        m_sampleRate = sampleRate;
        cur_bs = samplesPerBlock;

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
 
       
       
        activeThread->push(inputBuffer, outputBuffer);
     //   outputBuffer.applyGain(0.015f);
        normalizeAudioBuffer(outputBuffer);
 
 
      
         
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
 


    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ProcessorSwapper)
 
}; 
#endif
