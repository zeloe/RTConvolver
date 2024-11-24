

#ifndef GPUThread0_H
#define GPUThread0_H

#include "JuceHeader.h" 
#include "../GPU/GPUConvEngine.cuh" 

 
class GPUThread_0
{
public:

    GPUThread_0(){};
    GPUThread_0(GPUConvEngine* engine) :activeEngine(engine)
    {
        prepare_Def();
         
    }

    virtual ~GPUThread_0() {
         
    }
    virtual void start() {

    }

    virtual void reset() {
        
    }
   
    virtual void push(juce::AudioBuffer<float>& inputBuffer, juce::AudioBuffer<float>& outputBuffer) {
        return;
        






    }
     

    
    

private:
    void prepare_Def()
    {
        processingInBackground.store(false, std::memory_order_release);
        bufferToProcess.setSize(4, bs);
        bufferToProcess2.setSize(2, bs);
        bufferToProcess.clear();
        bufferToProcess2.clear();
    }

    juce::AudioBuffer<float> bufferToProcess;  // Buffer for storing the input data
    juce::AudioBuffer<float> bufferToProcess2; // Buffer for storing the output
    
    const int bs =  512;
    std::atomic<bool> processingInBackground{ false };  // Atomic flag to indicate whether the background thread is processing

     
    

    
    // Active engine instance pointer for the selected engine
    GPUConvEngine* activeEngine = nullptr;
     

  


    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GPUThread_0)
 
};

#endif
