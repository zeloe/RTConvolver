#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #include "../CUDA_sources/GPUConvolutionEngine.cuh"
#elif defined(__APPLE__)
    #include "../METAL_sources/convengine.h"
#else
    #error "Compiling for an unknown platform"
#endif
#include <JuceHeader.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include "Parameters.h"

//==============================================================================
class AudioPluginProcessor final : public juce::AudioProcessor, public juce::Thread
{
public:
    //==============================================================================
    AudioPluginProcessor();
    ~AudioPluginProcessor() override;
    
    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;
    
    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;
    
    //==============================================================================
    const juce::String getName() const override;
    
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;
    
    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;
    
    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;
    
    juce::AudioProcessorValueTreeState treeState {
        *this, nullptr, "Parameters", Parameters::createParameterLayout()
    };
    juce::Identifier sizeParamID{ "SizeMenuIndex" }; // Identifier for your SizeMenu parameter
    
    void getSize(float size);
    
    
private:
    Parameters params;
    std::unique_ptr<GPU_ConvolutionEngine> gpu_convolution;
    juce::AudioBuffer<float> out;
    int maxBs;
    int totalSize = 0;
    int bs_process = 256;
    static constexpr float sizes[7] = {0.25f, 0.5f, 1.f, 1.5f, 2.f, 2.5f, 3.f};
    void run() override;
    std::atomic<bool> isProcessing;
    juce::AbstractFifo audioFifo_to_GPU { 1024 * 10 };  // enough for 4 blocks (adjust as needed)
    juce::AbstractFifo audioFifo_from_GPU { 1024 * 10 };  // enough for 4 blocks (adjust as needed)
    juce::AudioBuffer<float> fifoInputBuffer { 4, 4096 * 10 };  // 4 channels, 4096 samples
    juce::AudioBuffer<float> fifoOutputBuffer { 4, 4096 * 10}; // to hold output after processing
    juce::AudioBuffer<float> sliceBuf { 4, 4096 * 10 };
    float lastSize = 0.f;
    static constexpr float epsilon = 0.001f;  // Define a small tolerance value
    static constexpr float rms = 0.707f;
    float lastNormA = 0.5f;
    float lastNormB = 0.5f;
    
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginProcessor)
};
