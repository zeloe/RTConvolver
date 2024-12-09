#pragma once
#include <JuceHeader.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include "pluginparamers/PluginParameters.h"
#include "DSP/ProcessorSwapper.h"
#include "DSP/GainStaging.h"
//==============================================================================
class AudioPluginAudioProcessor final : public juce::AudioProcessor
{
public:
    //==============================================================================
    AudioPluginAudioProcessor();
    ~AudioPluginAudioProcessor() override;

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

    juce::AudioProcessorValueTreeState treeState;
    juce::Identifier sizeParamID{ "SizeMenuIndex" }; // Identifier for your SizeMenu parameter

    void getSize(float size);
  
    std::unique_ptr<ProcessorSwapper<float>>  swapper;
private:
    std::unique_ptr <Gain> gain;
    juce::AudioBuffer<float> out;
    juce::AudioBuffer<float> sliceBuf;
    int maxBs;
    int totalSize = 0;
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessor)
};
