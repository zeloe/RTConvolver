#pragma once

#include "PluginProcessor.h"
#include "GUI/GUI.h"
#include "pluginparamers/PluginParameters.h"
//==============================================================================
class AudioPluginEditor final : public juce::AudioProcessorEditor
{
public:
    explicit AudioPluginEditor (AudioPluginProcessor&);
    ~AudioPluginEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;
     
    
private:
   
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    AudioPluginProcessor& processorRef;
    std::unique_ptr<GUI> gui;


    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginEditor)
};
