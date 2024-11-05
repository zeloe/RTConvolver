
#pragma once
#ifndef _GUI_H
#define _GUI_H

#include "../PluginProcessor.h"
#include <JuceHeader.h>

#include "GUIDefines.h"
#include "../pluginparamers/PluginParameters.h"
#include "CSlider.h"
class GUI : public juce::Component
{
public:
    
    GUI(AudioPluginAudioProcessor& processor);
    
    ~GUI() override ;
    
    void resized() override;
    void paint(juce::Graphics& g) override;
    AudioPluginAudioProcessor& proc;
private:
    std::unique_ptr<ZenSlider> VolumeKnob;
    juce::AudioProcessorValueTreeState::SliderAttachment  volumeAttach;
     
  
};

#endif // !_GUI_H