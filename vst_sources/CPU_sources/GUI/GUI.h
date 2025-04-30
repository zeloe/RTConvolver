
#pragma once
#ifndef _GUI_H
#define _GUI_H

#include "../PluginProcessor.h"
#include <JuceHeader.h>

#include "GUIDefines.h"
#include "../Parameters.h"
#include "CSlider.h"
#include "SizeMenu.h"
class GUI : public juce::Component
{
public:
    
    GUI(AudioPluginProcessor& processor);
    
    ~GUI() override ;
    
    void resized() override;
    void paint(juce::Graphics& g) override;
    AudioPluginProcessor& proc;
private:
    std::unique_ptr<ZenSlider> VolumeKnob;
    std::unique_ptr<SizeMenu> sizeMenu;
    juce::AudioProcessorValueTreeState::SliderAttachment  volumeAttach;
    juce::Label dbLabel;
  
    
    void updateDbLabel();
  
};

#endif // !_GUI_H
