
#include "GUI.h"

GUI::GUI(AudioPluginAudioProcessor& processor) : proc(processor),
VolumeKnob(std::make_unique<ZenSlider>(juce::Colours::red)),
volumeAttach(proc.treeState, "param_Gain", *VolumeKnob.get())
{

    
    setSize(guiwidth,guiheight);
    VolumeKnob->setSliderStyle(juce::Slider::RotaryVerticalDrag);
    VolumeKnob->setTextBoxStyle(juce::Slider::NoTextBox, true, 0, 0);

    addAndMakeVisible(VolumeKnob.get());
    
    
}

GUI::~GUI()
{

 
}
 

void GUI::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
 
    
}


void GUI::resized()
{
    if(!getLocalBounds().isEmpty())
    {
        
        VolumeKnob->setBounds(offsetx, offsety, size, size);
        
        
    }
}

 