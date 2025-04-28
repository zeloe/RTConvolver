
#include "GUI.h"

GUI::GUI(AudioPluginProcessor& processor) : proc(processor),
VolumeKnob(std::make_unique<ZenSlider>(juce::Colours::red)),
volumeAttach(proc.treeState, "gain", *VolumeKnob.get())
{

    sizeMenu = std::make_unique<SizeMenu>(proc.treeState.state, proc.sizeParamID,proc);
    setSize(guiwidth,guiheight);
    VolumeKnob->setSliderStyle(juce::Slider::RotaryVerticalDrag);
    VolumeKnob->setTextBoxStyle(juce::Slider::NoTextBox,true,0,0);
    dbLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    dbLabel.setText(juce::String(VolumeKnob->getValue(), 1) + " dB", juce::dontSendNotification);

    addAndMakeVisible(dbLabel);
    addAndMakeVisible(VolumeKnob.get());
    addAndMakeVisible(sizeMenu.get());
    VolumeKnob->onValueChange = [this]() { updateDbLabel(); };
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
        
        VolumeKnob->setBounds(volumeRect);
        auto textWidth = 50; // Adding padding
        int xpos = VolumeKnob->getBounds().getCentreX() - textWidth / 2;
        dbLabel.setBounds(xpos, VolumeKnob->getBottom(), textWidth, 20);

        sizeMenu->setBounds(sizeRect);
        
    }
}

void GUI::updateDbLabel() {
    auto valueInDb = static_cast<float>(VolumeKnob->getValue());
    dbLabel.setText(juce::String(valueInDb, 1) + " dB", juce::dontSendNotification);
}
