#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p)
{
    gui = std::make_unique<GUI>(processorRef);
    addAndMakeVisible(gui.get());
 
    setResizable(true, true);
    setResizeLimits(guiwidth,guiheight,guiwidth * 2,guiheight * 2);
  
    setSize (guiwidth,guiwidth);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
   
}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g)
{
   
}

void AudioPluginAudioProcessorEditor::resized()
{
    
     
    auto area = getLocalBounds();
        if (area.isEmpty ())
        {
            return;
        }
    else
    {
       
       
     
        float scaleX = (float)area.getWidth() / (float)guiwidth;
        float scaleY = (float)area.getHeight() / (float)guiheight;
        
       
        gui->setBounds(area);
        gui->setTransform(juce::AffineTransform::scale(scaleX,scaleY));
        
    }
}
 
 
 