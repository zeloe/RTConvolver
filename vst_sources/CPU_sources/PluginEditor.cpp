#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginEditor::AudioPluginEditor (AudioPluginProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p)
{
    gui = std::make_unique<GUI>(processorRef);
    addAndMakeVisible(gui.get());
 
    setResizable(true, true);
    setResizeLimits(guiwidth,guiheight,guiwidth * 2,guiheight * 2);
  
    setSize (guiwidth,guiwidth);
}

AudioPluginEditor::~AudioPluginEditor()
{
   
}

//==============================================================================
void AudioPluginEditor::paint (juce::Graphics& g)
{
   
}

void AudioPluginEditor::resized()
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
 
 
 
