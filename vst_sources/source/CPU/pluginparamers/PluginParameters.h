


#ifndef PLUGINPARAMETER_H
#define PLUGINPARAMETER_H

#include <JuceHeader.h>
#include "../GUI/GUIDefines.h"

class PluginParameter
{
public:
    PluginParameter();
    ~PluginParameter();
    
    inline static const juce::String
        GAIN = "param_Gain";
      
     
    
    inline static const juce::String
        GAIN_NAME = "Gain";
      
    
    
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
  
    

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginParameter)
};

#endif 
