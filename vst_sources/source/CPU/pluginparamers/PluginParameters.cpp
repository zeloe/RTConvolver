#include "PluginParameters.h"


PluginParameter::PluginParameter()
{
    
}

PluginParameter::~PluginParameter()
{
    
}


juce::AudioProcessorValueTreeState::ParameterLayout PluginParameter::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    
    params.push_back (std::make_unique<juce::AudioParameterFloat> (GAIN,
                                                                   GAIN_NAME,
                                                                       0.0,
                                                                       1.f,
                                                                       0.5));
    
   
    
    
    
    
    
    return { params.begin(), params.end() };
}
