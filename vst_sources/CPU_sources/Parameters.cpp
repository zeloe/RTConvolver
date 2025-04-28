
#include "Parameters.h"

template<typename T>
static void castParameter(juce::AudioProcessorValueTreeState& apvts, const juce::ParameterID& id, T& destination) {
    destination = dynamic_cast<T>(apvts.getParameter(id.getParamID()));
    jassert(destination);
}



Parameters::Parameters(juce::AudioProcessorValueTreeState& treeState)
{
    castParameter(treeState,gainParamID,gainParam);
}


void Parameters::prepareToPlay(double sampleRate) noexcept
{
    double duration = 0.02;
    gainSmoother.reset(sampleRate,duration);

}

void Parameters::reset() noexcept
{
    gain = 0.f;

    gainSmoother.setCurrentAndTargetValue(juce::Decibels::decibelsToGain(gainParam->get()));

}


juce::AudioProcessorValueTreeState::ParameterLayout Parameters::createParameterLayout() {
    
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    
    layout.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID {"gain", 1},
                                                           "Output Gain",
                                                           juce::NormalisableRange<float>(-60.f,0.f),
                                                           0.f));
    return layout;
}

void Parameters::update() noexcept
{
    gainSmoother.setTargetValue(juce::Decibels::decibelsToGain( gainParam->get()));
}

void Parameters::smoothen() noexcept
{
    gain = gainSmoother.getNextValue();
}
