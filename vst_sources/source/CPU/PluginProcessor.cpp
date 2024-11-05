#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginAudioProcessor::AudioPluginAudioProcessor()
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                     .withInput("Main Input", juce::AudioChannelSet::stereo(), true)
                     .withInput("Sidechain Input", juce::AudioChannelSet::stereo(), true) // Sidechain bus
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ),
                       treeState (*this, nullptr, juce::Identifier ("Parameters"), PluginParameter::createParameterLayout())
{
    
    swapper = std::make_unique<ProcessorSwapper<float>>();
    gain = std::make_unique<Gain>(treeState);


}

AudioPluginAudioProcessor::~AudioPluginAudioProcessor()
{
}

//==============================================================================
const juce::String AudioPluginAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool AudioPluginAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool AudioPluginAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool AudioPluginAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double AudioPluginAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int AudioPluginAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int AudioPluginAudioProcessor::getCurrentProgram()
{
    return 0;
}

void AudioPluginAudioProcessor::setCurrentProgram (int index)
{
    juce::ignoreUnused (index);
}

const juce::String AudioPluginAudioProcessor::getProgramName (int index)
{
    juce::ignoreUnused (index);
    return {};
}

void AudioPluginAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
    juce::ignoreUnused (index, newName);
}

//==============================================================================
void AudioPluginAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    DBG(samplesPerBlock);
    int totalSize = (((sampleRate * 2) / samplesPerBlock) + 1) * samplesPerBlock;
    gain->prepare();
    swapper->prepare(samplesPerBlock, totalSize);
    
   
    sliceBuf.setSize(4,samplesPerBlock);
    sliceBuf.clear();
    out.setSize(2, samplesPerBlock);
    out.clear();
}

void AudioPluginAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}
bool AudioPluginAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    // Ensure that main buses are not disabled
    if (layouts.getMainInputChannelSet() == juce::AudioChannelSet::disabled() ||
        layouts.getMainOutputChannelSet() == juce::AudioChannelSet::disabled())
    {
        return false;
    }

    // Ensure that the main output bus is either mono or stereo
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono() &&
        layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
    {
        return false;
    }

    // Ensure that the input and output channel sets are the same
    if (layouts.getMainInputChannelSet() != layouts.getMainOutputChannelSet())
    {
        return false; // Input and output must match
    }

    // Handle sidechain if it exists
    if (layouts.getBuses(true).size() > 1) // Checking input buses
    {
        for (int i = 1; i < layouts.getBuses(true).size(); ++i) // Start from the second bus
        {
            if (layouts.getNumChannels(true, i) > 0) // Check if there's a channel
            {
                auto sidechainChannelSet = layouts.getChannelSet(true, i);
                // Ensure sidechain is either mono or stereo
                if (sidechainChannelSet != juce::AudioChannelSet::mono() &&
                    sidechainChannelSet != juce::AudioChannelSet::stereo())
                {
                    return false;
                }
            }
        }
    }

    return true; // All checks passed; layout is supported
}

void AudioPluginAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused(midiMessages);
    juce::ScopedNoDenormals noDenormals;

    const int bs = buffer.getNumSamples();


    // Check for sufficient input channels
 
  
    if (buffer.getMagnitude(0, bs) == false && out.getMagnitude(0, bs) == false) {
        return;
    }

    auto inputs = getBusBuffer(buffer, true, 0);



    auto sidechainInput = getBusBuffer(buffer, true, 1); // 1 is the index for the sidechain bus
    // Clear and copy input data to sliceBuf
    sliceBuf.clear();
    const float* inputA = inputs.getReadPointer(0);
    const float* inputB = inputs.getReadPointer(1);
    
    const float* sideChainC = sidechainInput.getReadPointer(0);
    const float* sideChainD = sidechainInput.getReadPointer(1);

    float* sliceA = sliceBuf.getWritePointer(0);
    float* sliceB = sliceBuf.getWritePointer(1);
    float* sliceC = sliceBuf.getWritePointer(2);
    float* sliceD = sliceBuf.getWritePointer(3);


    std::copy(inputA, inputA + bs, sliceA);
    std::copy(inputB, inputB + bs, sliceB);
    std::copy(sideChainC, sideChainC + bs, sliceC);
    std::copy(sideChainD, sideChainD + bs, sliceD);
    



    swapper->push(sliceBuf);
    
    swapper->retrieveProcessedBuffer(out,bs);

    buffer.copyFrom(0, 0, out, 0, 0, bs);
    buffer.copyFrom(1, 0, out, 1, 0, bs);


    gain->process(buffer);

   
}

    

//==============================================================================
bool AudioPluginAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor()
{
    return new AudioPluginAudioProcessorEditor (*this); //juce::GenericAudioProcessorEditor(*this);//
}

//==============================================================================
void AudioPluginAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    juce::MemoryOutputStream mos(destData, true);
        treeState.state.writeToStream(mos);
}

void AudioPluginAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    auto tree = juce::ValueTree::readFromData(data, sizeInBytes);
        if (tree.isValid())
        {
            treeState.replaceState(tree);
        }
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AudioPluginAudioProcessor();
}
