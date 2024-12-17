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
    limiterL = std::make_unique<Limiter>();
    limiterR = std::make_unique<Limiter>();

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
    return 0.f;
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
    //this is the total amount of samples in circualr buffer do not set this too high
    
    gain->prepare();
    swapper->prepare(samplesPerBlock, sampleRate);
    float size = treeState.state.getProperty(sizeParamID, 0.5f);
    swapper->prepareEngines(size);
   
    sliceBuf.setSize(4,samplesPerBlock);
    sliceBuf.clear();
    limiterL->prepare(sampleRate);
    limiterR->prepare(sampleRate);

    maxBs = samplesPerBlock;
}

void AudioPluginAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}
bool AudioPluginAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const int numInputMain = layouts.getNumChannels(true, 0);
    const int numInputSide = layouts.getNumChannels(true, 1);
    const int numOutput = layouts.getMainOutputChannels();



    if (numInputMain == 2 && numOutput == 2 && numInputSide == 2)
        return true;

    return false;

}
 
void AudioPluginAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused(midiMessages);
    juce::ScopedNoDenormals noDenormals;

    const int bs = buffer.getNumSamples();
    auto inputBus = getBus(true, 0);
    auto inputBuffer= inputBus->getBusBuffer(buffer);
    auto sideChainBus = getBus(true, 1);
    auto sideChainBuffer = sideChainBus->getBusBuffer(buffer);


    // Clear and copy input data to sliceBuf
    sliceBuf.clear();
    sliceBuf.copyFrom(0, 0, inputBuffer, 0 ,0, bs);
    sliceBuf.copyFrom(1, 0, inputBuffer, 1, 0, bs);
    sliceBuf.copyFrom(2, 0, sideChainBuffer, 0, 0, bs);
    sliceBuf.copyFrom(3, 0, sideChainBuffer, 1, 0, bs);
    auto outBus = getBus(false, 0);
    auto outBuffer = outBus->getBusBuffer(buffer);
    swapper->push(sliceBuf, sliceBuf); 
     

    gain->process(sliceBuf);
    sliceBuf.applyGain(0.5f);
    limiterL->process(sliceBuf.getReadPointer(0), sliceBuf.getWritePointer(0), maxBs);
    limiterR->process(sliceBuf.getReadPointer(1), sliceBuf.getWritePointer(1), maxBs);
    outBuffer.copyFrom(0, 0, sliceBuf, 0, 0, bs);
    outBuffer.copyFrom(1, 0, sliceBuf, 1, 0, bs);



   
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

void AudioPluginAudioProcessor::getSize(float Size) {

    swapper->prepareEngines(Size);
}