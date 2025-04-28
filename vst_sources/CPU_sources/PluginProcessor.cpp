#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginProcessor::AudioPluginProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     .withInput("Main Input", juce::AudioChannelSet::stereo(), true)
                     .withInput("Sidechain Input", juce::AudioChannelSet::stereo(), true) // Sidechain bus
                     .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
juce::Thread("GPUThread"),
treeState (*this, nullptr, juce::Identifier ("Parameters"), PluginParameter::createParameterLayout())
#endif
{
    
    gpu_convolution = std::make_unique<GPU_ConvolutionEngine>();
    gain = std::make_unique<Gain>(treeState);
    
    startThread(Priority::highest);
}

AudioPluginProcessor::~AudioPluginProcessor()
{
    stopThread(2000);
}

//==============================================================================
const juce::String AudioPluginProcessor::getName() const
{
    return JucePlugin_Name;
}

bool AudioPluginProcessor::acceptsMidi() const
{
    return false;
}

bool AudioPluginProcessor::producesMidi() const
{
    return false;
}

bool AudioPluginProcessor::isMidiEffect() const
{
    return false;
}

double AudioPluginProcessor::getTailLengthSeconds() const
{
    return 0.f;
}

int AudioPluginProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int AudioPluginProcessor::getCurrentProgram()
{
    return 0;
}

void AudioPluginProcessor::setCurrentProgram (int index)
{
    juce::ignoreUnused (index);
}

const juce::String AudioPluginProcessor::getProgramName (int index)
{
    juce::ignoreUnused (index);
    return {};
}

void AudioPluginProcessor::changeProgramName (int index, const juce::String& newName)
{
    juce::ignoreUnused (index, newName);
}

//==============================================================================
void AudioPluginProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    
    auto sizeParamValue = treeState.state.getProperty(sizeParamID);
    if(sizeParamValue) {
    // Load the value and cast it to a float
    float Size = static_cast<float>(sizeParamValue);
    gpu_convolution->setSize(Size * float(sampleRate));
    }
    
    gain->prepare();
    
    
   
   
    sliceBuf.clear();
   

    maxBs = samplesPerBlock;
    lastNormA = 0.5f;
    lastNormB = 0.5f;
}

void AudioPluginProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}
bool AudioPluginProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const int numInputMain = layouts.getNumChannels(true, 0);
    const int numInputSide = layouts.getNumChannels(true, 1);
    const int numOutput = layouts.getMainOutputChannels();



    if (numInputMain == 2 && numOutput == 2 && numInputSide == 2)
        return true;

    return false;

}
 

void AudioPluginProcessor::processBlock(juce::AudioBuffer<float>& buffer,[[maybe_unused]] juce::MidiBuffer& midiMessages)
{
        const int bs = buffer.getNumSamples();  // Block size
        juce::ignoreUnused(midiMessages);
        juce::ScopedNoDenormals noDenormals;

        auto inputBus = getBus(true, 0);
        auto inputBuffer = inputBus->getBusBuffer(buffer);
        auto sideChainBus = getBus(true, 1);
        auto sideChainBuffer = sideChainBus->getBusBuffer(buffer);
        auto outBus = getBus(false, 0);
        auto outBuffer = outBus->getBusBuffer(buffer);

        // Attempt to push data into FIFO
        int start1, size1, start2, size2;
        audioFifo_to_GPU.prepareToWrite(bs, start1, size1, start2, size2);

        // Writing the data to the FIFO buffers
        if (size1 > 0)
        {
            fifoInputBuffer.copyFrom(0, start1, inputBuffer, 0, 0, size1);
            fifoInputBuffer.copyFrom(1, start1, inputBuffer, 1, 0, size1);
            fifoInputBuffer.copyFrom(2, start1, sideChainBuffer, 0, 0, size1);
            fifoInputBuffer.copyFrom(3, start1, sideChainBuffer, 1, 0, size1);
        }
        if (size2 > 0)
        {
            fifoInputBuffer.copyFrom(0, start2, inputBuffer, 0, 0, size2);
            fifoInputBuffer.copyFrom(1, start2, inputBuffer, 1, 0, size2);
            fifoInputBuffer.copyFrom(2, start2, sideChainBuffer, 0, 0, size2);
            fifoInputBuffer.copyFrom(3, start2, sideChainBuffer, 1, 0, size2);
        }

        // Finish writing the data into FIFO
        audioFifo_to_GPU.finishedWrite(size1 + size2);

        outBuffer.clear();

        // Now check if we have enough samples in the FIFO for output
        int availableOutSamples = audioFifo_from_GPU.getNumReady();
        if (availableOutSamples >= bs)
        {
            int start3, size3, start4, size4;
            audioFifo_from_GPU.prepareToRead(bs, start3, size3, start4, size4);

            if (size3 > 0)
            {
                outBuffer.copyFrom(0, 0, fifoOutputBuffer, 0, start3, size3);
                outBuffer.copyFrom(1, 0, fifoOutputBuffer, 1, start3, size3);
            }

            if (size4 > 0)
            {
                outBuffer.addFrom(0, size3, fifoOutputBuffer, 0, start4, size4);
                outBuffer.addFrom(1, size3, fifoOutputBuffer, 1, start4, size4);
            }

            audioFifo_from_GPU.finishedRead(size3 + size4);
        }
    
    const float outRMSA  = outBuffer.getRMSLevel(0, 0, bs);
    const float outRMSB  = outBuffer.getRMSLevel(1, 0, bs);
    
    float normA = 0.5f;
    if(bool(outRMSA) == true) {
        if(outRMSA < rms) {
        normA = sqrt(outRMSA / rms);
         
        } else {
            normA = sqrt(rms / outRMSA);
        }
    }
    float normB = 0.5f;
    if(bool(outRMSB) == true) {
        if(outRMSB < rms) {
            normB = sqrt(outRMSB / rms);
        }  else {
            normB = sqrt(rms / outRMSB);
        }
    }
    
    
    outBuffer.applyGainRamp(0, 0, bs, lastNormA,normA);
    outBuffer.applyGainRamp(1, 0, bs, lastNormB,normB);
        
    lastNormA = normA;
    lastNormB = normB;
    
    gain->process(outBuffer);
}

    

//==============================================================================
bool AudioPluginProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* AudioPluginProcessor::createEditor()
{
    return new AudioPluginEditor (*this); //juce::GenericAudioProcessorEditor(*this);//
}

//==============================================================================
void AudioPluginProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    copyXmlToBinary(*treeState.copyState().createXml(), destData);
}

void AudioPluginProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data,sizeInBytes));
        if(xml.get() != nullptr && xml->hasTagName(treeState.state.getType())) {
            treeState.replaceState(juce::ValueTree::fromXml(*xml));
        }
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AudioPluginProcessor();
}

void AudioPluginProcessor::getSize(float Size) {

    if(std::abs(Size - lastSize) > epsilon) {
        gpu_convolution->setSize(Size * float(getSampleRate()));
        lastSize = Size;
    }
}

void AudioPluginProcessor::run()
{
    while (!threadShouldExit())
    {
        int availableSamples = audioFifo_to_GPU.getNumReady();
        if (availableSamples >= 1024 * 2)
        {
            int start1, size1, start2, size2;
            
            audioFifo_to_GPU.prepareToRead(1024, start1, size1, start2, size2);
            
          
            if (size1 > 0)
            {
                sliceBuf.clear();
                sliceBuf.copyFrom(0, 0, fifoInputBuffer, 0, start1, size1);
                sliceBuf.copyFrom(1, 0, fifoInputBuffer, 1, start1, size1);
                sliceBuf.copyFrom(2, 0, fifoInputBuffer, 2, start1, size1);
                sliceBuf.copyFrom(3, 0, fifoInputBuffer, 3, start1, size1);
            }

            // Handle wraparound
            if (size2 > 0)
            {
                sliceBuf.addFrom(0, size1, fifoInputBuffer, 0, start2, size2);
                sliceBuf.addFrom(1, size1, fifoInputBuffer, 1, start2, size2);
                sliceBuf.addFrom(2, size1, fifoInputBuffer, 2, start2, size2);
                sliceBuf.addFrom(3, size1, fifoInputBuffer, 3, start2, size2);
            }
            audioFifo_to_GPU.finishedRead(size1 + size2);
           
            auto in_A = sliceBuf.getReadPointer(0);
            auto in_B = sliceBuf.getReadPointer(1);
            auto in_C = sliceBuf.getReadPointer(2);
            auto in_D = sliceBuf.getReadPointer(3);
            auto out_A = sliceBuf.getWritePointer(0);
            auto out_B = sliceBuf.getWritePointer(1);

            gpu_convolution->process(in_A, in_B, in_C, in_D, out_A, out_B);

            // Copy processed data into output FIFO
            fifoOutputBuffer.copyFrom(0, start1, sliceBuf, 0, 0, size1 + size2);
            fifoOutputBuffer.copyFrom(1, start1, sliceBuf, 1, 0, size1 + size2);
            audioFifo_from_GPU.finishedWrite(size1 + size2);
        }
    }
}
