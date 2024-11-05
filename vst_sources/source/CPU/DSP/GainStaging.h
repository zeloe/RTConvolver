
 
#pragma once 
#include "JuceHeader.h"
#include "../pluginparamers/PluginParameters.h"

class Gain : public juce::Thread
{
public:
	Gain(juce::AudioProcessorValueTreeState& treeState) : parameters(treeState), juce::Thread("ParamThread") {
		getParams();
		startThread(Priority::normal);
	}
	~Gain() 
	{
		stopThread(2000);
	}

	void getParams() {
		gain = parameters.getRawParameterValue("param_Gain");


	}

	void prepare() {
		pole = 0.f;
		paramComp = 0.f;
	}

	void run() override {
		while (!threadShouldExit()) {
			if (std::abs(*gain - paramComp) > epsilon) {
				float newValue = *gain;
				float newParam = newValue * 0.01f;
				atomic_gain.store(newParam);
				paramComp = newValue;
			}
		}
	}

	void process(juce::AudioBuffer<float> bufferToProcess) {
		const int bs = bufferToProcess.getNumSamples();



		float fac = atomic_gain.load();
		float*const* outs = bufferToProcess.getArrayOfWritePointers();
		for (int channel = 0; channel < 2; ++channel) {
			float* buffer = outs[channel]; // Get the float pointer for each channel
			for (int sample = 0; sample < bs; ++sample) {
				new_Gain = fac + pole * 0.99;
				buffer[sample] *= new_Gain;
				pole = new_Gain;
			}
		}





	}

private:
	juce::AudioProcessorValueTreeState& parameters;
	std::atomic<float>* gain;
	std::atomic<float> atomic_gain;
	float pole = 0;
	float new_Gain = 0;
	const float epsilon = 0.0001f;  // Define a small tolerance value
	float paramComp = 0;




};

