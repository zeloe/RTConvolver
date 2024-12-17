#pragma once
#include "JuceHeader.h"
#include <cmath>
#include <vector>

class Limiter
{
public:
    Limiter() = default;
    ~Limiter() = default;

    void prepare(float sampleRate, float attackMs = 1.0f, float releaseMs = 5000.0f,
        float smoothingAttackMs = 10.0f, float smoothingReleaseMs = 50.0f,
        float thresholdDb = -1.0f, int delaySamples = 5)
    {
        // Convert time constants from milliseconds to coefficients
        attackCoef = expf(-1.0f / (sampleRate * attackMs * 0.001f));
        releaseCoef = expf(-1.0f / (sampleRate * releaseMs * 0.001f));
        smoothingAttackCoef = expf(-1.0f / (sampleRate * smoothingAttackMs * 0.001f));
        smoothingReleaseCoef = expf(-1.0f / (sampleRate * smoothingReleaseMs * 0.001f));

        // Threshold on linear scale
        lt = powf(10.f, thresholdDb / 20.f);

        // Initialize delay buffer
        delayBuffer.clear();
        delayBuffer.resize(delaySamples, 0.0f);

        // Reset state variables
        xpeak = 0.f;
        g = 1.f;
        writeIndex = 0;
        readIndex = delaySamples - 1;
    }

    void process(const float* input, float* output, const int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i) {
            float a = fabs(input[i]);
            g = 1.f;
            if (a > lt) {
                // Update xpeak (peak measurement filter)
                float coef = (a > xpeak) ? attackCoef : releaseCoef;
                xpeak = (1.f - coef) * xpeak + coef * a;

                // Compute f based on threshold
                float f = fmin(1.f, lt / xpeak);

                // Update gain g (smoothing filter)
                coef = (f < g) ? smoothingAttackCoef : smoothingReleaseCoef;
                g = (1.f - coef) * g + coef * f;
            }
      

            // Apply gain to delayed sample
            output[i] = g * input[i];
        }
    }

   

private:
    float attackCoef = 0.02f;       // Coefficient for peak attack
    float releaseCoef = 0.01f;      // Coefficient for peak release
    float smoothingAttackCoef = 0.3f; // Coefficient for gain smoothing attack
    float smoothingReleaseCoef = 0.01f; // Coefficient for gain smoothing release
    float lt = 1.f;                // Threshold (linear scale)
    float xpeak = 0.f;             // Peak level
    float g = 1.f;                 // Gain

    std::vector<float> delayBuffer; // Delay line buffer
    int writeIndex = 0;             // Write position in delay buffer
    int readIndex = 0;              // Read position in delay buffer
};
 