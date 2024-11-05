
#include "JuceHeader.h"
#include "GUIDefines.h"
#ifndef ZENLOOK_H
#define ZENLOOK_H


class ZenLook : public juce::LookAndFeel_V4 {
public:
	ZenLook() {};
    ZenLook(juce::Colour interiorColour)
    {
        this->interiorColour = interiorColour;
    };
	~ZenLook() override {} ;

	void drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height, float sliderPos,
		const float rotaryStartAngle, const float rotaryEndAngle, juce::Slider& slider) override
	{

            const juce::Colour outLineColor = juce::Colour(241, 242, 242);
            const juce::Colour backgroundColor = juce::Colour(35, 31, 32);
            const juce::Colour startColour = juce::Colour(128, 128, 128).brighter(); // Gray color
            const juce::Colour endColour = juce::Colour(255, 0, 0).brighter(); // Red color
         
            if (slider.isMouseOverOrDragging())
            {
                outline = outLineColor;
                fill = backgroundColor;
            }
            else
            {
                outline = backgroundColor;
                fill = outLineColor;

            }
           

            // Ensure the bounds are square
            auto size = juce::jmin(width, height);
            auto bounds = juce::Rectangle<float>(x, y, size, size).reduced(10);
            auto radius = bounds.getWidth() / 2.0f;
            auto centreX = bounds.getCentreX();
            auto centreY = bounds.getCentreY();
            auto lineW = juce::jmin(8.0f, radius * 0.5f);
            auto arcRadius = radius - lineW * 0.5f;

            // Calculate the end angle for the arc
            auto endAngle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);

           
            // Draw the arc with gradient
            for (float i = 0.0f; i <= sliderPos; i += 0.01f)
            {
                float angle = rotaryStartAngle + i * (rotaryEndAngle - rotaryStartAngle);
                Colour colour = startColour.interpolatedWith(interiorColour.darker(), i);
                juce::Path segment;
                segment.addCentredArc(centreX, centreY, arcRadius, arcRadius, 0.0f, angle, angle + 0.05f, true);
                g.setColour(colour);
                g.strokePath(segment, juce::PathStrokeType(lineW));
            }

            // Fill the interior
            g.setColour(interiorColour);
            g.fillEllipse(bounds.reduced(lineW));
           

            

            // Outline the ellipse
            g.setColour(outline);
            g.drawEllipse(bounds, 1.0f);

            // Draw the pointer
            g.setColour(outline);
            auto thumbWidth = lineW * 0.75f;

               
    }






private:
    juce::Colour fill, outline;
    juce::Colour interiorColour;





};

#endif // !ZENLOOK_H