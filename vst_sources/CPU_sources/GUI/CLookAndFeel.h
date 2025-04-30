
#include "JuceHeader.h"
#include "GUIDefines.h"
#ifndef ZENLOOK_H
#define ZENLOOK_H


class ZenLook : public juce::LookAndFeel_V4 {
public:
	ZenLook() {}
    ZenLook(juce::Colour interiorColour)
    {
        this->interiorColour = interiorColour;
    }
	~ZenLook() override {}

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
           

               
    }

    void drawComboBox(juce::Graphics& g, int width, int height, bool,
        int, int, int, int, juce::ComboBox& box) override
    {
        const juce::Colour backgroundColor = juce::Colour(35, 31, 32); // Same as slider background
        const juce::Colour outlineColor = juce::Colour(241, 242, 242); // Same as slider outline
        const juce::Colour arrowColor = juce::Colour(128, 128, 128).brighter(); // Similar to start color

        auto cornerSize = 5.0f; // Rounded corners
        juce::Rectangle<int> boxBounds(0, 0, width, height);

        // Fill the background
        g.setColour(backgroundColor);
        g.fillRoundedRectangle(boxBounds.toFloat(), cornerSize);

        // Draw the outline
        g.setColour(outlineColor);
        g.drawRoundedRectangle(boxBounds.toFloat().reduced(0.5f), cornerSize, 1.5f);

        // Draw a subtle gradient inside the box
        juce::ColourGradient gradient(juce::Colour(128, 128, 128).brighter(), 0, 0,
            juce::Colour(255, 0, 0).brighter(), width, height, false);
        g.setGradientFill(gradient);
        g.fillRoundedRectangle(boxBounds.reduced(2).toFloat(), cornerSize);

        // Draw the dropdown arrow
        juce::Rectangle<int> arrowZone(width - 30, 0, 20, height);
        juce::Path path;
        path.startNewSubPath((float)arrowZone.getX() + 3.0f, (float)arrowZone.getCentreY() - 2.0f);
        path.lineTo((float)arrowZone.getCentreX(), (float)arrowZone.getCentreY() + 3.0f);
        path.lineTo((float)arrowZone.getRight() - 3.0f, (float)arrowZone.getCentreY() - 2.0f);

        g.setColour(arrowColor.withAlpha(box.isEnabled() ? 0.9f : 0.2f));
        g.strokePath(path, juce::PathStrokeType(2.0f));
    }

    void drawPopupMenuItem(juce::Graphics& g,
        const juce::Rectangle<int>& area,
        bool isSeparator,
        bool isActive,
        bool isHighlighted,
        bool isTicked,
        bool hasSubMenu,
        const juce::String& text,
        const juce::String& shortcutKeyText,
        const juce::Drawable* icon,
        const juce::Colour* textColourToUse) override
    {
        const juce::Colour backgroundColor = juce::Colour(35, 31, 32); // Same as slider background
        const juce::Colour hoverColor = juce::Colour(128, 128, 128).brighter(); // Highlighted background
        const juce::Colour textColor = juce::Colour(241, 242, 242); // Same as slider outline
        const juce::Colour activeTextColor = juce::Colour(255, 0, 0).brighter(); // Red for ticked items

        auto bounds = area.reduced(2);

        // Draw background
        if (isHighlighted && isActive)
            g.setColour(hoverColor);
        else
            g.setColour(backgroundColor);

        g.fillRect(bounds);

        // Draw tick or submenu arrow
        if (isTicked)
        {
            g.setColour(activeTextColor);
            auto tickBounds = bounds.removeFromLeft(bounds.getHeight()).reduced(4);
            g.drawLine(tickBounds.getX(), tickBounds.getCentreY(),
                tickBounds.getRight(), tickBounds.getBottom(), 2.0f);
            g.drawLine(tickBounds.getX(), tickBounds.getCentreY(),
                tickBounds.getRight(), tickBounds.getY(), 2.0f);
        }

        if (hasSubMenu)
        {
            g.setColour(textColor);
            auto arrowBounds = bounds.removeFromRight(bounds.getHeight()).reduced(6);
            juce::Path arrowPath;
            arrowPath.addTriangle(arrowBounds.getX(), arrowBounds.getY(),
                arrowBounds.getRight(), arrowBounds.getCentreY(),
                arrowBounds.getX(), arrowBounds.getBottom());
            g.fillPath(arrowPath);
        }

        // Draw text
        g.setColour(textColourToUse != nullptr ? *textColourToUse : (isActive ? textColor : textColor.withAlpha(0.5f)));
        g.setFont(juce::Font(15.0f));
        g.drawText(text, bounds, juce::Justification::centredLeft, true);

        // Draw shortcut key text (if any)
        if (!shortcutKeyText.isEmpty())
        {
            g.setColour(textColor.withAlpha(isActive ? 1.0f : 0.5f));
            g.drawText(shortcutKeyText, bounds, juce::Justification::centredRight, true);
        }
    }

    void drawPopupMenuBackground(juce::Graphics& g, int width, int height) override
    {
        const juce::Colour backgroundColor = juce::Colour(35, 31, 32); // Same as slider background
        const juce::Colour outlineColor = juce::Colour(241, 242, 242); // Same as slider outline

        // Fill the background
        g.setColour(backgroundColor);
        g.fillRect(0, 0, width, height);

        // Draw an outline around the popup menu
        g.setColour(outlineColor);
        g.drawRect(0, 0, width, height, 1);
    }





private:
    juce::Colour fill, outline;
    juce::Colour interiorColour;





};

#endif // ZENLOOK_H
