
#include "JuceHeader.h"
#include "GUIDefines.h"
#include "CLookAndFeel.h"
class SizeMenu : public juce::Component, 
public juce::ComboBox::Listener,
public juce::ValueTree::Listener
{
public:
    SizeMenu(juce::ValueTree & state, const juce::Identifier & parameterID, AudioPluginProcessor& processor)
        : stateTree(state), paramID(parameterID), proc(processor)
    {
        // Initialize combo box with items
        int counter = 1;
        for (auto& sizes : sizesLabel) {
            sizesBox.addItem(sizes, counter++);
        }

        addAndMakeVisible(sizesBox);
        sizesBox.addListener(this);

        // Set LookAndFeel
        lnf = std::make_unique<ZenLook>(outLineColor);
        setLookAndFeel(lnf.get());

        // Attach ValueTree Listener
        stateTree.addListener(this);

        // Initialize the combo box from the state
        int initialValue = stateTree.getProperty(paramID, 0);

       
        sizesBox.setSelectedItemIndex(initialValue, juce::dontSendNotification);
        auto selectedText = sizesBox.getText();
        float resSize = selectedText.getFloatValue();
        proc.getSize(resSize);
    }

    ~SizeMenu() override
    {
        sizesBox.removeListener(this);
        stateTree.removeListener(this);
        setLookAndFeel(nullptr);
    }

    void resized() override { sizesBox.setBounds(getLocalBounds()); }

    void comboBoxChanged(juce::ComboBox * comboBoxThatHasChanged) override
    {
        if (comboBoxThatHasChanged == &sizesBox)
        {
            auto selectedId = sizesBox.getSelectedId();
            
            stateTree.setProperty(paramID, selectedId - 1, nullptr); // Update the state
            // Retrieve the text value of the currently selected item
            auto selectedText = sizesBox.getText();
            float resSize = selectedText.getFloatValue();
          
            proc.getSize(resSize);
        }
    }

    void valueTreePropertyChanged(juce::ValueTree & tree, const juce::Identifier & property) override
    {
        if (tree == stateTree && property == paramID)
        {
            int newValue = stateTree.getProperty(paramID, 0);
            sizesBox.setSelectedItemIndex(newValue, juce::dontSendNotification);
         
        }
    }

private:
    juce::ValueTree & stateTree;
    juce::Identifier paramID;
    juce::ComboBox sizesBox;
    std::unique_ptr<ZenLook> lnf;
    AudioPluginProcessor& proc;
};
