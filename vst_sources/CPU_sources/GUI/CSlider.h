
#include "JuceHeader.h"
#include "GUIDefines.h"
#include "CLookAndFeel.h"
class ZenSlider : public juce::Slider {
public:
	ZenSlider(juce::Colour color) 
	{
		lnf = std::make_unique<ZenLook>(color);
		setLookAndFeel(lnf.get());
        
        
       
        
        
	}

	~ZenSlider() override
	{
	setLookAndFeel(nullptr);
    }
    
   



private:
   
	std::unique_ptr<ZenLook> lnf;
};
