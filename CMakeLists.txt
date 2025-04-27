cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(RTConvolver LANGUAGES VERSION 0.0.1 CUDA CXX )



# IDEs:  Enable grouping of source files into folders in IDEs.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# IDEs:  Create a folder in the IDE with the JUCE Module code.
option(JUCE_ENABLE_MODULE_SOURCE_GROUPS "Show all module sources in IDE projects" ON)


set(LIB_JUCE_TAG "8.0.6")

include(FetchContent)

# Keep dependencies outside of the "Build" directory.
# This allows to do a clean build of the project without re-downloading or
# rebuilding the dependencies.
set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/Libs" CACHE PATH "External dependencies path." FORCE)

FetchContent_Declare(juce
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        ${LIB_JUCE_TAG}
    GIT_SHALLOW    TRUE
    GIT_CONFIG     advice.detachedHead=false    # Disable detached HEAD warning for fetching a specific tag
    SOURCE_DIR     "${FETCHCONTENT_BASE_DIR}/JUCE"
    SUBBUILD_DIR   "${FETCHCONTENT_BASE_DIR}/JUCE-Subbuild"
    BINARY_DIR     "${FETCHCONTENT_BASE_DIR}/JUCE-Build")

FetchContent_MakeAvailable(juce)

 
 
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-10")
endif()
set (TARGET_NAME ${PROJECT_NAME})
 
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS "ON")
 
juce_add_plugin(${TARGET_NAME}
    # VERSION ...                               # Set this if the plugin version is different to the project version
    # ICON_BIG ...                              # ICON_* arguments specify a path to an image file to use as an icon for the Standalone
    # ICON_SMALL ...
    COMPANY_NAME "zelo"                          # Specify the name of the plugin's author
    # IS_SYNTH TRUE/FALSE                       # Is this a synth or an effect?
    # NEEDS_MIDI_INPUT TRUE/FALSE               # Does the plugin need midi input?
    # NEEDS_MIDI_OUTPUT TRUE/FALSE              # Does the plugin need midi output?
    # IS_MIDI_EFFECT TRUE/FALSE                 # Is this plugin a MIDI effect?
    # EDITOR_WANTS_KEYBOARD_FOCUS TRUE/FALSE    # Does the editor need keyboard focus?
    # COPY_PLUGIN_AFTER_BUILD TRUE/FALSE        # Should the plugin be installed to a default location after building?
    PLUGIN_MANUFACTURER_CODE Zelo               # A four-character manufacturer id with at least one upper-case character
    PLUGIN_CODE Dem0                            # A unique four-character plugin id with exactly one upper-case character
                                                # GarageBand 10.3 requires the first letter to be upper-case, and the remaining letters to be lower-case
    FORMATS  VST3                   # The formats to build. Other valid formats are: AAX Unity VST AU AUv3
    PRODUCT_NAME ${TARGET_NAME}
    VST3_AUTO_MANIFEST FALSE
    JUCE_COPY_PLUGIN_AFTER_BUILD=ON)        # The name of the final executable, which can differ from the target name


target_compile_definitions(${TARGET_NAME}
    PRIVATE
     JUCE_VST3_CAN_REPLACE_VST2=0
        # JUCE_WEB_BROWSER and JUCE_USE_CURL would be on by default, but you might not need them.
        JUCE_WEB_BROWSER=0  # If you remove this, add `NEEDS_WEB_BROWSER TRUE` to the `juce_add_console_app` call
        JUCE_USE_CURL=0)    # If you remove this, add `NEEDS_CURL TRUE` to the `juce_add_console_app` call
# Set C++ standard
set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_STANDARD 17)



 
add_subdirectory(vst_sources)

#juce_enable_vst3_manifest_step(${TARGET_NAME})
target_link_libraries(${TARGET_NAME}
    PRIVATE
        juce::juce_core
        juce::juce_audio_utils
    	juce::juce_dsp
        CU_LIB
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags
)

juce_generate_juce_header(${TARGET_NAME})
