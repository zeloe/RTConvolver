# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Define Metal shaders
set(METAL_SHADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/shift_and_insert.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/convolution.metal
)

# Compile each Metal shader to an object file
set(METAL_OBJECTS)

foreach(SHADER ${METAL_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(OUTPUT_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.air) # Using .air as output format
    list(APPEND METAL_OBJECTS ${OUTPUT_OBJECT})

    add_custom_command(
        OUTPUT ${OUTPUT_OBJECT}
        COMMAND xcrun -sdk macosx metal -c ${SHADER} -o ${OUTPUT_OBJECT}
        DEPENDS ${SHADER}
        COMMENT "Compiling Metal shader ${SHADER} to ${OUTPUT_OBJECT}"
    )
endforeach()

# Create a Metal library from the compiled Metal object files
set(METAL_LIBRARY ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)

add_custom_command(
    OUTPUT ${METAL_LIBRARY}
    COMMAND xcrun -sdk macosx metallib ${METAL_OBJECTS} -o ${METAL_LIBRARY}
    DEPENDS ${METAL_OBJECTS}
    COMMENT "Linking Metal object files into default.metallib"
)

# Ensure that METAL_LIBRARY is built before the METAL target
add_custom_target(BuildMetalLibrary DEPENDS ${METAL_LIBRARY})

# Add the library
add_library(GPU_LIB STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/convengine.h
    ${CMAKE_CURRENT_SOURCE_DIR}/convengine.mm
)

# Make METAL depend on the Metal library custom target
add_dependencies(GPU_LIB BuildMetalLibrary)

# Set properties for Metal and C++ linking
set_target_properties(GPU_LIB PROPERTIES LINKER_LANGUAGE CXX)

# Include Metal C++ headers
target_include_directories(GPU_LIB PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp/Metal/Metal.hpp
    "${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp"
)

# Find and link Metal frameworks
find_library(METAL Metal)
find_library(METALKIT MetalKit)
find_library(FOUNDATION Foundation)
find_library(QUARTZCORE QuartzCore)
find_library(APPKIT AppKit)
find_library(COREGRAPHICS CoreGraphics)

# Link Metal frameworks to the library
target_link_libraries(GPU_LIB PUBLIC
    stdc++
    "-framework Metal"
    "-framework MetalKit"
    "-framework AppKit"
    "-framework Foundation"
    "-framework CoreGraphics"
    "-framework QuartzCore"
)
# Set the path for the Metal library
set(METAL_LIBRARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/default.metallib")
# Add a compile definition for the Metal library path
target_compile_definitions(GPU_LIB PUBLIC METAL_LIBRARY_PATH="${METAL_LIBRARY_PATH}")

