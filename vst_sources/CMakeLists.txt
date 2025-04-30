 
file(GLOB_RECURSE CPU_SOURCES CONFIGURE_DEPENDS CPU_sources/*.cpp CPU_sources/*.h)


# Add CPU sources to the target
target_sources(${TARGET_NAME} PRIVATE ${CPU_SOURCES})


if (CMAKE_SYSTEM_NAME MATCHES Windows)
	enable_language(CUDA)
	# Add CUDA source files to the list of sources
	file(GLOB CU_SOURCES CONFIGURE_DEPENDS CUDA_sources/*.cu CUDA_sources/*.cuh)


	# Compile CUDA source files into a static library
	add_library(GPU_LIB STATIC ${CU_SOURCES})
	#set_property(TARGET GPU_LIB PROPERTY CUDA_ARCHITECTURES native)
 
	set_target_properties(GPU_LIB PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	set_property(TARGET GPU_LIB PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
 
	set_target_properties(GPU_LIB PROPERTIES POSITION_INDEPENDENT_CODE ON)
	set_source_files_properties(${CU_SOURCES} PROPERTIES LANGUAGE CUDA)
else()
	add_subdirectory(METAL_sources)
endif()