# Add CPU source files to the list of sources for the executable
file(GLOB_RECURSE CPU_SOURCES CONFIGURE_DEPENDS source/*.cpp source/*.h)

# Add CUDA source files to the list of sources
file(GLOB CU_SOURCES CONFIGURE_DEPENDS source/CPU/GPU/*.cu source/CPU/GPU/*.cuh)

# Add CPU sources to the target
target_sources(${TARGET_NAME} PRIVATE ${CPU_SOURCES})

  

# Compile CUDA source files into a static library
add_library(CU_LIB STATIC ${CU_SOURCES})
set_property(TARGET CU_LIB PROPERTY CUDA_ARCHITECTURES native)
 
set_target_properties(CU_LIB PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET CU_LIB PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
 # Ensure CU_LIB is compiled with nvcc
set_target_properties(CU_LIB PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_source_files_properties(${CU_SOURCES} PROPERTIES LANGUAGE CUDA)