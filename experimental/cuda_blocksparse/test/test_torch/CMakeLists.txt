cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(test_cpu)

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit found at: ${CUDAToolkit_ROOT}\n"
  "CUDA toolkit ROOT: ${CUDAToolkit_LIBRARY_ROOT}\n"
  "CUDA include directories: ${CUDAToolkit_INCLUDE_DIRS}\n"
  "CUDA version: ${CUDAToolkit_VERSION}")
# set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(test_cpu test_cpu.cpp)
target_link_libraries(test_cpu "${TORCH_LIBRARIES}")
set_property(TARGET test_cpu PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET test_cpu
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
