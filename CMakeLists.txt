cmake_minimum_required(VERSION 3.5)
project(bilin_test)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set (CUDA_NVCC_FLAGS "${CUD
set (CUDA_PROPAGATE_HOST_FLAGS ON)

cuda_add_executable(bilin_test main.cpp call_kernel.h call_kernel.cu input_data.h input_data.cpp)

target_link_libraries(bilin_test opencv_core opencv_imgcodecs opencv_imgproc)
