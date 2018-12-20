cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_CUDA_COMPILER  /usr/local/cuda-9.2/bin/nvcc)

project(train)
include(FindCUDA)

set(CMAKE_CXX_STANDARD 14)
find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)

include_directories( ${OpenCV_INCLUDE_DIRS} )

cuda_add_executable(train train.cu)
target_link_libraries( train ${OpenCV_LIBS} )

