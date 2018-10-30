cmake_minimum_required(VERSION 3.9)
project(convolution)

set(CMAKE_CXX_STANDARD 11)

add_executable(conv conv.cpp)
add_executable(nn nn.cpp)
add_executable(ioexample ioexample.cpp)
