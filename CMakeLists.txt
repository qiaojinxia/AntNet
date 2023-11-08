cmake_minimum_required(VERSION 3.26)
project(AntNet)

set(CMAKE_CXX_STANDARD 17)

add_executable(AntNet main.cpp
        src/Device.h
        src/Device.cpp
        src/Layer.h
        src/Layer.cpp
        src/Tensor.h
        src/Tensor.cpp
        src/Optimizer.h
        src/Optimizer.cpp)
