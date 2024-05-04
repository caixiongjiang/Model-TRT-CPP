cmake_minimum_required(VERSION 3.10)
project(Model Infer)
set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_library(class_model SHARED
    src/model_infer.cpp
)