cmake_minimum_required(VERSION 3.23)
project(nao6_custom_module CXX)
set(CMAKE_CXX_STANDARD 17)
# Update PATH to SDK where located on your computer.
set(NAOQI_SDK_DIR "/home/steven/Documents/NAO6-development/naoqi-sdk-2.8.5.10-linux64")
# FIX: Force the compiler to use the legacy C++ string ABI to match the SDK pre-compiled binaries
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
# Include header paths
include_directories(${NAOQI_SDK_DIR}/include)
# Find your local system's thread and socket dependencies
find_package(Threads REQUIRED)
# Reference the SDK precompiled libraries explicitlly
link_directories(${NAOQI_SDK_DIR}/lib)
# Change to "add_library()" to generate a .so file for the module to run on the NAO6 Robot
add_executable(nao6_custom_module main.cpp)
# Link against the key core NAOqi components inside SDK folder
target_link_libraries(nao6_custom_module 
    ${NAOQI_SDK_DIR}/lib/libqi.so
    ${NAOQI_SDK_DIR}/lib/libalcommon.so
    ${NAOQI_SDK_DIR}/lib/libboost_system.so.1.59.0
    ${NAOQI_SDK_DIR}/lib/libboost_thread.so.1.59.0
    Threads::Threads
)