set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR i686)

# Base CTC Directory
set(CTC_DIR "/home/steven/Documents/NAO6-development/ctc-linux64-atom-2.8.5.10")

# Path to Cross Compilers (inside i686-sbr-linux)
set(CROSS_BIN_DIR "${CTC_DIR}/yocto-sdk/sysroots/x86_64-naoqisdk-linux/usr/bin/i686-sbr-linux")

set(CMAKE_C_COMPILER "${CROSS_BIN_DIR}/i686-sbr-linux-gcc")
set(CMAKE_CXX_COMPILER "${CROSS_BIN_DIR}/i686-sbr-linux-g++")

# Target Sysroot (core2-32-sbr-linux)
set(SYSROOT_DIR "${CTC_DIR}/yocto-sdk/sysroots/core2-32-sbr-linux")
set(CMAKE_SYSROOT "${SYSROOT_DIR}")
set(CMAKE_FIND_ROOT_PATH "${SYSROOT_DIR}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)