cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(pyprojections LANGUAGES CXX)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
FetchContent_MakeAvailable(Eigen)

# find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(extern/pybind11)

add_library(core STATIC
    src/core.h
    src/project_pinhole.cpp
    src/project_spherical.cpp
)

target_link_libraries(core PUBLIC Eigen3::Eigen)

# target_compile_options(core -Wall -Werror)
pybind11_add_module(_C src/ext.cpp)
target_link_libraries(_C PRIVATE core)

install(TARGETS _C DESTINATION pyprojections)