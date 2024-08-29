cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(pyprojections LANGUAGES CXX)

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

find_package(OpenMP)

find_package(pybind11 CONFIG REQUIRED)

# add_subdirectory(extern/pybind11)
add_library(core INTERFACE)
target_include_directories(core INTERFACE src)
target_link_libraries(core INTERFACE Eigen3::Eigen OpenMP::OpenMP_CXX)

pybind11_add_module(_C src/ext.cpp)
target_link_libraries(_C PUBLIC core Eigen3::Eigen)
install(TARGETS _C DESTINATION pyprojections)