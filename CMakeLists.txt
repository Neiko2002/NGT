cmake_minimum_required(VERSION 3.16)

project(ngt)

file(STRINGS "VERSION" ngt_VERSION)
message(STATUS "VERSION: ${ngt_VERSION}")
string(REGEX MATCH "^[0-9]+" ngt_VERSION_MAJOR ${ngt_VERSION})

set(ngt_VERSION ${ngt_VERSION})
set(ngt_SOVERSION ${ngt_VERSION_MAJOR})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)
string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_LOWER)
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_BUILD_TYPE_LOWER: ${CMAKE_BUILD_TYPE_LOWER}")



option(NGT_SHARED_MEMORY_ALLOCATOR "enable shared memory" OFF)
option(NGT_OPENMP_DISABLED "OpenMP" OFF)
option(NGT_MARCH_NATIVE_DISABLED "march" OFF)
option(NGT_AVX_DISABLED "AVX" OFF)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

if(NGT_AVX_DISABLED)
    message(STATUS "AVX will not be used to compute distances.")
endif()

if(NGT_OPENMP_DISABLED)
    message(STATUS "OpenMP is disabled.")
else()
    find_package(OpenMP REQUIRED COMPONENTS C CXX)
    if (OPENMP_FOUND)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

find_package(Threads REQUIRED)



add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(samples)