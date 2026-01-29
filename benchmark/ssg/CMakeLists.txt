cmake_minimum_required(VERSION 3.19)

project(ssg_benchmark)

set(SSG_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Boost REQUIRED)

# Benchmark executable
add_executable(ssg_build_and_test
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ssg_build_and_test.cpp)

# Link against the main library (defined in root)
target_link_libraries(ssg_build_and_test PUBLIC ssg OpenMP::OpenMP_CXX Boost::headers)

target_include_directories(ssg_build_and_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include)

target_compile_definitions(ssg_build_and_test PUBLIC
    "DATA_PATH=\"${DATA_PATH}\"")

