cmake_minimum_required(VERSION 3.19)


project(ngt_benchmark)

set(SSG_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)

# Benchmark executable
add_executable(ngt_build_and_test
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ngt_build_and_test.cpp)



set_property(TARGET ngt_build_and_test PROPERTY CXX_STANDARD 17)


# Link against the main library (defined in root)
target_link_libraries(ngt_build_and_test PUBLIC ngtstatic)

if(WIN32)
    target_link_libraries(ngt_build_and_test PUBLIC Psapi)
endif()



target_include_directories(ngt_build_and_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include)

target_compile_definitions(ngt_build_and_test PUBLIC
    "DATA_PATH=\"${DATA_PATH}\"")

