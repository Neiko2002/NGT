configure_file(${CMAKE_CURRENT_SOURCE_DIR}/defines.h.in ${CMAKE_CURRENT_BINARY_DIR}/NGT/defines.h)

set(NGT_SOURCES
    ArrayFile.cpp
    Capi.cpp
    Command.cpp
    Graph.cpp
    Index.cpp
    Node.cpp
    SharedMemoryAllocator.cpp
    Thread.cpp
    Tree.cpp
    Version.cpp
    NGTQ/Capi.cpp
    NGTQ/NGTQGCommand.cpp)
file(GLOB HEADER_FILES *.h *.hpp)
file(GLOB NGTQ_HEADER_FILES NGTQ/*.h NGTQ/*.hpp)

if(NGT_SHARED_MEMORY_ALLOCATOR)
    list(APPEND NGT_SOURCES MmapManager.cpp)
endif()

add_library(ngtstatic STATIC ${NGT_SOURCES})
set_target_properties(ngtstatic PROPERTIES POSITION_INDEPENDENT_CODE on)
if(NOT NGT_OPENMP_DISABLED)
    target_link_libraries(ngtstatic PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
endif()
target_include_directories(ngtstatic PUBLIC $<BUILD_INTERFACE:${NGT_BUILD_INTERFACE_INCLUDE_DIR}>
                                            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<INSTALL_INTERFACE:include>)
target_compile_features(ngtstatic PUBLIC cxx_std_17)
if(NOT NGT_AVX_DISABLED)
    target_compile_options(ngtstatic PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
                                            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mavx2>)
endif()
if(NOT NGT_MARCH_NATIVE_DISABLED)
    target_compile_options(ngtstatic PUBLIC $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-march=native>)
endif()

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/NGT/version_defs.h"
    DEPENDS ${NGT_SOURCES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/NGT/
    COMMAND ${PROJECT_SOURCE_DIR}/utils/mk_version_defs_h.sh ${PROJECT_SOURCE_DIR} version_defs.h)
target_sources(ngtstatic PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/NGT/version_defs.h")

add_library(ngt SHARED ${NGT_SOURCES})
set_target_properties(ngt PROPERTIES VERSION ${ngt_VERSION})
set_target_properties(ngt PROPERTIES SOVERSION ${ngt_SOVERSION})
target_link_libraries(ngt PRIVATE ngtstatic)

include(GNUInstallDirs)
install(
    TARGETS ngt ngtstatic
    EXPORT ${PROJECT_NAME}_EXPORT_TARGETS
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
install(
    EXPORT ${PROJECT_NAME}_EXPORT_TARGETS
    NAMESPACE ${PROJECT_NAME}::
    FILE ${PROJECT_NAME}Config.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
install(FILES ${HEADER_FILES} ${CMAKE_CURRENT_BINARY_DIR}/defines.h ${CMAKE_CURRENT_BINARY_DIR}/version_defs.h
        DESTINATION include/NGT)
install(FILES ${NGTQ_HEADER_FILES} DESTINATION include/NGT/NGTQ)
