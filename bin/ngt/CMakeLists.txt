add_executable(ngt_exe ngt.cpp)
target_link_libraries(ngt_exe PRIVATE ngtstatic)

install(TARGETS ngt_exe RUNTIME DESTINATION bin)
