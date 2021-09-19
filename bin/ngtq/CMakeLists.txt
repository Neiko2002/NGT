add_executable(ngtq_exe ngtq.cpp)
set_target_properties(ngtq_exe PROPERTIES OUTPUT_NAME ngtq)
target_link_libraries(ngtq_exe PRIVATE ngtstatic)

install(TARGETS ngtq_exe RUNTIME DESTINATION bin)
