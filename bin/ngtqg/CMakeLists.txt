add_executable(ngtqg_exe ngtqg.cpp)
set_target_properties(ngtqg_exe PROPERTIES OUTPUT_NAME ngtqg) 
target_link_libraries(ngtqg_exe PRIVATE ngtstatic)

install(TARGETS ngtqg_exe RUNTIME DESTINATION bin)
