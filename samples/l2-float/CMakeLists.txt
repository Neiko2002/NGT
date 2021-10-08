add_executable(l2-float l2-float.cpp)
target_link_libraries(l2-float PRIVATE ngtstatic)

add_executable(l2-float_evaluation l2-float_evaluation.cpp)
target_link_libraries(l2-float_evaluation PRIVATE ngtstatic)