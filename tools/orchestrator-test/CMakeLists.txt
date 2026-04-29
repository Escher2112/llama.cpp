set(TARGET orchestrator-test)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE moe_orchestrator llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
