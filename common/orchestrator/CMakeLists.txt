# common/orchestrator/CMakeLists.txt
# Predictive expert prefetch orchestrator for MoE inference.
# Built only when -DLLAMA_ORCHESTRATOR=ON is passed at top-level configure.

add_library(moe_orchestrator STATIC
    predictor.cpp
    cache.cpp
    orchestrator.cpp
    mode_b.cpp
)

target_include_directories(moe_orchestrator
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

# We need C++17 for structured bindings + fold expressions in our code.
target_compile_features(moe_orchestrator PUBLIC cxx_std_17)

# Position-independent code: required because libllama (a shared lib when
# BUILD_SHARED_LIBS=ON) links us in when LLAMA_ORCHESTRATOR=ON.
set_target_properties(moe_orchestrator PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Depends on ggml-base for the eval-callback bridge (ggml_get_name,
# ggml_backend_tensor_get, etc.). No external runtime deps beyond that —
# weight access is mmap'd and inference is naive scalar GEMMs.
target_link_libraries(moe_orchestrator PUBLIC ggml-base)

# Optional sanity-test executable.
add_executable(orchestrator-test-predict test_predict.cpp)
target_link_libraries(orchestrator-test-predict PRIVATE moe_orchestrator)
