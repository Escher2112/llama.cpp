// test_predict.cpp — minimal sanity test for MLPPredictor.
//
// Loads predictor_mlp_v1.bin, runs predict for (layer=0, horizon=1) on a
// deterministic synthetic hidden state, prints the top-16 expert IDs.
// Compare against the Python equivalent (scripts/test_predictor.py).
//
// Build target added inline in CMakeLists.txt of this directory.

#include "predictor.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <predictor.bin>\n", argv[0]);
        return 1;
    }
    moe_orch::MLPPredictor predictor;
    if (!predictor.load(argv[1])) {
        std::fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    std::printf("Loaded: hidden_dim=%u, hidden_units=%u, n_experts=%u, num_heads=%u, dtype=%s\n",
                predictor.hidden_dim(), predictor.hidden_units(),
                predictor.n_experts(), predictor.num_heads(),
                predictor.dtype() == moe_orch::PRED_DTYPE_FP16 ? "fp16" : "fp32");

    // Deterministic synthetic hidden state: x[i] = sin(i * 0.01) so output
    // is stable + reproducible across languages.
    std::vector<float> hidden(predictor.hidden_dim());
    for (uint32_t i = 0; i < predictor.hidden_dim(); i++) {
        hidden[i] = std::sin((float)i * 0.01f);
    }

    constexpr int32_t TOP_K = 16;
    std::vector<int32_t> top_indices(TOP_K);
    bool ok = predictor.predict_top_k(/*source_layer=*/0,
                                      /*horizon=*/1,
                                      hidden.data(),
                                      TOP_K,
                                      top_indices.data());
    if (!ok) {
        std::fprintf(stderr, "predict_top_k failed for (layer=0, horizon=1)\n");
        return 1;
    }

    std::printf("Top-%d experts for (layer=0, horizon=1): [", TOP_K);
    for (int32_t i = 0; i < TOP_K; i++) {
        std::printf("%d%s", top_indices[i], i == TOP_K - 1 ? "" : ", ");
    }
    std::printf("]\n");

    return 0;
}
