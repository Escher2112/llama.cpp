// test_hopfield.cpp — sanity test for HopfieldPredictor.
//
// Loads a .bin produced by scripts/export_hopfield_to_bin.py, runs retrieve()
// with a deterministic synthetic query, prints the top-16 expert distribution
// for the first stored layer. Cross-check by running the Python reference at
// scripts/probe_l2_hopfield.py with the same input vector.

#include "hopfield.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <hopfield.bin> [layer]\n", argv[0]);
        return 1;
    }
    moe_orch::HopfieldPredictor h;
    if (!h.load(argv[1])) {
        std::fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    std::printf("Loaded: n_patterns=%u hidden_dim=%u n_experts=%u n_layers=%u dtype=%s beta=%.3f\n",
                h.n_patterns(), h.hidden_dim(), h.n_experts(), h.n_layers(),
                h.dtype() == moe_orch::HFLD_DTYPE_FP16 ? "fp16" : "fp32",
                h.beta());

    // Deterministic synthetic query: x[i] = sin(i*0.01).
    std::vector<float> q(h.hidden_dim());
    for (uint32_t i = 0; i < h.hidden_dim(); i++) {
        q[i] = std::sin((float)i * 0.01f);
    }

    int32_t layer = (argc >= 3) ? std::atoi(argv[2]) : -1;
    if (layer < 0) {
        // Pick the first layer slot's physical index. Since the API is
        // by-physical-layer, we just probe a likely common layer index from
        // the export.
        for (int32_t L = 0; L < 100; L++) {
            if (h.has_layer(L)) { layer = L; break; }
        }
    }
    if (layer < 0) {
        std::fprintf(stderr, "no usable layer found\n");
        return 1;
    }

    constexpr int32_t TOP_N = 16;
    std::vector<int32_t> top_idx(TOP_N);
    std::vector<float>   top_w(TOP_N);
    if (!h.predict_top_n(layer, q.data(), TOP_N, top_idx.data(), top_w.data())) {
        std::fprintf(stderr, "predict_top_n failed for layer=%d\n", layer);
        return 1;
    }
    std::printf("Top-%d experts for layer=%d:\n", TOP_N, layer);
    for (int32_t i = 0; i < TOP_N; i++) {
        std::printf("  rank %2d: expert %3d  weight %.6f\n", i, top_idx[i], top_w[i]);
    }
    return 0;
}
