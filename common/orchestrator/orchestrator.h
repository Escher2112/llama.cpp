// orchestrator.h — top-level Orchestrator that wires predictor + cache together.
//
// Exposes two callbacks consumed by the llama.cpp hooks:
//   on_layer_input(layer, hidden_state)  — fires the L1 predictor + prefetch
//   on_routing(layer, indices)           — observes actual routing, updates cache LRU
//
// In shadow mode the orchestrator only updates bookkeeping. In live mode it
// also dispatches actual cudaMemcpyAsync swap operations on a dedicated stream
// (TODO: live mode plug-in is its own follow-on after shadow validates).

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cache.h"
#include "predictor.h"

namespace moe_orch {

struct OrchestratorConfig {
    // Cache sizing (per layer).
    int32_t l1_capacity            = 32;        // experts in VRAM per layer
    int32_t l2_capacity            = 80;        // experts in RAM per layer

    // L1 (per-token) predictor behavior.
    // Single horizon (1) is the post-Day-1 tuning result — less cache churn,
    // similar hit rate vs the (1,2) horizon set we started with.
    int32_t prefetch_horizons[4]   = {1, 0, 0, 0};
    int32_t num_prefetch_horizons  = 1;
    int32_t l1_prefetch_top_k      = 16;

    // L2 (per-window) Hopfield predictor behavior.
    int32_t l2_prefetch_top_n      = 64;
    int32_t embedding_layer        = 12;        // mid-network input hidden state
    int32_t embedding_window_tokens= 16;
    int32_t l2_refresh_every_tokens= 16;

    // Mode.
    bool    shadow_mode            = true;
};

class Orchestrator {
public:
    // Loads predictor weights from .bin file. The Hopfield L2 predictor is
    // optional in v0 — if hopfield_path is null/empty, only L1 prefetch is
    // active (shadow mode) and L2 promotions only happen via L1 evictions.
    Orchestrator(const OrchestratorConfig &  config,
                 const std::string &         predictor_mlp_bin_path,
                 const std::string &         predictor_hopfield_bin_path,  // may be ""
                 int32_t                     n_layers,
                 int32_t                     n_experts_per_layer,
                 int32_t                     hidden_dim);

    // Hook callbacks (called from llama.cpp graph-build hooks).
    //
    // hidden_state: pointer to float buffer of size hidden_dim. The caller is
    //   responsible for copying out of any GPU-resident tensor first; this
    //   function expects CPU-resident floats.
    void on_layer_input(int32_t layer, const float * hidden_state);

    // indices: pointer to int32 buffer of length top_k * seq_len. The caller
    //   is responsible for reading back from any GPU-resident tensor.
    void on_routing(int32_t layer, const int32_t * indices, int32_t count);

    // Router weights: float buffer of length top_k * seq_len, paired 1:1 with
    //   the indices passed to the most recent on_routing call for this layer.
    //   Used for weight-aware eviction.
    void on_routing_weights(int32_t layer, const float * weights, int32_t count);

    // Diagnostics
    CacheStats cache_stats() const { return cache_.stats(); }
    uint64_t   l1_predictor_calls() const { return l1_predictor_calls_; }
    uint64_t   l2_predictor_calls() const { return l2_predictor_calls_; }
    double     l1_predictor_avg_ms() const;
    double     l2_predictor_avg_ms() const;

    // Test-only access
    const ThreeTierCache & cache() const { return cache_; }

private:
    OrchestratorConfig config_;
    int32_t            n_layers_;
    int32_t            n_experts_per_layer_;
    int32_t            hidden_dim_;

    MLPPredictor       l1_predictor_;
    bool               l1_predictor_loaded_   = false;
    // TODO: HopfieldPredictor l2_predictor_;  — v0 stub (pass-through), wire after shadow test
    bool               l2_predictor_available_ = false;

    ThreeTierCache     cache_;

    // Conversation embedding state (sliding window of recent layer-`embedding_layer` inputs).
    std::vector<std::vector<float>> embedding_window_;
    int32_t            tokens_since_l2_refresh_ = 0;
    bool               l2_initialized_          = false;

    // Stats
    uint64_t l1_predictor_calls_ = 0;
    uint64_t l2_predictor_calls_ = 0;
    double   l1_predictor_total_ns_ = 0.0;
    double   l2_predictor_total_ns_ = 0.0;

    // Reusable scratch — avoids per-call allocations.
    std::vector<int32_t> scratch_top_;     // size l1_prefetch_top_k

    // Per-layer buffer of the most recent topk indices we've seen. Populated
    // by on_routing, consumed by on_routing_weights to pair weights→experts.
    // ffn_moe_topk-N always fires before ffn_moe_weights-N for the same N.
    std::vector<std::vector<int32_t>> last_topk_indices_;

    // Refresh L2 cache by querying Hopfield with current conversation embedding.
    void _refresh_l2();
};

} // namespace moe_orch

// ----- ggml eval callback bridge -----
//
// Plug this into llama_context_params::cb_eval, with cb_eval_user_data set to
// a moe_orch::Orchestrator pointer. The callback filters for tensors named
// "ffn_moe_topk-N" (the routing decision per MoE layer N), reads the
// expert-index data after compute, and forwards to orch->on_routing.
//
// This sidesteps source-modifying qwen3moe.cpp / llama-graph.cpp entirely —
// purely a runtime wiring via the public llama API.

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

bool ggml_eval_callback_orchestrator(
    struct ggml_tensor * t,
    bool                 ask,
    void *               user_data);

#ifdef __cplusplus
}
#endif

