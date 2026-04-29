// orchestrator.cpp — top-level Orchestrator implementation.

#include "orchestrator.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

namespace moe_orch {

Orchestrator::Orchestrator(const OrchestratorConfig &  config,
                           const std::string &         predictor_mlp_bin_path,
                           const std::string &         /*predictor_hopfield_bin_path*/,
                           int32_t                     n_layers,
                           int32_t                     n_experts_per_layer,
                           int32_t                     hidden_dim)
    : config_(config),
      n_layers_(n_layers),
      n_experts_per_layer_(n_experts_per_layer),
      hidden_dim_(hidden_dim),
      cache_(n_layers,
             n_experts_per_layer,
             config.l1_capacity,
             config.l2_capacity,
             config.shadow_mode)
{
    if (!l1_predictor_.load(predictor_mlp_bin_path.c_str())) {
        std::fprintf(stderr, "[orchestrator] failed to load MLP predictor from %s\n",
                     predictor_mlp_bin_path.c_str());
    }

    // L2 Hopfield predictor: stub for v0 — wire up once shadow run validates L1 path.
    // l2_predictor_available_ stays false; _refresh_l2() is a no-op.
    // TODO(day-6): instantiate HopfieldPredictor and load from predictor_hopfield_bin_path.

    embedding_window_.reserve(config.embedding_window_tokens);
    scratch_top_.resize(config.l1_prefetch_top_k);
}

double Orchestrator::l1_predictor_avg_ms() const {
    if (l1_predictor_calls_ == 0) return 0.0;
    return (l1_predictor_total_ns_ / (double)l1_predictor_calls_) / 1.0e6;
}

double Orchestrator::l2_predictor_avg_ms() const {
    if (l2_predictor_calls_ == 0) return 0.0;
    return (l2_predictor_total_ns_ / (double)l2_predictor_calls_) / 1.0e6;
}

void Orchestrator::on_layer_input(int32_t layer, const float * hidden_state) {
    // ---- conversation embedding update (only at the configured layer) ----
    if (layer == config_.embedding_layer) {
        std::vector<float> snapshot(hidden_state, hidden_state + hidden_dim_);
        if ((int32_t)embedding_window_.size() >= config_.embedding_window_tokens) {
            embedding_window_.erase(embedding_window_.begin());
        }
        embedding_window_.push_back(std::move(snapshot));
        tokens_since_l2_refresh_++;

        if (!l2_initialized_ ||
            tokens_since_l2_refresh_ >= config_.l2_refresh_every_tokens) {
            _refresh_l2();
            tokens_since_l2_refresh_ = 0;
            l2_initialized_ = true;
        }
    }

    // ---- L1 prefetch: predict experts for layers (layer + horizon_i) ----
    for (int32_t h = 0; h < config_.num_prefetch_horizons; h++) {
        int32_t M = config_.prefetch_horizons[h];
        if (M <= 0) continue;
        int32_t target_layer = layer + M;
        if (target_layer >= n_layers_) continue;

        auto t0 = std::chrono::steady_clock::now();
        bool ok = l1_predictor_.predict_top_k(layer,
                                              M,
                                              hidden_state,
                                              config_.l1_prefetch_top_k,
                                              scratch_top_.data());
        auto t1 = std::chrono::steady_clock::now();
        l1_predictor_total_ns_ +=
            (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        l1_predictor_calls_++;

        if (ok) {
            cache_.prefetch_to_l1(target_layer,
                                  scratch_top_.data(),
                                  config_.l1_prefetch_top_k);
        }
    }
}

void Orchestrator::on_routing(int32_t layer, const int32_t * indices, int32_t count) {
    for (int32_t i = 0; i < count; i++) {
        cache_.access(layer, indices[i]);
    }
}

void Orchestrator::_refresh_l2() {
    // v0 stub: when HopfieldPredictor is wired up, this computes the running
    // conversation embedding (mean of embedding_window_), queries Hopfield for
    // each layer's top-N experts, and prefetches them into L2.
    //
    // Until then, L2 only fills via L1 evictions. That gives us a "natural"
    // L2 set based on what was hot recently — not optimal, but correct.
    if (!l2_predictor_available_) return;

    // (Not yet implemented — see TODO in ctor.)
    l2_predictor_calls_++;
}

} // namespace moe_orch


// ============================================================================
// ggml eval callback bridge
// ============================================================================
//
// llama_context_params::cb_eval fires twice per tensor during sched compute:
//   - ask=true  → "do you want this tensor's data later?" return true to opt in
//   - ask=false → tensor is computed; you can read t->data
//
// We filter for tensors named "ffn_moe_topk-N" — these are the per-layer
// top-K expert index tensors produced by `ggml_argsort_top_k` in
// llama-graph.cpp:1453. Data is int32, shape (n_expert_used, n_tokens).

extern "C" bool ggml_eval_callback_orchestrator(
    struct ggml_tensor * t,
    bool                 ask,
    void *               user_data) {

    // The tensor name is set by `cb(selected_experts, "ffn_moe_topk", il)` →
    // ggml_format_name → "ffn_moe_topk-N".
    const char * name = ggml_get_name(t);
    if (!name || std::strncmp(name, "ffn_moe_topk-", 13) != 0) {
        return false;  // not a tensor we care about
    }

    if (ask) {
        return true;   // yes, please call us again with ask=false post-compute
    }

    // Post-compute path. Read the int32 expert indices.
    auto * orch = static_cast<moe_orch::Orchestrator *>(user_data);
    if (!orch) return false;

    // Parse layer index from suffix.
    int32_t layer = std::atoi(name + 13);

    // Tensor shape: (n_expert_used, n_tokens) per the ggml_argsort_top_k call.
    // Element type is I32. Total elements = n_expert_used * n_tokens.
    const int32_t n_elements = (int32_t)ggml_nelements(t);
    if (n_elements <= 0) return false;

    // Tensor data may be on a backend buffer (GPU). Use ggml_backend_tensor_get
    // to copy to a host buffer.
    std::vector<int32_t> host(n_elements);
    ggml_backend_tensor_get(t, host.data(), 0, n_elements * sizeof(int32_t));

    orch->on_routing(layer, host.data(), n_elements);
    return true;
}

