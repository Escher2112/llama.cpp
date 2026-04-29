// orchestrator.cpp — top-level Orchestrator implementation.

#include "orchestrator.h"

#include <algorithm>
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
    // Empty path = recency-only mode (Mode A): orchestrator observes routing
    // and runs the cache, but skips MLP prefetch entirely.
    if (predictor_mlp_bin_path.empty()) {
        l1_predictor_loaded_ = false;
        std::fprintf(stderr, "[orchestrator] recency-only mode (no MLP predictor)\n");
    } else if (l1_predictor_.load(predictor_mlp_bin_path.c_str())) {
        l1_predictor_loaded_ = true;
    } else {
        l1_predictor_loaded_ = false;
        std::fprintf(stderr, "[orchestrator] failed to load MLP predictor from %s\n",
                     predictor_mlp_bin_path.c_str());
    }

    // L2 Hopfield predictor: stub for v0 — wire up once shadow run validates L1 path.
    // l2_predictor_available_ stays false; _refresh_l2() is a no-op.
    // TODO(day-6): instantiate HopfieldPredictor and load from predictor_hopfield_bin_path.

    embedding_window_.reserve(config.embedding_window_tokens);
    scratch_top_.resize(config.l1_prefetch_top_k);
    last_topk_indices_.resize((size_t)n_layers);
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
    if (!l1_predictor_loaded_) return;
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
    if (layer < 0 || layer >= n_layers_) return;
    auto & buf = last_topk_indices_[layer];
    buf.assign(indices, indices + count);
    for (int32_t i = 0; i < count; i++) {
        cache_.access(layer, indices[i]);
    }
}

void Orchestrator::on_routing_weights(int32_t layer, const float * weights, int32_t count) {
    if (layer < 0 || layer >= n_layers_) return;
    const auto & buf = last_topk_indices_[layer];
    // Pair weights[i] with buf[i]. Sizes should match — if not, we silently
    // pair what we can; mismatches indicate either an arch we haven't seen
    // or a tokens>1 prefill where weights cover multiple tokens.
    const int32_t pair_count = (int32_t)std::min((size_t)count, buf.size());
    for (int32_t i = 0; i < pair_count; i++) {
        cache_.update_router_weight(layer, buf[i], weights[i]);
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

    const char * name = ggml_get_name(t);
    if (!name) return false;

    // Two prefixes we care about:
    //   ffn_moe_topk-N        → int32 indices (n_expert_used, n_tokens)
    //   ffn_moe_weights-N     → fp32 raw router weights, same shape (well, with
    //                           a leading 1: (1, n_expert_used, n_tokens))
    // We deliberately do NOT match ffn_moe_weights_softmax / _norm / _scaled —
    // their relative ordering is preserved through monotonic transforms, and
    // matching only "ffn_moe_weights-" (with the trailing dash) is exact.
    const bool is_topk    = std::strncmp(name, "ffn_moe_topk-",    13) == 0;
    const bool is_weights = std::strncmp(name, "ffn_moe_weights-", 16) == 0;
    // ffn_norm-N is the post-attention-norm output that feeds the router. This
    // matches the Python `Qwen3MoeSparseMoeBlock` pre-hook semantics — same
    // hidden state the predictor was trained against.
    const bool is_ffn_norm = std::strncmp(name, "ffn_norm-", 9) == 0;
    if (!is_topk && !is_weights && !is_ffn_norm) return false;

    if (ask) return true;

    auto * orch = static_cast<moe_orch::Orchestrator *>(user_data);
    if (!orch) return false;

    if (is_topk) {
        int32_t layer = std::atoi(name + 13);
        const int32_t n_elements = (int32_t)ggml_nelements(t);
        if (n_elements <= 0) return false;
        std::vector<int32_t> host(n_elements);
        ggml_backend_tensor_get(t, host.data(), 0, n_elements * sizeof(int32_t));
        orch->on_routing(layer, host.data(), n_elements);
        return true;
    }

    if (is_weights) {
        int32_t layer = std::atoi(name + 16);
        const int32_t n_elements = (int32_t)ggml_nelements(t);
        if (n_elements <= 0) return false;
        std::vector<float> host(n_elements);
        ggml_backend_tensor_get(t, host.data(), 0, n_elements * sizeof(float));
        orch->on_routing_weights(layer, host.data(), n_elements);
        return true;
    }

    // is_ffn_norm — fire on_layer_input per token in the batch.
    int32_t layer = std::atoi(name + 9);
    // Shape: (n_embd, n_tokens, ...). t->ne[0] is the embedding dim.
    const int64_t n_embd   = t->ne[0];
    const int64_t n_tokens = t->ne[1] > 0 ? t->ne[1] : 1;
    if (n_embd <= 0) return false;

    // Predictor expects fp32. ffn_norm output should already be fp32 in this
    // graph; if it ever isn't, we'd need to dequant — punting for now.
    if (t->type != GGML_TYPE_F32) return false;

    const size_t per_token_bytes = (size_t)n_embd * sizeof(float);
    std::vector<float> host(n_embd);
    for (int64_t i = 0; i < n_tokens; i++) {
        ggml_backend_tensor_get(t, host.data(), i * per_token_bytes, per_token_bytes);
        orch->on_layer_input(layer, host.data());
    }
    return true;
}

