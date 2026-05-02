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
             config.shadow_mode,
             config.global_l1)
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
    scratch_conf_.resize(config.l1_prefetch_top_k);
    last_topk_indices_.resize((size_t)n_layers);

    // Plumb the L2 promote threshold (cache default is -1.0 = disabled).
    cache_.set_l2_promote_threshold(config.l2_promote_threshold);

    if (!config_.dump_trace_path.empty()) {
        trace_buf_.resize((size_t)n_layers);
        _trace_open();
    }
}

Orchestrator::~Orchestrator() {
    _trace_finalize();
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

        // When confidence-gated L2->L1 promotion is on, request softmax
        // confidences from the predictor. Otherwise skip (saves the n_experts
        // exp() pass + normalization in the predictor inner loop).
        const bool want_conf = (config_.l2_promote_threshold >= 0.0f);
        auto t0 = std::chrono::steady_clock::now();
        bool ok = l1_predictor_.predict_top_k(layer,
                                              M,
                                              hidden_state,
                                              config_.l1_prefetch_top_k,
                                              scratch_top_.data(),
                                              want_conf ? scratch_conf_.data() : nullptr);
        auto t1 = std::chrono::steady_clock::now();
        l1_predictor_total_ns_ +=
            (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        l1_predictor_calls_++;

        if (ok) {
            cache_.prefetch_to_l1(target_layer,
                                  scratch_top_.data(),
                                  config_.l1_prefetch_top_k,
                                  want_conf ? scratch_conf_.data() : nullptr);
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

// ============================================================================
// Trace dumper — writes a binary file used by the offline predictor training
// pipeline. Format (little-endian):
//
//   Header (32 bytes):
//     char     magic[4]      = "MTRC"
//     uint32   version       = 1
//     uint32   n_layers
//     uint32   hidden_dim
//     uint32   top_k          (0 in header until first record; stamped after)
//     uint32   reserved       = 0
//     uint64   n_records      (filled in at close)
//
//   Each record (fixed size = 8 + 2*hidden_dim + 4*top_k + 4*top_k bytes):
//     int32    layer
//     int32    n_topk_actual
//     fp16[hidden_dim]  hidden_state
//     int32[top_k]      indices       (padded with -1 if n_topk_actual < top_k)
//     fp32[top_k]       weights       (padded with 0 if n_topk_actual < top_k)
//
// On a top_k=8 model with hidden_dim=2048: each record = 4168 bytes. With
// hidden_dim=4096 (235B): 8264 bytes.
// ============================================================================

static uint16_t fp16_from_fp32(float f) {
    // Naive scalar fp32 → fp16. Same as the predictor uses but inverted.
    // We don't need denormals or rounding subtleties here — training uses
    // fp32 widening for the GEMMs anyway.
    union { float f; uint32_t u; } v{f};
    const uint32_t sign = (v.u >> 31) & 0x1;
    const int32_t  exp  = ((v.u >> 23) & 0xFF) - 127 + 15;
    const uint32_t mant =  (v.u >>  13) & 0x3FF;
    if (exp <= 0) {
        return (uint16_t)(sign << 15);  // underflow to signed zero
    }
    if (exp >= 31) {
        return (uint16_t)((sign << 15) | (0x1F << 10));  // saturate to inf
    }
    return (uint16_t)((sign << 15) | (exp << 10) | mant);
}

void Orchestrator::_trace_open() {
    trace_fp_ = std::fopen(config_.dump_trace_path.c_str(), "wb");
    if (!trace_fp_) {
        std::fprintf(stderr, "[orchestrator] trace dump: failed to open %s\n",
                     config_.dump_trace_path.c_str());
        return;
    }
    // Reserve header. We rewrite it on close once n_records and top_k are known.
    char header[32] = {0};
    std::memcpy(header, "MTRC", 4);
    *reinterpret_cast<uint32_t *>(header +  4) = 1;                       // version
    *reinterpret_cast<uint32_t *>(header +  8) = (uint32_t)n_layers_;
    *reinterpret_cast<uint32_t *>(header + 12) = (uint32_t)hidden_dim_;
    *reinterpret_cast<uint32_t *>(header + 16) = 0;                       // top_k (filled later)
    *reinterpret_cast<uint32_t *>(header + 20) = 0;                       // reserved
    *reinterpret_cast<uint64_t *>(header + 24) = 0;                       // n_records
    std::fwrite(header, 1, 32, trace_fp_);
    std::fprintf(stderr, "[orchestrator] trace dump opened: %s\n",
                 config_.dump_trace_path.c_str());
}

void Orchestrator::trace_stash_hidden_batch(int32_t layer, const float * hidden, int32_t n_tokens) {
    if (!trace_fp_)                                       return;
    if (layer < 0 || layer >= (int32_t)trace_buf_.size()) return;
    auto & tb = trace_buf_[layer];
    tb.n_tokens = n_tokens;
    tb.hidden_flat.assign(hidden, hidden + (size_t)n_tokens * hidden_dim_);
    tb.has_hidden = true;
}

void Orchestrator::trace_stash_indices_batch(int32_t layer, const int32_t * indices, int32_t n_tokens, int32_t top_k) {
    if (!trace_fp_)                                       return;
    if (layer < 0 || layer >= (int32_t)trace_buf_.size()) return;
    auto & tb = trace_buf_[layer];
    if (tb.n_tokens != n_tokens) {
        // hidden batch hasn't matched yet; align here.
        tb.n_tokens = n_tokens;
    }
    tb.indices_flat.assign(indices, indices + (size_t)n_tokens * top_k);
    tb.has_indices = true;
}

void Orchestrator::trace_emit_weights_batch(int32_t layer, const float * weights, int32_t n_tokens, int32_t top_k) {
    if (!trace_fp_)                                       return;
    if (layer < 0 || layer >= (int32_t)trace_buf_.size()) return;
    auto & tb = trace_buf_[layer];
    if (!tb.has_hidden || !tb.has_indices)                return;
    if (tb.n_tokens != n_tokens)                          return;  // shape mismatch — skip
    if (trace_top_k_ == 0) trace_top_k_ = top_k;
    if (top_k != trace_top_k_) return;  // top_k drift — skip (defensive)

    constexpr int32_t MAX_K = 32;
    if (trace_top_k_ > MAX_K) trace_top_k_ = MAX_K;

    std::vector<uint16_t> hidden_fp16(hidden_dim_);

    for (int32_t ti = 0; ti < n_tokens; ti++) {
        int32_t hdr[2] = { layer, top_k };
        std::fwrite(hdr, sizeof(int32_t), 2, trace_fp_);

        // hidden_state for this token: fp32 → fp16
        const float * hsrc = tb.hidden_flat.data() + (size_t)ti * hidden_dim_;
        for (int32_t i = 0; i < hidden_dim_; i++) {
            hidden_fp16[i] = fp16_from_fp32(hsrc[i]);
        }
        std::fwrite(hidden_fp16.data(), sizeof(uint16_t), (size_t)hidden_dim_, trace_fp_);

        // indices for this token (top_k each)
        const int32_t * isrc = tb.indices_flat.data() + (size_t)ti * top_k;
        std::fwrite(isrc, sizeof(int32_t), (size_t)top_k, trace_fp_);

        // weights for this token (top_k each)
        const float * wsrc = weights + (size_t)ti * top_k;
        std::fwrite(wsrc, sizeof(float), (size_t)top_k, trace_fp_);

        trace_records_++;
    }

    tb.has_hidden = false;
    tb.has_indices = false;
}

void Orchestrator::_trace_finalize() {
    if (!trace_fp_) return;
    // Patch header with final top_k and n_records.
    std::fseek(trace_fp_, 16, SEEK_SET);
    uint32_t k32 = (uint32_t)trace_top_k_;
    std::fwrite(&k32, sizeof(uint32_t), 1, trace_fp_);
    std::fseek(trace_fp_, 24, SEEK_SET);
    std::fwrite(&trace_records_, sizeof(uint64_t), 1, trace_fp_);
    std::fclose(trace_fp_);
    trace_fp_ = nullptr;
    std::fprintf(stderr, "[orchestrator] trace dump closed: %llu records\n",
                 (unsigned long long)trace_records_);
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
        // Shape: (n_expert_used, n_tokens). t->ne[0]=top_k, t->ne[1]=n_tokens.
        const int32_t top_k    = (int32_t)t->ne[0];
        const int32_t n_tokens = t->ne[1] > 0 ? (int32_t)t->ne[1] : 1;
        const int32_t n_elements = (int32_t)ggml_nelements(t);
        if (n_elements <= 0) return false;
        std::vector<int32_t> host(n_elements);
        ggml_backend_tensor_get(t, host.data(), 0, n_elements * sizeof(int32_t));
        // Cache simulator: full-flat is fine (LRU is idempotent).
        orch->on_routing(layer, host.data(), n_elements);
        // Trace: stash per-token shape.
        if (orch->is_trace_enabled()) {
            orch->trace_stash_indices_batch(layer, host.data(), n_tokens, top_k);
        }
        return true;
    }

    if (is_weights) {
        int32_t layer = std::atoi(name + 16);
        // Shape: (1, n_expert_used, n_tokens). top_k=ne[1], n_tokens=ne[2].
        const int32_t top_k    = (int32_t)t->ne[1];
        const int32_t n_tokens = t->ne[2] > 0 ? (int32_t)t->ne[2] : 1;
        const int32_t n_elements = (int32_t)ggml_nelements(t);
        if (n_elements <= 0) return false;
        std::vector<float> host(n_elements);
        ggml_backend_tensor_get(t, host.data(), 0, n_elements * sizeof(float));
        // Cache simulator: full-flat update.
        orch->on_routing_weights(layer, host.data(), n_elements);
        // Trace: emit n_tokens records, completing the (hidden, indices, weights) triple.
        if (orch->is_trace_enabled()) {
            orch->trace_emit_weights_batch(layer, host.data(), n_tokens, top_k);
        }
        return true;
    }

    // is_ffn_norm — predictor needs the LAST token's hidden state; trace needs all tokens.
    int32_t layer = std::atoi(name + 9);
    // Shape: (n_embd, n_tokens, ...). t->ne[0] is the embedding dim.
    const int64_t n_embd   = t->ne[0];
    const int64_t n_tokens = t->ne[1] > 0 ? t->ne[1] : 1;
    if (n_embd <= 0) return false;

    // Predictor expects fp32. ffn_norm output should already be fp32 in this
    // graph; if it ever isn't, we'd need to dequant — punting for now.
    if (t->type != GGML_TYPE_F32) return false;

    // Pull the whole batch host-side once.
    std::vector<float> host_all((size_t)n_embd * (size_t)n_tokens);
    ggml_backend_tensor_get(t, host_all.data(), 0, host_all.size() * sizeof(float));

    // Predictor: fire on_layer_input per token (matches Python pre-hook semantics).
    const size_t per_token_floats = (size_t)n_embd;
    for (int64_t i = 0; i < n_tokens; i++) {
        orch->on_layer_input(layer, host_all.data() + i * per_token_floats);
    }
    // Trace: stash all tokens' hidden states.
    if (orch->is_trace_enabled()) {
        orch->trace_stash_hidden_batch(layer, host_all.data(), (int32_t)n_tokens);
    }
    return true;
}

