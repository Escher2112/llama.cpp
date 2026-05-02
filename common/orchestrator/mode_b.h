// mode_b.h — Mode B (slot-mapped) live-mode context for the orchestrator.
//
// In Mode B, per-layer expert tensors are NOT loaded at full size to VRAM.
// Instead, a smaller per-layer "slot pool" (shape [n_ff, n_embd, n_slot]
// where n_slot < n_expert) holds a working subset, plus a [n_expert]
// slot_map tensor that translates gate-selected expert IDs to slot IDs
// at MoE-op time.
//
// This file owns the slot tensor allocation, the slot_map tensors, and a
// CPU-side mirror of the full expert weights for paging. The orchestrator
// updates slot contents and slot_map between forward passes via
// ggml_backend_tensor_set.
//
// Access pattern: per-arch graph builders (qwen3moe.cpp, deepseek3.cpp)
// query the global ModeBContext singleton at graph build time. When
// active, they pass the slot tensors + slot_map to build_moe_ffn instead
// of the model's full-size expert tensors.

#pragma once

#include <cstdint>
#include <vector>

struct ggml_tensor;
struct ggml_context;
struct ggml_backend_buffer;
struct llama_model;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace moe_orch {

// Per-layer slot tensors and the slot_map for that layer.
struct ModeBLayer {
    ggml_tensor * up_slots   = nullptr;     // [n_ff, n_embd, n_slot]
    ggml_tensor * gate_slots = nullptr;     // [n_ff, n_embd, n_slot] (or nullptr if model is fused)
    ggml_tensor * down_slots = nullptr;     // [n_embd, n_ff, n_slot]
    ggml_tensor * slot_map   = nullptr;     // [n_expert] of int32 slot indices
    ggml_tensor * valid_mask = nullptr;     // [n_expert] of float: 0 if cached, -INF if uncached

    // Runtime cache state. Mirrors the slot tensor occupancy on the CPU
    // side so we can compute the diff between desired and current contents.
    std::vector<int32_t> slot_to_expert;    // size n_slot; slot K holds expert slot_to_expert[K], or -1
    std::vector<int32_t> expert_to_slot;    // size n_expert; expert E in slot expert_to_slot[E], or -1

    // Pointers into the source model's full-size expert tensors. Used as
    // CPU mirror for paging in via tensor_get + tensor_set. Set by
    // attach_model() after init.
    ggml_tensor * src_up   = nullptr;
    ggml_tensor * src_gate = nullptr;
    ggml_tensor * src_down = nullptr;

    // LRU usage order. MRU at front. Capacity bounded only by the number
    // of distinct experts ever observed; refresh_slots() keeps top-n_slot
    // resident.
    std::vector<int32_t> lru;               // MRU at index 0

    // Pending: experts that should be paged in but aren't yet. Filled by
    // on_routing(); drained by refresh_slots(). Avoids unbounded growth
    // by deduping against lru.
    std::vector<int32_t> dirty;
    bool                 slot_map_dirty = false;

    // ---- Tier 2 (pinned host) cache (commit #10) ----
    //
    // Three buffers (up/gate/down) of size n_tier2 * per_expert_bytes each,
    // backed by CUDA pinned host memory (cudaHostAlloc) for fast PCIe DMA
    // into the device-side slot tensors. Populated by promote_to_tier2()
    // from the CPU mirror; consumed by _page_in_expert() as a fast source.
    //
    // Lifetime: allocated when ModeBContext::init_tier2(n_tier2) is called.
    // Empty otherwise — _page_in_expert falls back to CPU mirror.
    void *               tier2_up_buf   = nullptr;
    void *               tier2_gate_buf = nullptr;
    void *               tier2_down_buf = nullptr;

    // Tier 2 occupancy. Same shape as Tier 1 but n_tier2 entries.
    std::vector<int32_t> tier2_to_expert;
    std::vector<int32_t> expert_to_tier2;
    // Round-robin write pointer for tier 2 evictions (FIFO; refined later).
    int32_t              tier2_next_write = 0;
};

class ModeBContext {
public:
    // Initialize for a model with n_layer MoE layers.
    //   model      — the loaded llama_model. Used to query per-layer expert
    //                tensor shapes/quant types so slot tensors are allocated
    //                with matching layout (required for the matmul kernels
    //                to accept them as drop-in substitutes).
    //   n_layer    — number of MoE layers in the model (caller computes).
    //   n_expert   — experts per layer (caller computes from model metadata).
    //   n_slot     — VRAM-resident slot count per layer. Typical 4-32 on
    //                consumer GPUs.
    //   device_id  — CUDA device ordinal.
    //
    // Allocates per-layer slot weight tensors (up_slots, gate_slots if model
    // has separate gate, down_slots), per-layer slot_map tensors, plus a
    // single CUDA backend buffer covering all of the above. Then populates
    // each slot with the corresponding model expert's weights (slot k holds
    // expert k for k in [0, n_slot)) and initializes slot_map to map
    // experts 0..n_slot-1 to slots 0..n_slot-1, mapping experts beyond
    // n_slot-1 to slot 0 as a sentinel (commit #5 simplification — commit #6
    // adds proper page-on-miss handling for unmapped experts).
    //
    // Returns false on any allocation, copy, or model-tensor-lookup failure.
    bool init(const llama_model * model,
              int n_layer, int n_expert, int n_slot, int device_id);

    // Clean up: free backend buffer and ggml context.
    ~ModeBContext();

    // Per-layer accessors. Returns nullptr if uninitialized or layer out of range.
    ggml_tensor * up_slots  (int layer) const;
    ggml_tensor * gate_slots(int layer) const;
    ggml_tensor * down_slots(int layer) const;
    ggml_tensor * slot_map  (int layer) const;
    ggml_tensor * valid_mask(int layer) const;

    int n_layer() const { return (int)layers_.size(); }
    int n_slot()  const { return n_slot_; }
    int n_expert() const { return n_expert_; }
    bool active() const { return ctx_ != nullptr && backend_buffer_ != nullptr; }

    // ---- Graceful-mask mode (commit #9) ----
    //
    // When OFF (default for backwards-compat): valid_mask carries -INFINITY
    // for uncached experts, so the gate is hard-constrained to pick from the
    // resident set. This is mode-collapse-prone at small n_slot.
    //
    // When ON: valid_mask is ALL ZEROS — gate selects freely. Cache misses
    // are handled by page-on-demand (slow Tier 3) or, if Hopfield+Tier 2 is
    // active, by fast PCIe swap from the pinned buffer. Pairs with the
    // orchestrator's Hopfield-driven mark_predicted to keep the resident set
    // hot ahead of the gate's actual decisions.
    void set_graceful_mask(bool g) { graceful_mask_ = g; }
    bool graceful_mask() const     { return graceful_mask_; }

    // Populate the slot_map tensor for `layer` from a host-side int32 array
    // of length n_expert (entries are slot indices in [0, n_slot)). Wraps
    // ggml_backend_tensor_set. Caller must have already populated the
    // corresponding slot weight tensors via populate_slot_weights().
    bool set_slot_map(int layer, const int32_t * map);

    // Populate one slot's weights for `layer` from a host buffer. Wraps
    // ggml_backend_tensor_set on the appropriate slot tensor. Used at init
    // (initial warm-up) and by the orchestrator's cache decisions to page
    // an expert from CPU mirror to slot.
    //
    // kind: 0=up, 1=gate, 2=down. The host buffer must match the slot's
    // physical byte size (same quant type as the model's expert tensors).
    bool set_slot_weight(int layer, int kind, int slot,
                         const void * src, size_t bytes);

    // Get the byte size of one slot's weights for `layer`/`kind`. Useful
    // for callers preparing host-side copies to page in.
    size_t slot_size_bytes(int layer, int kind) const;

    // ---- Runtime cache management (commit #6) ----

    // Provide pointers to the source model's full-size expert tensors.
    // Used by the runtime LRU path to page experts from the CPU mirror
    // (where the loaded model's experts live with --cpu-moe) into slots.
    // Called once after init() with the same llama_model passed there.
    void attach_model(const llama_model * model);

    // Observation: the gate at `layer` selected `experts[count]`. Updates
    // the per-layer LRU. Called from the cb_eval bridge when ffn_moe_topk-N
    // fires.
    void on_routing(int layer, const int32_t * experts, int count);

    // Predictor hint: a learned predictor expects experts[count] will be
    // wanted at `layer` in the near future (typically next forward pass
    // for that layer). Marks those experts as dirty for inclusion in the
    // next refresh_slots() call. Called from Orchestrator::on_layer_input
    // alongside cache_.prefetch_to_l1, so mode_b's slot pool and the
    // shadow-mode tiered cache stay aligned.
    //
    // Predictor predictions help BEFORE the gate fires (preemptive
    // page-in), where on_routing only helps AFTER (reactive). With both
    // wired, cache coverage converges much faster.
    void mark_predicted(int layer, const int32_t * experts, int count);

    // Apply queued cache decisions: any experts that the LRU has marked
    // "should be in slots" but currently aren't get paged in (CPU mirror
    // → slot via tensor_get/tensor_set). Updates slot_map to reflect new
    // slot occupancy. Synchronous for commit #6 — async pipelining is a
    // commit #7+ optimization.
    //
    // Should be called between forward passes (after each llama_decode).
    // Returns the number of slot updates made (for diagnostics).
    int refresh_slots();

    // ---- Tier 2 pinned host buffer (commit #10) ----
    //
    // Allocate per-layer pinned host buffers of size n_tier2 * per_expert_bytes
    // for {up, gate, down}. Returns false if allocation fails. May be called
    // any time after init(). n_tier2 should be > n_slot (typical: 2-3x).
    bool init_tier2(int n_tier2);
    int  n_tier2() const { return n_tier2_; }
    bool tier2_active() const { return n_tier2_ > 0; }

    // Promote experts to Tier 2 (copy from CPU mirror into pinned host buffer).
    // Idempotent: if expert is already in Tier 2, no copy is performed.
    // Returns the number of new (non-cached) Tier 2 promotions performed.
    int promote_to_tier2(int layer, const int32_t * experts, int count);

    // Helper: query Tier 2 occupancy.
    int  tier2_index(int layer, int expert) const;     // -1 if absent

    uint64_t total_tier2_promotions() const { return total_tier2_promotions_; }
    uint64_t total_pages_from_tier2() const { return total_pages_from_tier2_; }

    // Diagnostics
    uint64_t total_pages_in() const { return total_pages_in_; }
    uint64_t total_evictions() const { return total_evictions_; }

private:
    int n_slot_   = 0;
    int n_expert_ = 0;
    int n_tier2_  = 0;
    int device_id_ = 0;
    bool graceful_mask_ = false;
    ggml_context *           ctx_            = nullptr;
    ggml_backend_buffer_t    backend_buffer_ = nullptr;
    std::vector<ModeBLayer>  layers_;

    // Counters for diagnostics
    uint64_t total_pages_in_  = 0;
    uint64_t total_evictions_ = 0;
    uint64_t total_tier2_promotions_ = 0;
    uint64_t total_pages_from_tier2_ = 0;

    // Scratch buffer reused across page-in operations to avoid allocations.
    std::vector<uint8_t> scratch_;

    // Helpers
    bool _page_in_expert(int layer, int expert, int slot);
    int  _pick_eviction_slot(int layer);
};

// Global singleton. Set by orchestrator-test (or other host) before model
// load when --orchestrator-mode slot is active. Read by per-arch graph
// builders. nullptr means Mode B is OFF — the graph builders take the
// standard path with the model's full expert tensors.
ModeBContext * get_mode_b_context();
void           set_mode_b_context(ModeBContext * ctx);

} // namespace moe_orch
