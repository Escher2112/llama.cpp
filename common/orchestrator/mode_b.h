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
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace moe_orch {

// Per-layer slot tensors and the slot_map for that layer.
struct ModeBLayer {
    ggml_tensor * up_slots   = nullptr;     // [n_ff, n_embd, n_slot]
    ggml_tensor * gate_slots = nullptr;     // [n_ff, n_embd, n_slot] (or nullptr if model is fused)
    ggml_tensor * down_slots = nullptr;     // [n_embd, n_ff, n_slot]
    ggml_tensor * slot_map   = nullptr;     // [n_expert] of int32 slot indices
};

class ModeBContext {
public:
    // Initialize for a model with n_layer MoE layers.
    // n_slot: number of expert slots per layer (typical: 4-32 on consumer GPUs).
    // device_id: CUDA device ordinal.
    // Returns false if allocation or backend buffer creation fails.
    bool init(int n_layer, int n_slot, int n_expert, int device_id);

    // Clean up: free backend buffer and ggml context.
    ~ModeBContext();

    // Per-layer accessors. Returns nullptr if uninitialized or layer out of range.
    ggml_tensor * up_slots  (int layer) const;
    ggml_tensor * gate_slots(int layer) const;
    ggml_tensor * down_slots(int layer) const;
    ggml_tensor * slot_map  (int layer) const;

    int n_layer() const { return (int)layers_.size(); }
    int n_slot()  const { return n_slot_; }
    int n_expert() const { return n_expert_; }
    bool active() const { return ctx_ != nullptr && backend_buffer_ != nullptr; }

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

private:
    int n_slot_   = 0;
    int n_expert_ = 0;
    int device_id_ = 0;
    ggml_context *           ctx_            = nullptr;
    ggml_backend_buffer_t    backend_buffer_ = nullptr;
    std::vector<ModeBLayer>  layers_;
};

// Global singleton. Set by orchestrator-test (or other host) before model
// load when --orchestrator-mode slot is active. Read by per-arch graph
// builders. nullptr means Mode B is OFF — the graph builders take the
// standard path with the model's full expert tensors.
ModeBContext * get_mode_b_context();
void           set_mode_b_context(ModeBContext * ctx);

} // namespace moe_orch
