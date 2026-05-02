// mode_b.cpp — Mode B slot-pool implementation.

#include "mode_b.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstring>

namespace moe_orch {

// Singleton storage. Set by the host process (orchestrator-test) before
// model load if Mode B is requested.
static ModeBContext * g_ctx = nullptr;

ModeBContext * get_mode_b_context()                        { return g_ctx; }
void           set_mode_b_context(ModeBContext * ctx)      { g_ctx = ctx;  }

// For commit #4 we punt on the actual quantization-aware tensor allocation
// and instead allocate F16 slot tensors at init. The model's expert tensors
// are typically quantized (Q3_K_M, Q4_K_M, Q8_0). Real Mode B will need to
// allocate slot tensors with matching quantization to avoid format conversion
// on every page-in. That's commit #5 — for now we get the plumbing in place
// and ship the simpler F16 path. Rationale: the goal of #4 is "Mode B end-
// to-end compiles, runs, exposes the path." Quant-matching is an
// optimization on top.
//
// The dimensions here are placeholders; they'll be filled from model hparams
// once we wire orchestrator-test to call init() with the right values.
// For now we accept them as parameters and trust the caller.
//
// init() does NOT need n_ff or n_embd because we accept them implicitly: at
// init time we only allocate slot_map tensors (small, [n_expert] int32 each).
// The slot weight tensors require knowing n_ff/n_embd — caller must use the
// upcoming populate_layer() API to attach those once the model is loaded.
// Keeping init lean lets it run before model load.

bool ModeBContext::init(int n_layer, int n_slot, int n_expert, int device_id) {
    if (n_layer <= 0 || n_slot <= 0 || n_expert <= 0) {
        std::fprintf(stderr, "[mode_b] init: invalid params (n_layer=%d n_slot=%d n_expert=%d)\n",
                     n_layer, n_slot, n_expert);
        return false;
    }

    n_slot_    = n_slot;
    n_expert_  = n_expert;
    device_id_ = device_id;

    // ggml_context with no_alloc; we'll use a backend buffer for the data.
    // Estimate tensor count: per layer we have 1 slot_map (commit #4 scope —
    // weight slot tensors come in commit #5). Plus a small overhead.
    const size_t n_tensors = (size_t)n_layer + 8; // +slack
    struct ggml_init_params iparams = {};
    iparams.mem_size   = ggml_tensor_overhead() * n_tensors;
    iparams.mem_buffer = nullptr;
    iparams.no_alloc   = true;
    ctx_ = ggml_init(iparams);
    if (!ctx_) {
        std::fprintf(stderr, "[mode_b] init: ggml_init failed\n");
        return false;
    }

    layers_.resize((size_t)n_layer);
    for (int il = 0; il < n_layer; il++) {
        char name[64];
        std::snprintf(name, sizeof(name), "moe_orch.slot_map.layer_%d", il);
        ggml_tensor * smap = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_expert);
        if (!smap) {
            std::fprintf(stderr, "[mode_b] init: tensor alloc failed at layer %d\n", il);
            ggml_free(ctx_); ctx_ = nullptr;
            return false;
        }
        ggml_set_name(smap, name);
        layers_[il].slot_map = smap;
        // Weight slot tensors (up_slots, gate_slots, down_slots) are NOT
        // allocated in commit #4 — added in commit #5 once we have model
        // hparams for n_ff and n_embd. For now layers_[il].*_slots stay null.
    }

    // Allocate backend buffer on the requested CUDA device. This binds the
    // tensor data pointers to VRAM addresses on that device.
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(device_id);
    if (!buft) {
        std::fprintf(stderr, "[mode_b] init: ggml_backend_cuda_buffer_type(%d) returned null\n", device_id);
        ggml_free(ctx_); ctx_ = nullptr;
        return false;
    }
    backend_buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, buft);
    if (!backend_buffer_) {
        std::fprintf(stderr, "[mode_b] init: backend buffer allocation failed (likely OOM)\n");
        ggml_free(ctx_); ctx_ = nullptr;
        return false;
    }

    std::printf("[mode_b] context initialized: n_layer=%d n_slot=%d n_expert=%d device=%d\n",
                n_layer, n_slot, n_expert, device_id);
    std::printf("[mode_b]   slot_map tensors allocated (%d × %d int32 = %d bytes total)\n",
                n_layer, n_expert, n_layer * n_expert * 4);
    std::printf("[mode_b]   weight slot tensors deferred to commit #5 (need model hparams)\n");

    // Initialize slot_map to FULL identity (slot_map[e] = e for all e in
    // [0, n_expert)). This makes the slot_map remap a behavioral no-op:
    // the gate's selected_experts gets remapped to itself, then used to
    // index into the model's full-size expert tensors as before. Result is
    // identical output to baseline.
    //
    // In commit #5+ this gets replaced with the actual mapping (slot_map[e]
    // = slot_id for cached experts, sentinel for un-cached) once the smaller
    // per-layer slot weight tensors are allocated and the orchestrator
    // populates them from its cache decisions.
    std::vector<int32_t> initial_map(n_expert);
    for (int e = 0; e < n_expert; e++) {
        initial_map[e] = e;
    }
    for (int il = 0; il < n_layer; il++) {
        if (!set_slot_map(il, initial_map.data())) {
            std::fprintf(stderr, "[mode_b] init: set_slot_map failed at layer %d\n", il);
            return false;
        }
    }

    return true;
}

ModeBContext::~ModeBContext() {
    if (backend_buffer_) {
        ggml_backend_buffer_free(backend_buffer_);
        backend_buffer_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    layers_.clear();
}

ggml_tensor * ModeBContext::up_slots(int layer) const {
    if (layer < 0 || layer >= (int)layers_.size()) return nullptr;
    return layers_[layer].up_slots;
}

ggml_tensor * ModeBContext::gate_slots(int layer) const {
    if (layer < 0 || layer >= (int)layers_.size()) return nullptr;
    return layers_[layer].gate_slots;
}

ggml_tensor * ModeBContext::down_slots(int layer) const {
    if (layer < 0 || layer >= (int)layers_.size()) return nullptr;
    return layers_[layer].down_slots;
}

ggml_tensor * ModeBContext::slot_map(int layer) const {
    if (layer < 0 || layer >= (int)layers_.size()) return nullptr;
    return layers_[layer].slot_map;
}

bool ModeBContext::set_slot_map(int layer, const int32_t * map) {
    ggml_tensor * t = slot_map(layer);
    if (!t || !map) return false;
    const size_t bytes = (size_t)n_expert_ * sizeof(int32_t);
    ggml_backend_tensor_set(t, map, 0, bytes);
    return true;
}

bool ModeBContext::set_slot_weight(int layer, int kind, int slot,
                                   const void * src, size_t bytes) {
    if (!src || bytes == 0) return false;
    if (slot < 0 || slot >= n_slot_) return false;
    ggml_tensor * t = nullptr;
    switch (kind) {
        case 0: t = up_slots  (layer); break;
        case 1: t = gate_slots(layer); break;
        case 2: t = down_slots(layer); break;
        default: return false;
    }
    if (!t) return false;  // weight slot tensor not allocated (commit #4)
    const size_t per_slot = ggml_nbytes(t) / (size_t)n_slot_;
    if (bytes != per_slot) {
        std::fprintf(stderr, "[mode_b] set_slot_weight: size mismatch (got %zu expected %zu)\n",
                     bytes, per_slot);
        return false;
    }
    ggml_backend_tensor_set(t, src, (size_t)slot * per_slot, per_slot);
    return true;
}

size_t ModeBContext::slot_size_bytes(int layer, int kind) const {
    ggml_tensor * t = nullptr;
    switch (kind) {
        case 0: t = up_slots  (layer); break;
        case 1: t = gate_slots(layer); break;
        case 2: t = down_slots(layer); break;
        default: return 0;
    }
    if (!t || n_slot_ == 0) return 0;
    return ggml_nbytes(t) / (size_t)n_slot_;
}

} // namespace moe_orch
