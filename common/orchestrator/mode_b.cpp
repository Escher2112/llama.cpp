// mode_b.cpp — Mode B slot-pool implementation.

#include "mode_b.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace moe_orch {

// Singleton storage. Set by the host process (orchestrator-test) before
// model load if Mode B is requested.
static ModeBContext * g_ctx = nullptr;

ModeBContext * get_mode_b_context()                        { return g_ctx; }
void           set_mode_b_context(ModeBContext * ctx)      { g_ctx = ctx;  }

bool ModeBContext::init(const llama_model * model,
                        int n_layer, int n_expert, int n_slot, int device_id) {
    if (!model || n_layer <= 0 || n_slot <= 0 || n_expert <= 0) {
        std::fprintf(stderr, "[mode_b] init: invalid params (model=%p n_layer=%d n_slot=%d n_expert=%d)\n",
                     (const void *)model, n_layer, n_slot, n_expert);
        return false;
    }

    n_slot_    = n_slot;
    n_expert_  = n_expert;
    device_id_ = device_id;

    // ggml_context with no_alloc; we'll use a single CUDA backend buffer for
    // the data. Tensor count: per layer we have up_slots + gate_slots +
    // down_slots + slot_map = 4. Plus slack.
    const size_t n_tensors = (size_t)n_layer * 4 + 8;
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

    // Per-layer: locate the model's expert tensors by name, allocate matching
    // slot tensors with the same shape/quant pattern but n_slot in the expert
    // dim. Each model expert tensor has shape [a, b, n_expert] (where a,b are
    // arch-specific); we replace the third dim with n_slot.
    //
    // Names follow llama.cpp convention: blk.<L>.ffn_<kind>_exps.weight where
    // kind ∈ {up, gate, down}. Some MoE archs may have nullptr gate (fused
    // gate_up); we tolerate that by leaving gate_slots nullptr.
    size_t total_slot_bytes = 0;
    for (int il = 0; il < n_layer; il++) {
        char nm[64];
        auto get_or = [&](const char * kind) -> ggml_tensor * {
            std::snprintf(nm, sizeof(nm), "blk.%d.ffn_%s_exps.weight", il, kind);
            return llama_model_get_tensor(model, nm);
        };
        ggml_tensor * up_t   = get_or("up");
        ggml_tensor * gate_t = get_or("gate");
        ggml_tensor * down_t = get_or("down");

        if (!up_t || !down_t) {
            // Not a MoE layer (e.g., dense layers in hybrid arches), skip.
            // slot_map stays null too — qwen3moe.cpp will see null and fall
            // back to model tensors / no remap.
            continue;
        }

        // Allocate slot tensors matching the model's quant + spatial dims,
        // with n_slot in the expert dim (third axis). The model expert
        // tensors have shape ne[0]=a, ne[1]=b, ne[2]=n_expert.
        layers_[il].up_slots   = ggml_new_tensor_3d(ctx_, up_t->type,   up_t->ne[0],   up_t->ne[1],   n_slot);
        layers_[il].down_slots = ggml_new_tensor_3d(ctx_, down_t->type, down_t->ne[0], down_t->ne[1], n_slot);
        if (gate_t) {
            layers_[il].gate_slots = ggml_new_tensor_3d(ctx_, gate_t->type, gate_t->ne[0], gate_t->ne[1], n_slot);
        }
        layers_[il].slot_map = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_expert);

        // Set debug names for traceability in graph dumps.
        char dbg[64];
        std::snprintf(dbg, sizeof(dbg), "moe_orch.up_slots.layer_%d",   il); ggml_set_name(layers_[il].up_slots, dbg);
        std::snprintf(dbg, sizeof(dbg), "moe_orch.down_slots.layer_%d", il); ggml_set_name(layers_[il].down_slots, dbg);
        if (layers_[il].gate_slots) {
            std::snprintf(dbg, sizeof(dbg), "moe_orch.gate_slots.layer_%d", il); ggml_set_name(layers_[il].gate_slots, dbg);
        }
        std::snprintf(dbg, sizeof(dbg), "moe_orch.slot_map.layer_%d", il); ggml_set_name(layers_[il].slot_map, dbg);

        total_slot_bytes += ggml_nbytes(layers_[il].up_slots);
        total_slot_bytes += ggml_nbytes(layers_[il].down_slots);
        if (layers_[il].gate_slots) total_slot_bytes += ggml_nbytes(layers_[il].gate_slots);
        total_slot_bytes += ggml_nbytes(layers_[il].slot_map);
    }

    // Allocate the single backend buffer covering all tensors in ctx_.
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(device_id);
    if (!buft) {
        std::fprintf(stderr, "[mode_b] init: ggml_backend_cuda_buffer_type(%d) returned null\n", device_id);
        ggml_free(ctx_); ctx_ = nullptr;
        return false;
    }
    backend_buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, buft);
    if (!backend_buffer_) {
        std::fprintf(stderr, "[mode_b] init: backend buffer allocation failed (likely OOM, "
                             "needed %.2f GB VRAM)\n",
                     total_slot_bytes / (1024.0 * 1024.0 * 1024.0));
        ggml_free(ctx_); ctx_ = nullptr;
        return false;
    }

    std::printf("[mode_b] context initialized: n_layer=%d n_slot=%d n_expert=%d device=%d\n",
                n_layer, n_slot, n_expert, device_id);
    std::printf("[mode_b]   total slot tensor bytes: %.2f GB\n",
                total_slot_bytes / (1024.0 * 1024.0 * 1024.0));

    // Initial population: copy first n_slot experts' weights from the model's
    // full-size expert tensors into our slot tensors. Each expert occupies
    // (total_bytes_of_model_tensor / n_expert) bytes. Use ggml_backend_tensor_get
    // to extract from the model's CPU buffer, then ggml_backend_tensor_set to
    // write into our CUDA backend buffer.
    std::vector<uint8_t> scratch;
    auto copy_first_n_slot = [&](ggml_tensor * src, ggml_tensor * dst, const char * label) -> bool {
        if (!src || !dst) return true;  // skipped layer
        const size_t src_per_expert = ggml_nbytes(src) / (size_t)n_expert;
        const size_t dst_per_slot   = ggml_nbytes(dst) / (size_t)n_slot;
        if (src_per_expert != dst_per_slot) {
            std::fprintf(stderr, "[mode_b] init: %s per-expert size mismatch "
                                 "(src=%zu dst=%zu) — quant/dim layout differs?\n",
                         label, src_per_expert, dst_per_slot);
            return false;
        }
        if (scratch.size() < src_per_expert) scratch.resize(src_per_expert);
        for (int s = 0; s < n_slot; s++) {
            ggml_backend_tensor_get(src, scratch.data(), (size_t)s * src_per_expert, src_per_expert);
            ggml_backend_tensor_set(dst, scratch.data(), (size_t)s * dst_per_slot, dst_per_slot);
        }
        return true;
    };
    for (int il = 0; il < n_layer; il++) {
        if (!layers_[il].up_slots) continue; // skipped layer
        char nm[64];
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_up_exps.weight", il);
        ggml_tensor * up_t = llama_model_get_tensor(model, nm);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_exps.weight", il);
        ggml_tensor * gate_t = llama_model_get_tensor(model, nm);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", il);
        ggml_tensor * down_t = llama_model_get_tensor(model, nm);

        if (!copy_first_n_slot(up_t,   layers_[il].up_slots,   "up"))   return false;
        if (!copy_first_n_slot(gate_t, layers_[il].gate_slots, "gate")) return false;
        if (!copy_first_n_slot(down_t, layers_[il].down_slots, "down")) return false;
    }

    // Initial slot_map: experts 0..n_slot-1 → slots 0..n_slot-1 (identity);
    // experts n_slot..n_expert-1 → slot 0 (sentinel — they read slot 0's
    // weights, which is "wrong but bounded" output. commit #6 adds a real
    // page-on-miss path). The orchestrator's predictor will update this
    // dynamically in commit #6 to reflect actual cache contents.
    std::vector<int32_t> initial_map(n_expert, 0);
    for (int e = 0; e < n_expert && e < n_slot; e++) {
        initial_map[e] = e;
    }
    for (int il = 0; il < n_layer; il++) {
        if (!layers_[il].slot_map) continue;
        if (!set_slot_map(il, initial_map.data())) {
            std::fprintf(stderr, "[mode_b] init: set_slot_map failed at layer %d\n", il);
            return false;
        }
    }

    int active_layers = 0;
    for (auto & l : layers_) if (l.up_slots) active_layers++;
    std::printf("[mode_b]   populated %d MoE layers with first %d experts each\n",
                active_layers, n_slot);
    std::printf("[mode_b]   slot_map: experts [0,%d) → slots [0,%d), experts [%d,%d) → slot 0 (sentinel)\n",
                n_slot, n_slot, n_slot, n_expert);

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
