// mode_b.cpp — Mode B slot-pool implementation.

#include "mode_b.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    // down_slots + slot_map + valid_mask = 5. Plus slack.
    const size_t n_tensors = (size_t)n_layer * 5 + 16;
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
        layers_[il].slot_to_expert.assign((size_t)n_slot, -1);
        layers_[il].expert_to_slot.assign((size_t)n_expert, -1);
        layers_[il].lru.reserve((size_t)n_slot * 2);
        layers_[il].dirty.reserve((size_t)n_slot);
    }

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
        layers_[il].slot_map   = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, n_expert);
        layers_[il].valid_mask = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, n_expert);

        // Set debug names for traceability in graph dumps.
        char dbg[64];
        std::snprintf(dbg, sizeof(dbg), "moe_orch.up_slots.layer_%d",   il); ggml_set_name(layers_[il].up_slots, dbg);
        std::snprintf(dbg, sizeof(dbg), "moe_orch.down_slots.layer_%d", il); ggml_set_name(layers_[il].down_slots, dbg);
        if (layers_[il].gate_slots) {
            std::snprintf(dbg, sizeof(dbg), "moe_orch.gate_slots.layer_%d", il); ggml_set_name(layers_[il].gate_slots, dbg);
        }
        std::snprintf(dbg, sizeof(dbg), "moe_orch.slot_map.layer_%d",   il); ggml_set_name(layers_[il].slot_map, dbg);
        std::snprintf(dbg, sizeof(dbg), "moe_orch.valid_mask.layer_%d", il); ggml_set_name(layers_[il].valid_mask, dbg);

        total_slot_bytes += ggml_nbytes(layers_[il].up_slots);
        total_slot_bytes += ggml_nbytes(layers_[il].down_slots);
        if (layers_[il].gate_slots) total_slot_bytes += ggml_nbytes(layers_[il].gate_slots);
        total_slot_bytes += ggml_nbytes(layers_[il].slot_map);
        total_slot_bytes += ggml_nbytes(layers_[il].valid_mask);
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

        // Cache the source tensor pointers for runtime page-in, and mark the
        // initial slot occupancy (slots 0..n_slot-1 hold experts 0..n_slot-1).
        layers_[il].src_up   = up_t;
        layers_[il].src_gate = gate_t;
        layers_[il].src_down = down_t;
        for (int s = 0; s < n_slot; s++) {
            if (s < n_expert) {
                layers_[il].slot_to_expert[s] = s;
                layers_[il].expert_to_slot[s] = s;
                layers_[il].lru.push_back(s);  // MRU first; doesn't matter for init
            }
        }
    }

    // Initial slot_map: experts 0..n_slot-1 → slots 0..n_slot-1 (identity);
    // experts n_slot..n_expert-1 → slot 0 (sentinel). With cache-aware
    // gating (the valid_mask tensor) the gate masks uncached experts to
    // -INFINITY before argsort_top_k, so the gate can't actually select
    // them. The slot 0 sentinel is just a never-read fallback.
    std::vector<int32_t> initial_map(n_expert, 0);
    for (int e = 0; e < n_expert && e < n_slot; e++) {
        initial_map[e] = e;
    }
    // Initial valid_mask. Two regimes:
    //   - graceful_mask_ OFF (default): -INFINITY for uncached experts. Hard
    //     constraint — gate picks from cached set only. Mode-collapse-prone
    //     at small n_slot.
    //   - graceful_mask_ ON: all zeros. Gate selects freely. Cache misses go
    //     through the page-on-demand fallback. Pair with Hopfield prefetch
    //     for the Tier-2-set keeping the resident pool warm.
    std::vector<float> initial_mask(n_expert, graceful_mask_ ? 0.0f : -INFINITY);
    if (!graceful_mask_) {
        for (int e = 0; e < n_expert && e < n_slot; e++) {
            initial_mask[e] = 0.0f;
        }
    }
    for (int il = 0; il < n_layer; il++) {
        if (!layers_[il].slot_map) continue;
        if (!set_slot_map(il, initial_map.data())) {
            std::fprintf(stderr, "[mode_b] init: set_slot_map failed at layer %d\n", il);
            return false;
        }
        if (layers_[il].valid_mask) {
            ggml_backend_tensor_set(layers_[il].valid_mask, initial_mask.data(),
                                    0, (size_t)n_expert * sizeof(float));
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
    // Free per-layer pinned host buffers (Tier 2). Order matters:
    // unregister with CUDA before freeing the heap allocation.
    for (auto & L : layers_) {
        auto release = [](void *& p) {
            if (!p) return;
            ggml_backend_cuda_unregister_host_buffer(p);
            std::free(p);
            p = nullptr;
        };
        release(L.tier2_up_buf);
        release(L.tier2_gate_buf);
        release(L.tier2_down_buf);
    }
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

ggml_tensor * ModeBContext::valid_mask(int layer) const {
    if (layer < 0 || layer >= (int)layers_.size()) return nullptr;
    return layers_[layer].valid_mask;
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

// ---- Commit #6: runtime LRU + page-on-miss ----

void ModeBContext::attach_model(const llama_model * model) {
    if (!model) return;
    for (size_t il = 0; il < layers_.size(); il++) {
        if (!layers_[il].up_slots) continue;
        // Re-fetch (init already did this but stash for safety/idempotence)
        char nm[64];
        std::snprintf(nm, sizeof(nm), "blk.%zu.ffn_up_exps.weight",   il);
        layers_[il].src_up   = llama_model_get_tensor(model, nm);
        std::snprintf(nm, sizeof(nm), "blk.%zu.ffn_gate_exps.weight", il);
        layers_[il].src_gate = llama_model_get_tensor(model, nm);
        std::snprintf(nm, sizeof(nm), "blk.%zu.ffn_down_exps.weight", il);
        layers_[il].src_down = llama_model_get_tensor(model, nm);
    }
}

void ModeBContext::on_routing(int layer, const int32_t * experts, int count) {
    if (layer < 0 || layer >= (int)layers_.size()) return;
    auto & L = layers_[layer];
    if (!L.up_slots) return;  // skipped (non-MoE) layer
    for (int i = 0; i < count; i++) {
        const int32_t e = experts[i];
        if (e < 0 || e >= n_expert_) continue;

        // Move e to MRU position in lru list.
        bool found = false;
        for (auto it = L.lru.begin(); it != L.lru.end(); ++it) {
            if (*it == e) { L.lru.erase(it); found = true; break; }
        }
        L.lru.insert(L.lru.begin(), e);
        // Cap lru size to avoid unbounded growth — keep at most 2*n_slot
        // entries (provides a small "warm tail" beyond the resident set).
        const size_t cap = (size_t)n_slot_ * 2;
        if (L.lru.size() > cap) L.lru.resize(cap);

        // If this expert isn't currently in slots, mark dirty for refresh.
        if (L.expert_to_slot[e] < 0) {
            // Already in dirty queue?
            bool in_dirty = false;
            for (int32_t d : L.dirty) {
                if (d == e) { in_dirty = true; break; }
            }
            if (!in_dirty) L.dirty.push_back(e);
            L.slot_map_dirty = true;
        }
        (void)found;
    }
}

void ModeBContext::mark_predicted(int layer, const int32_t * experts, int count) {
    if (layer < 0 || layer >= (int)layers_.size()) return;
    auto & L = layers_[layer];
    if (!L.up_slots) return;  // skipped layer
    for (int i = 0; i < count; i++) {
        const int32_t e = experts[i];
        if (e < 0 || e >= n_expert_) continue;
        // Move to MRU position so refresh_slots prefers keeping it.
        // (Same logic as on_routing — predicted-and-not-yet-used acts like
        // a soft "I expect this soon.")
        bool found = false;
        for (auto it = L.lru.begin(); it != L.lru.end(); ++it) {
            if (*it == e) { L.lru.erase(it); found = true; break; }
        }
        L.lru.insert(L.lru.begin(), e);
        const size_t cap = (size_t)n_slot_ * 2;
        if (L.lru.size() > cap) L.lru.resize(cap);

        if (L.expert_to_slot[e] < 0) {
            bool in_dirty = false;
            for (int32_t d : L.dirty) {
                if (d == e) { in_dirty = true; break; }
            }
            if (!in_dirty) L.dirty.push_back(e);
            L.slot_map_dirty = true;
        }
        (void)found;
    }
}

int ModeBContext::_pick_eviction_slot(int layer) {
    auto & L = layers_[layer];
    // Walk lru from LRU end (back) and find the first entry currently
    // occupying a slot — that's the LRU resident expert. Evict it.
    for (auto it = L.lru.rbegin(); it != L.lru.rend(); ++it) {
        int32_t e = *it;
        int32_t s = L.expert_to_slot[e];
        if (s >= 0) {
            return s;
        }
    }
    // Fallback: slot 0 (shouldn't happen in steady state)
    return 0;
}

bool ModeBContext::_page_in_expert(int layer, int expert, int slot) {
    auto & L = layers_[layer];
    if (slot < 0 || slot >= n_slot_) return false;

    // Fast path: if expert is in Tier 2 (pinned host buffer), copy directly
    // from there to slot via tensor_set. Pinned source enables CUDA's
    // page-locked DMA → measurably faster than from a random heap allocation.
    const int t2 = (n_tier2_ > 0) ? L.expert_to_tier2[(size_t)expert] : -1;

    // For Tier 3 path we need separate scratch for each kind because the
    // CUDA backend's tensor_set may queue cudaMemcpyAsync and return before
    // the host->device DMA completes. Reusing one scratch across kinds
    // would corrupt in-flight copies. Three separate buffers are cheap and
    // correct without forcing a stream sync.
    auto copy_one = [&](ggml_tensor * src, ggml_tensor * dst,
                        void * tier2_buf, std::vector<uint8_t> & local_scratch) -> bool {
        if (!src || !dst) return true;  // optional (e.g., gate)
        const size_t per_expert = ggml_nbytes(src) / (size_t)n_expert_;
        const size_t per_slot   = ggml_nbytes(dst) / (size_t)n_slot_;
        if (per_expert != per_slot) return false;
        if (t2 >= 0 && tier2_buf) {
            // Tier 2 → slot. Pinned host source.
            const uint8_t * src_ptr =
                (const uint8_t *)tier2_buf + (size_t)t2 * per_expert;
            ggml_backend_tensor_set(dst, src_ptr, (size_t)slot * per_slot, per_slot);
        } else {
            // Tier 3 (CPU mirror) → slot. Slow path; goes through scratch.
            if (local_scratch.size() < per_expert) local_scratch.resize(per_expert);
            ggml_backend_tensor_get(src, local_scratch.data(), (size_t)expert * per_expert, per_expert);
            ggml_backend_tensor_set(dst, local_scratch.data(), (size_t)slot * per_slot, per_slot);
        }
        return true;
    };

    // Three persistent scratch buffers, kept around across calls. They MUST
    // remain valid until the tensor_set's cudaMemcpyAsync completes — which
    // for our usage means until the slot_map tensor_set lands later in the
    // stream (it forces a sync-equivalent barrier for the graph reads).
    if (!copy_one(L.src_up,   L.up_slots,   L.tier2_up_buf,   scratch_up_))   return false;
    if (!copy_one(L.src_gate, L.gate_slots, L.tier2_gate_buf, scratch_gate_)) return false;
    if (!copy_one(L.src_down, L.down_slots, L.tier2_down_buf, scratch_down_)) return false;

    if (t2 >= 0) total_pages_from_tier2_++;

    // Bookkeeping: evict whatever was here, install new expert.
    int32_t prev = L.slot_to_expert[slot];
    if (prev >= 0) {
        L.expert_to_slot[prev] = -1;
        total_evictions_++;
    }
    L.slot_to_expert[slot] = expert;
    L.expert_to_slot[expert] = slot;
    total_pages_in_++;
    return true;
}

int ModeBContext::refresh_slots() {
    int updates = 0;
    std::vector<int32_t> map_buf;
    map_buf.resize((size_t)n_expert_);

    for (size_t il = 0; il < layers_.size(); il++) {
        auto & L = layers_[il];
        if (!L.up_slots) continue;

        // Page in any dirty experts. For each, pick a victim slot via LRU
        // among currently-resident experts (not in dirty).
        for (int32_t e : L.dirty) {
            // Skip if it's already cached (could have been paged in by
            // another expert that shares the slot — defensive).
            if (L.expert_to_slot[e] >= 0) continue;
            const int slot = _pick_eviction_slot((int)il);
            if (!_page_in_expert((int)il, e, slot)) {
                // Failure — log once and continue with other dirty experts
                std::fprintf(stderr, "[mode_b] page_in failed: layer=%zu expert=%d slot=%d\n",
                             il, e, slot);
                continue;
            }
            updates++;
        }
        L.dirty.clear();

        // If the slot occupancy changed, rebuild slot_map AND valid_mask
        // for this layer. valid_mask carries -INFINITY for uncached experts
        // when graceful_mask_ is OFF (hard cache-aware gating); when ON, it's
        // all zeros (gate selects freely, page-on-demand handles misses).
        if (L.slot_map_dirty || updates > 0) {
            std::vector<float> mask_buf((size_t)n_expert_,
                                        graceful_mask_ ? 0.0f : -INFINITY);
            for (int e = 0; e < n_expert_; e++) {
                int32_t s = L.expert_to_slot[e];
                map_buf[(size_t)e] = (s >= 0) ? s : 0;
                if (!graceful_mask_ && s >= 0) mask_buf[(size_t)e] = 0.0f;
            }
            set_slot_map((int)il, map_buf.data());
            if (L.valid_mask) {
                ggml_backend_tensor_set(L.valid_mask, mask_buf.data(),
                                        0, (size_t)n_expert_ * sizeof(float));
            }
            L.slot_map_dirty = false;
        }
    }
    return updates;
}

// ---- Commit #13a: VRAM probe + dynamic n_slot/tile sizing ----

bool ModeBContext::init_dynamic(const llama_model * model,
                                int n_layer, int n_expert, int top_k, int device_id,
                                size_t reserved_vram_bytes) {
    if (!model || n_layer <= 0 || n_expert <= 0 || top_k <= 0) {
        std::fprintf(stderr, "[mode_b] init_dynamic: bad params "
                             "(model=%p n_layer=%d n_expert=%d top_k=%d)\n",
                     (const void *)model, n_layer, n_expert, top_k);
        return false;
    }

    // Probe VRAM via ggml's backend wrapper.
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_cuda_get_device_memory(device_id, &free_vram, &total_vram);
    if (total_vram == 0) {
        std::fprintf(stderr, "[mode_b] init_dynamic: ggml_backend_cuda_get_device_memory "
                             "returned 0; falling back to default n_slot=top_k=%d\n", top_k);
        free_vram = (size_t)4 * 1024 * 1024 * 1024;  // assume 4 GB free
    }

    // Compute per-slot byte cost from a probe layer's expert tensors.
    // Walk model tensors by name to find a representative MoE layer.
    size_t per_expert_up = 0, per_expert_gate = 0, per_expert_down = 0;
    for (int il = 0; il < n_layer; il++) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_up_exps.weight", il);
        ggml_tensor * up = llama_model_get_tensor(model, nm);
        if (!up) continue;
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", il);
        ggml_tensor * down = llama_model_get_tensor(model, nm);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_exps.weight", il);
        ggml_tensor * gate = llama_model_get_tensor(model, nm);
        if (!up || !down) continue;

        per_expert_up   = ggml_nbytes(up)   / (size_t)n_expert;
        per_expert_down = ggml_nbytes(down) / (size_t)n_expert;
        per_expert_gate = gate ? ggml_nbytes(gate) / (size_t)n_expert : 0;
        break;
    }
    if (per_expert_up == 0) {
        std::fprintf(stderr, "[mode_b] init_dynamic: no MoE layer found\n");
        return false;
    }
    // Per LAYER per SLOT byte cost: one slot worth of all three kinds.
    const size_t per_layer_per_slot = per_expert_up + per_expert_gate + per_expert_down;

    // Count active MoE layers (some hybrid arches have dense layers we skip).
    int n_moe_layers = 0;
    for (int il = 0; il < n_layer; il++) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_up_exps.weight", il);
        if (llama_model_get_tensor(model, nm)) n_moe_layers++;
    }
    if (n_moe_layers == 0) {
        std::fprintf(stderr, "[mode_b] init_dynamic: 0 MoE layers found\n");
        return false;
    }

    // Per-n_slot VRAM cost: n_moe_layers * per_layer_per_slot.
    const size_t bytes_per_slot_step = (size_t)n_moe_layers * per_layer_per_slot;

    // Available budget = free_vram - reserved_vram_bytes (compute graph + KV + activations).
    const size_t budget = (free_vram > reserved_vram_bytes)
        ? free_vram - reserved_vram_bytes : 0;

    // Largest n_slot that fits in budget.
    int n_slot_max = (bytes_per_slot_step > 0) ? (int)(budget / bytes_per_slot_step) : 0;
    n_slot_max = std::min(n_slot_max, n_expert);

    // Three tiers: optimal, recommended, floor.
    const int n_slot_optimal     = std::min(2 * top_k, n_expert);
    const int n_slot_recommended = std::min(top_k, n_expert);

    // Pick.
    int n_slot_actual;
    int tile_size;
    bool degraded;
    const char * label;
    if (n_slot_max >= n_slot_optimal) {
        n_slot_actual = n_slot_optimal;
        tile_size     = top_k;
        degraded      = false;
        label         = "optimal";
    } else if (n_slot_max >= n_slot_recommended) {
        n_slot_actual = n_slot_max;       // use everything we can up to optimal
        tile_size     = top_k;
        degraded      = false;
        label         = "recommended";
    } else if (n_slot_max >= 1) {
        n_slot_actual = n_slot_max;
        tile_size     = n_slot_max;
        degraded      = true;
        label         = "degraded";
    } else {
        std::fprintf(stderr, "[mode_b] init_dynamic: not enough VRAM for even 1 slot "
                             "(budget=%zu bytes, per_slot=%zu bytes)\n",
                     budget, bytes_per_slot_step);
        return false;
    }
    const int n_tiles = (top_k + tile_size - 1) / tile_size;

    // Save sizing decision.
    sizing_.free_vram_bytes        = free_vram;
    sizing_.total_vram_bytes       = total_vram;
    sizing_.per_layer_per_slot_bytes = per_layer_per_slot;
    sizing_.reserved_bytes         = reserved_vram_bytes;
    sizing_.top_k                  = top_k;
    sizing_.n_slot_optimal         = n_slot_optimal;
    sizing_.n_slot_recommended     = n_slot_recommended;
    sizing_.n_slot_actual          = n_slot_actual;
    sizing_.tile_size              = tile_size;
    sizing_.n_tiles                = n_tiles;
    sizing_.degraded_mode          = degraded;
    sizing_.mode_label             = label;

    // Print sizing decision. The "degraded" message is the one Chris specified.
    const double GiB = 1.0 / (1024.0 * 1024.0 * 1024.0);
    std::printf("[mode_b] VRAM probe: %.2f GB free / %.2f GB total (reserved %.2f GB for compute)\n",
                free_vram * GiB, total_vram * GiB, reserved_vram_bytes * GiB);
    std::printf("[mode_b]   per-slot cost: %.2f MB/expert all-kinds across %d MoE layers\n",
                per_layer_per_slot / (1024.0 * 1024.0), n_moe_layers);
    std::printf("[mode_b]   sizing tiers: optimal n_slot=%d, recommended=%d, max-fit=%d\n",
                n_slot_optimal, n_slot_recommended, n_slot_max);
    std::printf("[mode_b]   selected: n_slot=%d, tile_size=%d, n_tiles=%d (mode=%s)\n",
                n_slot_actual, tile_size, n_tiles, label);
    if (degraded) {
        std::printf("\n");
        std::printf("[mode_b] *** System VRAM is below the recommended size for this model. ***\n");
        std::printf("[mode_b] *** Dynamically adjusting n_slot=%d, tiling FFN across %d passes ***\n",
                    n_slot_actual, n_tiles);
        std::printf("[mode_b] *** to fit VRAM constraints. Quality is preserved (full top_k=%d ***\n",
                    top_k);
        std::printf("[mode_b] *** weighted sum); expect %dx kernel-launch overhead at the FFN. ***\n",
                    n_tiles);
        std::printf("\n");
    }

    return init(model, n_layer, n_expert, n_slot_actual, device_id);
}

// ---- Commit #10: Tier 2 pinned host buffer ----

// Read /proc/meminfo MemAvailable (Linux) and return bytes. Returns 0 on
// any read failure; caller should treat 0 as "couldn't determine, fall
// back to a conservative estimate."
static size_t read_mem_available_bytes() {
#ifdef __linux__
    FILE * f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    size_t result = 0;
    while (std::fgets(line, sizeof(line), f)) {
        // Format: "MemAvailable:   12345678 kB\n"
        if (std::strncmp(line, "MemAvailable:", 13) == 0) {
            size_t kb = 0;
            if (std::sscanf(line + 13, " %zu kB", &kb) == 1) {
                result = kb * 1024;
            }
            break;
        }
    }
    std::fclose(f);
    return result;
#else
    return 0;
#endif
}

bool ModeBContext::init_tier2_auto(double target_frac, double ram_headroom_frac) {
    if (!ctx_ || !backend_buffer_) {
        std::fprintf(stderr, "[mode_b] init_tier2_auto: ModeBContext not initialized\n");
        return false;
    }
    if (n_tier2_ > 0) {
        std::fprintf(stderr, "[mode_b] init_tier2_auto: already initialized (n_tier2=%d)\n", n_tier2_);
        return false;
    }
    if (target_frac <= 0.0 || target_frac > 1.0) {
        std::fprintf(stderr, "[mode_b] init_tier2_auto: bad target_frac=%.3f (must be in (0,1])\n",
                     target_frac);
        return false;
    }
    if (ram_headroom_frac < 0.0 || ram_headroom_frac >= 1.0) {
        std::fprintf(stderr, "[mode_b] init_tier2_auto: bad ram_headroom_frac=%.3f\n",
                     ram_headroom_frac);
        return false;
    }

    // Per-layer byte cost for one expert across all kinds (up + gate + down).
    // Sum across active MoE layers to get the per-expert-across-all-layers cost.
    size_t per_expert_bytes_total = 0;
    int    active_layers = 0;
    for (auto & L : layers_) {
        if (!L.up_slots) continue;
        active_layers++;
        per_expert_bytes_total += ggml_nbytes(L.up_slots)   / (size_t)n_slot_;
        per_expert_bytes_total += ggml_nbytes(L.down_slots) / (size_t)n_slot_;
        if (L.gate_slots) {
            per_expert_bytes_total += ggml_nbytes(L.gate_slots) / (size_t)n_slot_;
        }
    }
    if (per_expert_bytes_total == 0) {
        std::fprintf(stderr, "[mode_b] init_tier2_auto: no MoE layers — nothing to do\n");
        return false;
    }

    // Target by fraction.
    int n_tier2_by_frac = (int)std::round(target_frac * (double)n_expert_);

    // Cap by available host RAM.
    const size_t mem_avail = read_mem_available_bytes();
    int n_tier2_by_ram = n_expert_;  // sentinel = no ram cap if we can't query
    size_t budget = 0;
    if (mem_avail > 0) {
        budget = (size_t)((double)mem_avail * (1.0 - ram_headroom_frac));
        const size_t per_tier2_step_bytes = per_expert_bytes_total;  // adding 1 to n_tier2 costs this
        if (per_tier2_step_bytes > 0) {
            n_tier2_by_ram = (int)(budget / per_tier2_step_bytes);
        }
    }

    // Apply caps and floor: at minimum keep n_tier2 >= n_slot (otherwise no benefit
    // over plain Mode B), at maximum n_expert.
    int n_tier2_capped = std::min({ n_tier2_by_frac, n_tier2_by_ram, n_expert_ });
    int n_tier2 = std::max(n_tier2_capped, n_slot_);

    const char * cap_reason;
    if (n_tier2_capped < n_slot_)              cap_reason = "floored at n_slot";
    else if (n_tier2 == n_expert_)             cap_reason = "saturated to n_expert";
    else if (mem_avail > 0 &&
             n_tier2_by_ram < n_tier2_by_frac) cap_reason = "by RAM budget";
    else                                       cap_reason = "by target fraction";

    std::printf("[mode_b] init_tier2_auto: target_frac=%.2f -> %d, ram_avail=%.1f GB "
                "(headroom=%.0f%%, budget=%.1f GB) -> %d, %d MoE layers, "
                "per-expert all-kinds=%.2f MB\n",
                target_frac, n_tier2_by_frac,
                mem_avail / (1024.0 * 1024.0 * 1024.0),
                ram_headroom_frac * 100.0,
                budget / (1024.0 * 1024.0 * 1024.0),
                n_tier2_by_ram,
                active_layers,
                per_expert_bytes_total / (1024.0 * 1024.0));
    std::printf("[mode_b]   selected n_tier2=%d (%s)\n", n_tier2, cap_reason);

    return init_tier2(n_tier2);
}

bool ModeBContext::init_tier2(int n_tier2) {
    if (!ctx_ || !backend_buffer_) {
        std::fprintf(stderr, "[mode_b] init_tier2: ModeBContext not initialized\n");
        return false;
    }
    if (n_tier2 <= 0 || n_tier2 > n_expert_) {
        std::fprintf(stderr, "[mode_b] init_tier2: bad n_tier2=%d (n_expert=%d)\n",
                     n_tier2, n_expert_);
        return false;
    }
    if (n_tier2_ > 0) {
        std::fprintf(stderr, "[mode_b] init_tier2: already initialized (n_tier2=%d)\n", n_tier2_);
        return false;
    }

    n_tier2_ = n_tier2;
    size_t total_pinned_bytes = 0;
    int active_layers = 0;
    for (size_t il = 0; il < layers_.size(); il++) {
        auto & L = layers_[il];
        if (!L.up_slots) continue;  // skipped (non-MoE) layer

        // Sizes are derived from slot tensor metadata (per_slot_bytes).
        const size_t up_bytes   = (size_t)n_tier2 * (ggml_nbytes(L.up_slots)   / (size_t)n_slot_);
        const size_t down_bytes = (size_t)n_tier2 * (ggml_nbytes(L.down_slots) / (size_t)n_slot_);
        const size_t gate_bytes = L.gate_slots
            ? (size_t)n_tier2 * (ggml_nbytes(L.gate_slots) / (size_t)n_slot_)
            : 0;

        // 64-byte aligned for SIMD-friendly layout. cudaHostRegister wants
        // page-aligned in practice but accepts arbitrary alignment.
        auto alloc_pinned = [](size_t bytes) -> void * {
            if (bytes == 0) return nullptr;
            void * p = std::aligned_alloc(64, ((bytes + 63) / 64) * 64);
            if (!p) return nullptr;
            // Register with CUDA for page-locked DMA. Failure here is non-fatal
            // — we'll just have a regular host buffer (still functions, just
            // not pinned for fast PCIe).
            if (!ggml_backend_cuda_register_host_buffer(p, bytes)) {
                static bool warned = false;
                if (!warned) {
                    std::fprintf(stderr, "[mode_b] init_tier2: cuda host registration failed "
                                         "(non-fatal — buffer remains unpinned)\n");
                    warned = true;
                }
            }
            return p;
        };

        L.tier2_up_buf   = alloc_pinned(up_bytes);
        L.tier2_down_buf = alloc_pinned(down_bytes);
        L.tier2_gate_buf = alloc_pinned(gate_bytes);

        if (!L.tier2_up_buf || !L.tier2_down_buf || (gate_bytes > 0 && !L.tier2_gate_buf)) {
            std::fprintf(stderr, "[mode_b] init_tier2: allocation failed at layer %zu "
                                 "(needed up=%zu down=%zu gate=%zu)\n",
                         il, up_bytes, down_bytes, gate_bytes);
            n_tier2_ = 0;
            return false;
        }

        L.tier2_to_expert.assign((size_t)n_tier2, -1);
        L.expert_to_tier2.assign((size_t)n_expert_, -1);
        L.tier2_next_write = 0;

        // Seed Tier 2 with the slots' current contents (experts 0..n_slot-1
        // populated at init). This means a "warm" Tier 2 already exists for
        // those experts, so refresh_slots after first prompt picks them up
        // via the fast path even before promote_to_tier2 is called.
        for (int s = 0; s < n_slot_; s++) {
            const int e = L.slot_to_expert[(size_t)s];
            if (e < 0) continue;
            // Tier 2 indexes 0..n_tier2-1; first n_slot entries mirror slots.
            if (s >= n_tier2) break;
            L.tier2_to_expert[(size_t)s] = e;
            L.expert_to_tier2[(size_t)e] = s;
            // Copy expert weights from CPU mirror into Tier 2 buffer
            const size_t up_per   = ggml_nbytes(L.up_slots) / (size_t)n_slot_;
            ggml_backend_tensor_get(L.src_up, (uint8_t *)L.tier2_up_buf + (size_t)s * up_per,
                                    (size_t)e * up_per, up_per);
            const size_t dn_per   = ggml_nbytes(L.down_slots) / (size_t)n_slot_;
            ggml_backend_tensor_get(L.src_down, (uint8_t *)L.tier2_down_buf + (size_t)s * dn_per,
                                    (size_t)e * dn_per, dn_per);
            if (L.gate_slots && L.tier2_gate_buf) {
                const size_t gp = ggml_nbytes(L.gate_slots) / (size_t)n_slot_;
                ggml_backend_tensor_get(L.src_gate, (uint8_t *)L.tier2_gate_buf + (size_t)s * gp,
                                        (size_t)e * gp, gp);
            }
        }
        L.tier2_next_write = std::min(n_slot_, n_tier2);

        total_pinned_bytes += up_bytes + down_bytes + gate_bytes;
        active_layers++;
    }

    std::printf("[mode_b] Tier 2 active: n_tier2=%d, %d layers, %.2f GB pinned host RAM\n",
                n_tier2, active_layers, total_pinned_bytes / (1024.0 * 1024.0 * 1024.0));
    return true;
}

int ModeBContext::promote_to_tier2(int layer, const int32_t * experts, int count) {
    if (n_tier2_ <= 0) return 0;
    if (layer < 0 || layer >= (int)layers_.size()) return 0;
    auto & L = layers_[layer];
    if (!L.up_slots || !L.tier2_up_buf) return 0;

    const size_t up_per   = ggml_nbytes(L.up_slots)   / (size_t)n_slot_;
    const size_t dn_per   = ggml_nbytes(L.down_slots) / (size_t)n_slot_;
    const size_t gt_per   = (L.gate_slots && L.tier2_gate_buf)
        ? ggml_nbytes(L.gate_slots) / (size_t)n_slot_ : 0;

    int new_promotions = 0;
    for (int i = 0; i < count; i++) {
        const int32_t e = experts[i];
        if (e < 0 || e >= n_expert_) continue;
        if (L.expert_to_tier2[(size_t)e] >= 0) continue;  // already there

        // Pick a Tier 2 victim slot via round-robin write pointer. Doesn't
        // evict experts that are CURRENTLY in Tier 1 (prefer to keep those
        // hot in Tier 2 too — fast re-promote on miss). Walks at most n_tier2
        // probes before giving up.
        int t2_idx = -1;
        for (int probe = 0; probe < n_tier2_; probe++) {
            int candidate = L.tier2_next_write;
            L.tier2_next_write = (L.tier2_next_write + 1) % n_tier2_;
            const int32_t occupant = L.tier2_to_expert[(size_t)candidate];
            // Skip if occupant is currently in Tier 1.
            if (occupant >= 0 && L.expert_to_slot[(size_t)occupant] >= 0) continue;
            t2_idx = candidate;
            break;
        }
        if (t2_idx < 0) {
            // All Tier 2 slots are also Tier 1 residents — pick any (round-robin
            // landed back at start). This is an over-pinned regime; rare.
            t2_idx = L.tier2_next_write;
            L.tier2_next_write = (L.tier2_next_write + 1) % n_tier2_;
        }

        // Evict whatever was here.
        int32_t prev = L.tier2_to_expert[(size_t)t2_idx];
        if (prev >= 0) L.expert_to_tier2[(size_t)prev] = -1;

        // Copy CPU mirror → Tier 2 pinned buffer.
        ggml_backend_tensor_get(L.src_up,
                                (uint8_t *)L.tier2_up_buf + (size_t)t2_idx * up_per,
                                (size_t)e * up_per, up_per);
        ggml_backend_tensor_get(L.src_down,
                                (uint8_t *)L.tier2_down_buf + (size_t)t2_idx * dn_per,
                                (size_t)e * dn_per, dn_per);
        if (gt_per > 0) {
            ggml_backend_tensor_get(L.src_gate,
                                    (uint8_t *)L.tier2_gate_buf + (size_t)t2_idx * gt_per,
                                    (size_t)e * gt_per, gt_per);
        }

        L.tier2_to_expert[(size_t)t2_idx] = e;
        L.expert_to_tier2[(size_t)e] = t2_idx;
        new_promotions++;
        total_tier2_promotions_++;
    }
    return new_promotions;
}

int ModeBContext::on_gate_fired_sync(int layer, const int32_t * experts, int count) {
    if (layer < 0 || layer >= (int)layers_.size()) return 0;
    if (!graceful_mask_) return 0;  // hard-mask constrains the gate; no swap needed
    auto & L = layers_[layer];
    if (!L.up_slots) return 0;
    if (count <= 0 || !experts) return 0;

    // Build the unique set of selected experts for this batch.
    // Small set in practice (top_k * n_tokens, capped by n_expert).
    std::vector<int32_t> selected;
    selected.reserve((size_t)count);
    {
        std::vector<bool> seen((size_t)n_expert_, false);
        for (int i = 0; i < count; i++) {
            const int32_t e = experts[i];
            if (e < 0 || e >= n_expert_) continue;
            if (!seen[(size_t)e]) {
                seen[(size_t)e] = true;
                selected.push_back(e);
            }
        }
    }

    // Identify misses (selected but not slot-resident).
    std::vector<int32_t> misses;
    misses.reserve(selected.size());
    for (int32_t e : selected) {
        if (L.expert_to_slot[(size_t)e] < 0) misses.push_back(e);
    }
    if (misses.empty()) return 0;

    // Build the set of "evictable" slots: slots holding experts that are
    // NOT in `selected`. These are safe to overwrite without breaking the
    // current batch's FFN computation. If we can't find enough evictable
    // slots to fit all misses, we evict the LRU residents we have to.
    std::vector<bool> selected_mask((size_t)n_expert_, false);
    for (int32_t e : selected) selected_mask[(size_t)e] = true;

    std::vector<int32_t> safe_slots;
    safe_slots.reserve((size_t)n_slot_);
    for (int s = 0; s < n_slot_; s++) {
        const int32_t occ = L.slot_to_expert[(size_t)s];
        if (occ < 0 || !selected_mask[(size_t)occ]) safe_slots.push_back(s);
    }

    // If there aren't enough safe slots, fall back to LRU eviction (which
    // can clobber currently-needed experts — degraded quality but at least
    // forward progress). Walk lru from the back.
    auto pick_lru_slot = [&]() -> int {
        for (auto it = L.lru.rbegin(); it != L.lru.rend(); ++it) {
            int32_t e = *it;
            int32_t s = L.expert_to_slot[(size_t)e];
            if (s >= 0) return s;
        }
        return 0;  // pathological fallback
    };

    int swaps = 0;
    for (int32_t e : misses) {
        int slot;
        if (!safe_slots.empty()) {
            slot = safe_slots.back();
            safe_slots.pop_back();
        } else {
            slot = pick_lru_slot();
        }
        if (!_page_in_expert(layer, e, slot)) continue;
        // Move e to MRU
        for (auto it = L.lru.begin(); it != L.lru.end(); ++it) {
            if (*it == e) { L.lru.erase(it); break; }
        }
        L.lru.insert(L.lru.begin(), e);
        const size_t cap = (size_t)n_slot_ * 2;
        if (L.lru.size() > cap) L.lru.resize(cap);
        swaps++;
    }
    if (swaps == 0) return 0;

    // Rebuild slot_map for this layer and push to GPU. The CUDA stream
    // serializes this write before the next graph op (the get_rows that
    // reads slot_map), so the FFN sees the updated mapping.
    std::vector<int32_t> map_buf((size_t)n_expert_, 0);
    std::vector<float>   mask_buf((size_t)n_expert_, 0.0f);  // graceful: all 0
    for (int e = 0; e < n_expert_; e++) {
        int32_t s = L.expert_to_slot[(size_t)e];
        map_buf[(size_t)e] = (s >= 0) ? s : 0;
    }
    set_slot_map(layer, map_buf.data());
    if (L.valid_mask) {
        ggml_backend_tensor_set(L.valid_mask, mask_buf.data(),
                                0, (size_t)n_expert_ * sizeof(float));
    }
    L.slot_map_dirty = false;

    total_sync_swaps_ += (uint64_t)swaps;
    total_pages_in_   += (uint64_t)swaps;
    return swaps;
}

int ModeBContext::tile_size_for_top_k(int top_k) const {
    if (top_k <= 0) return 0;
    // Diagnostic env var: force tiling even when slots have headroom. Useful
    // for isolating whether n_tiles>1 graph topology is broken vs swap pressure.
    // KNOWN BUG (2026-05-04): n_tiles>1 produces garbage output (see handoff).
    if (const char * env = std::getenv("MOE_FORCE_TILE_SIZE")) {
        int forced = std::atoi(env);
        if (forced > 0 && forced < top_k) return forced;
    }
    // Prefer init_dynamic's decision when populated (sizing_.tile_size > 0).
    if (sizing_.tile_size > 0 && sizing_.tile_size < top_k) {
        return sizing_.tile_size;
    }
    // Explicit init() with small n_slot: tile at n_slot.
    if (n_slot_ > 0 && n_slot_ < top_k) {
        return n_slot_;
    }
    return top_k;  // no tiling — slots can hold full top_k
}

int ModeBContext::n_tiles_for_top_k(int top_k) const {
    const int ts = tile_size_for_top_k(top_k);
    if (ts <= 0 || ts >= top_k) return 1;
    return (top_k + ts - 1) / ts;
}

int ModeBContext::on_tile_fired_sync(int layer, int /*tile_idx*/,
                                     const int32_t * experts, int count) {
    // Same swap math as on_gate_fired_sync — the existing routine already
    // builds safe-slots from "experts not in `selected`" semantics, which
    // is correct per-tile when `selected` is just this tile's experts.
    // tile_idx is reserved for future per-tile diagnostics / policy hooks.
    return on_gate_fired_sync(layer, experts, count);
}

int ModeBContext::tier2_index(int layer, int expert) const {
    if (n_tier2_ <= 0) return -1;
    if (layer < 0 || layer >= (int)layers_.size()) return -1;
    if (expert < 0 || expert >= n_expert_) return -1;
    const auto & L = layers_[layer];
    if (L.expert_to_tier2.empty()) return -1;
    return L.expert_to_tier2[(size_t)expert];
}

} // namespace moe_orch
