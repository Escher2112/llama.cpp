// cache.h — three-tier expert cache for the orchestrator fork.
//
// L1 = VRAM hot pool, dynamic slot allocator
// L2 = system RAM warm pool, larger
// L3 = NVMe cold tier, all weights live here always
//
// In *shadow mode* (initial integration), this is pure bookkeeping — no actual
// weight movement. Once correctness is validated, the live mode plugs in real
// cudaMemcpyAsync calls + event sync.
//
// Mirrors the Python ThreeTierCache semantics from src/moe_engine/cache/three_tier.py.

#pragma once

#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace moe_orch {

enum class ExpertLocation : uint8_t {
    L1_VRAM   = 0,
    L2_RAM    = 1,
    L3_NVME   = 2,
};

struct CacheEvent {
    int64_t  timestamp_ns;
    int32_t  layer;
    int32_t  expert;
    enum Kind : uint8_t {
        HIT_L1                = 0,
        HIT_L2                = 1,
        MISS_L3               = 2,
        PROMOTE_L2_TO_L1      = 3,
        PROMOTE_L3_TO_L1      = 4,
        PROMOTE_L3_TO_L2      = 5,
        EVICT_L1_TO_L2        = 6,
        EVICT_L2_TO_L3        = 7,
    } kind;
};

struct CacheStats {
    uint64_t events_total            = 0;
    uint64_t hits_L1                 = 0;
    uint64_t hits_L2                 = 0;
    uint64_t misses_L3               = 0;
    uint64_t promotions_to_L1        = 0;
    uint64_t evictions_L1_to_L2      = 0;
    uint64_t evictions_L2_to_L3      = 0;
    double   l1_hit_rate             = 0.0;

    // Per-layer breakdown. Indexed by layer; size = n_layers.
    std::vector<uint64_t> per_layer_hits_L1;
    std::vector<uint64_t> per_layer_hits_L2;
    std::vector<uint64_t> per_layer_misses_L3;
};

class ThreeTierCache {
public:
    // Per-layer capacities (in number of expert slots, not bytes).
    // n_layers and n_experts_per_layer come from the model config.
    ThreeTierCache(int32_t n_layers,
                   int32_t n_experts_per_layer,
                   int32_t l1_capacity,
                   int32_t l2_capacity,
                   bool    shadow_mode = true);

    // Queries
    ExpertLocation location_of(int32_t layer, int32_t expert) const;
    bool is_in_l1(int32_t layer, int32_t expert) const;

    // Access path — called by the routing observer when an expert is actually
    // selected by the model. Updates LRU + access counts. Returns the tier
    // the expert was found in (NOT promoted to).
    ExpertLocation access(int32_t layer, int32_t expert);

    // Router-weight update — called when we observe the router's softmax
    // weight for an expert at a given layer. Signal is used by the eviction
    // policy: low-weight experts evict first.
    void update_router_weight(int32_t layer, int32_t expert, float weight);

    // Prefetch path — called by the L1 (MLP) predictor with the experts it
    // wants resident in VRAM by the next access at this layer.
    // `confidences` is optional; values default to 1.0 if not provided.
    void prefetch_to_l1(int32_t layer,
                        const int32_t * experts,
                        int32_t count,
                        const float * confidences = nullptr);

    // Conversation-level prefetch — called by the L2 (Hopfield) predictor.
    // Promotes experts from L3 to L2 if not already there. L1-resident
    // experts are left alone.
    void prefetch_to_l2(int32_t layer,
                        const int32_t * experts,
                        int32_t count);

    // Diagnostics
    CacheStats stats() const;
    const std::vector<CacheEvent> & events() const { return events_; }
    void clear_events() { events_.clear(); }

    // Snapshot — for debugging. Returns set of expert IDs currently in L1
    // for the given layer.
    std::vector<int32_t> l1_contents(int32_t layer) const;
    std::vector<int32_t> l2_contents(int32_t layer) const;

private:
    int32_t n_layers_;
    int32_t n_experts_per_layer_;
    int32_t l1_capacity_;
    int32_t l2_capacity_;
    bool    shadow_mode_;

    // Location map: locations_[layer * n_experts_per_layer_ + expert]
    std::vector<ExpertLocation> locations_;

    // Per-layer LRU lists. We use std::list<int32_t> for the LRU order and
    // a parallel hash map for O(1) lookup of the iterator. Same pattern as
    // the canonical LRU-cache idiom.
    struct LRU {
        std::list<int32_t> order;                                  // front = newest, back = oldest
        std::unordered_map<int32_t, std::list<int32_t>::iterator> by_expert;
        size_t size() const { return order.size(); }
        bool   contains(int32_t e) const { return by_expert.find(e) != by_expert.end(); }
        void   bump(int32_t e);                                    // move to front
        void   insert(int32_t e);                                  // insert at front
        bool   remove(int32_t e);                                  // returns true if present
        int32_t pop_oldest();                                      // returns and removes back
    };
    std::vector<LRU> l1_lru_;
    std::vector<LRU> l2_lru_;

    // Access counts for LFU + predictor confidence weighting on L1 evictions.
    std::vector<uint64_t> access_count_;             // [layer * n_experts + expert]
    std::vector<float>    predictor_confidence_;     // [layer * n_experts + expert]
    // Most recent router softmax weight observed for this expert. Drives
    // weight-aware eviction: low-weight experts are preferred victims.
    std::vector<float>    router_weight_;            // [layer * n_experts + expert]

    // Event log (capped to avoid OOM in long runs).
    std::vector<CacheEvent> events_;
    static constexpr size_t MAX_EVENTS = 1u << 22;   // ~4M events; ~64 MB at 16 B/event

    // Internals
    void _promote(int32_t layer, int32_t expert,
                  ExpertLocation from, ExpertLocation to);
    void _make_room_in_l1(int32_t layer);
    void _make_room_in_l2(int32_t layer);
    int32_t _pick_l1_victim(int32_t layer);
    void _log(CacheEvent::Kind kind, int32_t layer, int32_t expert);

    inline size_t _idx(int32_t layer, int32_t expert) const {
        return (size_t)layer * n_experts_per_layer_ + (size_t)expert;
    }
};

} // namespace moe_orch
