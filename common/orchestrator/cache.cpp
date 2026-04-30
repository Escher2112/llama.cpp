// cache.cpp — three-tier expert cache implementation.

#include "cache.h"

#include <algorithm>
#include <chrono>
#include <cassert>

namespace moe_orch {

// ---------- LRU helpers ----------

void ThreeTierCache::LRU::bump(int32_t e) {
    auto it = by_expert.find(e);
    if (it == by_expert.end()) return;
    order.erase(it->second);
    order.push_front(e);
    it->second = order.begin();
}

void ThreeTierCache::LRU::insert(int32_t e) {
    order.push_front(e);
    by_expert[e] = order.begin();
}

bool ThreeTierCache::LRU::remove(int32_t e) {
    auto it = by_expert.find(e);
    if (it == by_expert.end()) return false;
    order.erase(it->second);
    by_expert.erase(it);
    return true;
}

int32_t ThreeTierCache::LRU::pop_oldest() {
    if (order.empty()) return -1;
    int32_t e = order.back();
    order.pop_back();
    by_expert.erase(e);
    return e;
}

// ---------- ctor ----------

ThreeTierCache::ThreeTierCache(int32_t n_layers,
                               int32_t n_experts_per_layer,
                               int32_t l1_capacity,
                               int32_t l2_capacity,
                               bool    shadow_mode)
    : n_layers_(n_layers),
      n_experts_per_layer_(n_experts_per_layer),
      l1_capacity_(l1_capacity),
      l2_capacity_(l2_capacity),
      shadow_mode_(shadow_mode),
      locations_((size_t)n_layers * n_experts_per_layer, ExpertLocation::L3_NVME),
      l1_lru_((size_t)n_layers),
      l2_lru_((size_t)n_layers),
      access_count_((size_t)n_layers * n_experts_per_layer, 0),
      predictor_confidence_((size_t)n_layers * n_experts_per_layer, 0.0f),
      router_weight_((size_t)n_layers * n_experts_per_layer, 0.0f),
      per_layer_hits_L1_((size_t)n_layers, 0),
      per_layer_hits_L2_((size_t)n_layers, 0),
      per_layer_misses_L3_((size_t)n_layers, 0)
{
    events_.reserve(1 << 16);
}

// ---------- router weight update ----------

void ThreeTierCache::update_router_weight(int32_t layer, int32_t expert, float weight) {
    // EMA with alpha=0.3 — stable enough across consecutive accesses, responsive
    // enough to track expert-importance shifts as the conversation moves.
    const size_t idx = _idx(layer, expert);
    constexpr float alpha = 0.3f;
    router_weight_[idx] = (1.0f - alpha) * router_weight_[idx] + alpha * weight;
}

// ---------- queries ----------

ExpertLocation ThreeTierCache::location_of(int32_t layer, int32_t expert) const {
    return locations_[_idx(layer, expert)];
}

bool ThreeTierCache::is_in_l1(int32_t layer, int32_t expert) const {
    return locations_[_idx(layer, expert)] == ExpertLocation::L1_VRAM;
}

std::vector<int32_t> ThreeTierCache::l1_contents(int32_t layer) const {
    std::vector<int32_t> out;
    out.reserve(l1_lru_[layer].order.size());
    for (auto e : l1_lru_[layer].order) out.push_back(e);
    return out;
}

std::vector<int32_t> ThreeTierCache::l2_contents(int32_t layer) const {
    std::vector<int32_t> out;
    out.reserve(l2_lru_[layer].order.size());
    for (auto e : l2_lru_[layer].order) out.push_back(e);
    return out;
}

// ---------- access path ----------

ExpertLocation ThreeTierCache::access(int32_t layer, int32_t expert) {
    const size_t idx = _idx(layer, expert);
    ExpertLocation loc = locations_[idx];
    access_count_[idx]++;

    switch (loc) {
        case ExpertLocation::L1_VRAM:
            l1_lru_[layer].bump(expert);
            _log(CacheEvent::HIT_L1, layer, expert);
            return loc;

        case ExpertLocation::L2_RAM:
            _log(CacheEvent::HIT_L2, layer, expert);
            _promote(layer, expert, ExpertLocation::L2_RAM, ExpertLocation::L1_VRAM);
            return ExpertLocation::L2_RAM;

        case ExpertLocation::L3_NVME:
        default:
            _log(CacheEvent::MISS_L3, layer, expert);
            _promote(layer, expert, ExpertLocation::L3_NVME, ExpertLocation::L1_VRAM);
            return ExpertLocation::L3_NVME;
    }
}

// ---------- prefetch paths ----------

void ThreeTierCache::prefetch_to_l1(int32_t layer,
                                    const int32_t * experts,
                                    int32_t count,
                                    const float * confidences) {
    // Skip-if-resident policy:
    //   already in L1 → bump LRU, done.
    //   already in L2 → bump L2 LRU only. The predictor's interest keeps the
    //                   expert from aging to L3, but we don't promote to L1
    //                   speculatively. If the model actually accesses it,
    //                   access() handles the L2→L1 promotion. This eliminates
    //                   the redundant "promote to L1, evict, promote again"
    //                   churn that dominated Mode B in the warm regime
    //                   (per STATUS_apr30_morning: 99.7% of Mode B promotions
    //                   were re-evicted before being hit).
    //   in L3        → promote L3→L1 (with eviction). This is the predictor's
    //                   actual job: pulling cold experts into VRAM ahead of
    //                   the access.
    for (int32_t i = 0; i < count; i++) {
        const int32_t e = experts[i];
        const float   c = confidences ? confidences[i] : 1.0f;
        predictor_confidence_[_idx(layer, e)] = c;
        ExpertLocation loc = locations_[_idx(layer, e)];
        if (loc == ExpertLocation::L1_VRAM) {
            l1_lru_[layer].bump(e);
        } else if (loc == ExpertLocation::L2_RAM) {
            l2_lru_[layer].bump(e);
        } else {
            _promote(layer, e, ExpertLocation::L3_NVME, ExpertLocation::L1_VRAM);
        }
    }
}

void ThreeTierCache::prefetch_to_l2(int32_t layer,
                                    const int32_t * experts,
                                    int32_t count) {
    for (int32_t i = 0; i < count; i++) {
        const int32_t e = experts[i];
        ExpertLocation loc = locations_[_idx(layer, e)];
        if (loc == ExpertLocation::L2_RAM) {
            l2_lru_[layer].bump(e);
        } else if (loc == ExpertLocation::L3_NVME) {
            _promote(layer, e, ExpertLocation::L3_NVME, ExpertLocation::L2_RAM);
        }
        // L1-resident: leave alone (it's already hotter than L2).
    }
}

// ---------- internals ----------

void ThreeTierCache::_promote(int32_t layer, int32_t expert,
                              ExpertLocation from, ExpertLocation to) {
    if (to == ExpertLocation::L1_VRAM) {
        _make_room_in_l1(layer);
        l1_lru_[layer].insert(expert);
        if (from == ExpertLocation::L2_RAM) l2_lru_[layer].remove(expert);
        locations_[_idx(layer, expert)] = ExpertLocation::L1_VRAM;
        if (from == ExpertLocation::L2_RAM) {
            _log(CacheEvent::PROMOTE_L2_TO_L1, layer, expert);
        } else {
            _log(CacheEvent::PROMOTE_L3_TO_L1, layer, expert);
        }
    } else if (to == ExpertLocation::L2_RAM) {
        _make_room_in_l2(layer);
        l2_lru_[layer].insert(expert);
        locations_[_idx(layer, expert)] = ExpertLocation::L2_RAM;
        _log(CacheEvent::PROMOTE_L3_TO_L2, layer, expert);
    }
}

void ThreeTierCache::_make_room_in_l1(int32_t layer) {
    while ((int32_t)l1_lru_[layer].size() >= l1_capacity_) {
        int32_t victim = _pick_l1_victim(layer);
        l1_lru_[layer].remove(victim);
        // Demote to L2 (if room; otherwise this triggers L2->L3 eviction below).
        _make_room_in_l2(layer);
        l2_lru_[layer].insert(victim);
        locations_[_idx(layer, victim)] = ExpertLocation::L2_RAM;
        _log(CacheEvent::EVICT_L1_TO_L2, layer, victim);
    }
}

void ThreeTierCache::_make_room_in_l2(int32_t layer) {
    while ((int32_t)l2_lru_[layer].size() >= l2_capacity_) {
        int32_t victim = l2_lru_[layer].pop_oldest();   // plain LRU
        if (victim < 0) break;
        locations_[_idx(layer, victim)] = ExpertLocation::L3_NVME;
        _log(CacheEvent::EVICT_L2_TO_L3, layer, victim);
    }
}

int32_t ThreeTierCache::_pick_l1_victim(int32_t layer) {
    // Router-weight-aware LFU.
    //
    // The model's own softmax router tells us how much it valued each
    // expert at the most recent step — that's strictly better signal for
    // eviction than a synthetic predictor confidence we'd have to estimate.
    //
    // score = freq + 100 * router_weight + 10 * predictor_confidence
    // Lower score = better victim. router_weight dominates predictor_conf
    // because router_weight is observed truth; predictor_conf is a guess.
    int32_t best_victim   = -1;
    double  best_score    = 1e308;
    for (auto e : l1_lru_[layer].order) {
        const size_t idx = _idx(layer, e);
        const double score = (double)access_count_[idx]
                           + 100.0 * (double)router_weight_[idx]
                           +  10.0 * (double)predictor_confidence_[idx];
        if (score < best_score) {
            best_score = score;
            best_victim = e;
        }
    }
    return best_victim;
}

void ThreeTierCache::_log(CacheEvent::Kind kind, int32_t layer, int32_t expert) {
    // Update running counters first — these are stats source-of-truth and must
    // not be lost when events_ is cleared or capped.
    const bool layer_in_range = layer >= 0 && layer < n_layers_;
    switch (kind) {
        case CacheEvent::HIT_L1:
            total_hits_L1_++;
            if (layer_in_range) per_layer_hits_L1_[layer]++;
            break;
        case CacheEvent::HIT_L2:
            total_hits_L2_++;
            if (layer_in_range) per_layer_hits_L2_[layer]++;
            break;
        case CacheEvent::MISS_L3:
            total_misses_L3_++;
            if (layer_in_range) per_layer_misses_L3_[layer]++;
            break;
        case CacheEvent::PROMOTE_L2_TO_L1:
        case CacheEvent::PROMOTE_L3_TO_L1:
            total_promotions_to_L1_++;
            break;
        case CacheEvent::EVICT_L1_TO_L2:
            total_evictions_L1_to_L2_++;
            break;
        case CacheEvent::EVICT_L2_TO_L3:
            total_evictions_L2_to_L3_++;
            break;
        default: break;
    }

    // Audit log (bounded). Cap silently — stats survive in the counters above.
    if (events_.size() >= MAX_EVENTS) return;
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    events_.push_back(CacheEvent{ns, layer, expert, kind});
}

// ---------- stats ----------

CacheStats ThreeTierCache::stats() const {
    // Read directly from running counters — stats survive clear_events() and
    // the events_ MAX_EVENTS cap. (Pre-fix this walked events_; would lose data
    // mid-run at high churn.)
    CacheStats s;
    s.events_total        = events_.size();   // audit log size, not a stat per se
    s.hits_L1             = total_hits_L1_;
    s.hits_L2             = total_hits_L2_;
    s.misses_L3           = total_misses_L3_;
    s.promotions_to_L1    = total_promotions_to_L1_;
    s.evictions_L1_to_L2  = total_evictions_L1_to_L2_;
    s.evictions_L2_to_L3  = total_evictions_L2_to_L3_;
    s.per_layer_hits_L1   = per_layer_hits_L1_;
    s.per_layer_hits_L2   = per_layer_hits_L2_;
    s.per_layer_misses_L3 = per_layer_misses_L3_;

    const uint64_t total_accesses = s.hits_L1 + s.hits_L2 + s.misses_L3;
    s.l1_hit_rate = total_accesses ? (double)s.hits_L1 / (double)total_accesses : 0.0;
    return s;
}

} // namespace moe_orch
