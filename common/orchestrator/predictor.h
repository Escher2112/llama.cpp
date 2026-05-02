// predictor.h — L1 MLP expert-routing predictor for the llama.cpp orchestrator fork.
//
// Loads weights from the .bin produced by scripts/export_predictor_to_bin.py and
// runs inference: hidden_state -> top-K predicted next-layer expert IDs.
//
// Per-call cost: 2 GEMMs + GELU + topK on hidden_dim=2048, hidden_units=256,
// n_experts=128. Sub-millisecond on a single CPU core; trivially parallelizable
// across (layer, horizon) heads if needed.
//
// Drop into llama.cpp/common/orchestrator/ once the fork branch is created.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moe_orch {

enum PredictorDType : uint32_t {
    PRED_DTYPE_FP32 = 0,
    PRED_DTYPE_FP16 = 1,
};

struct PredictorHeader {
    char     magic[4];          // "MPRD"
    uint32_t version;           // 1
    uint32_t hidden_dim;
    uint32_t hidden_units;
    uint32_t n_experts;
    uint32_t num_heads;
    uint32_t dtype;             // PredictorDType
    uint32_t reserved0;
    uint32_t reserved1;
};
static_assert(sizeof(PredictorHeader) == 36, "PredictorHeader layout drift");

struct PredictorIndexEntry {
    int32_t  source_layer;
    int32_t  horizon;
    uint64_t offset_bytes;      // from start of file
    uint64_t size_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
};
static_assert(sizeof(PredictorIndexEntry) == 32, "PredictorIndexEntry layout drift");

// Per-(layer, horizon) head, with raw pointers into the mmaped weight blob.
// Pointers are dtype-typed per the header's dtype field.
struct PredictorHead {
    int32_t       source_layer;
    int32_t       horizon;
    const void *  w1;           // (hidden_dim, hidden_units) row-major
    const void *  b1;           // (hidden_units,)
    const void *  w2;           // (hidden_units, n_experts) row-major
    const void *  b2;           // (n_experts,)
};

class MLPPredictor {
public:
    // Loads the predictor file (mmap on Unix, MapViewOfFile on Windows — caller
    // owns the lifetime via close() / dtor). Returns true on success.
    bool load(const char * path);
    void close();

    // Predict the top-K experts at (source_layer + horizon) given the input
    // hidden state at source_layer.
    //   hidden_state    : float pointer of length hidden_dim_
    //   out_indices     : caller-provided int32 buffer of length top_k
    //   out_confidences : OPTIONAL caller-provided float buffer of length top_k.
    //                     When non-null, filled with softmax(logits)[top_k_indices]
    //                     in the same order as out_indices. These are the predictor's
    //                     confidence in each named expert (sum across all 128 = 1.0).
    //                     Used by cache.prefetch_to_l1 to gate L2->L1 promotion.
    // Returns false if no head exists for the requested (source_layer, horizon).
    bool predict_top_k(
        int32_t source_layer,
        int32_t horizon,
        const float * hidden_state,
        int32_t top_k,
        int32_t * out_indices,
        float * out_confidences = nullptr) const;

    uint32_t hidden_dim()    const { return hidden_dim_; }
    uint32_t hidden_units()  const { return hidden_units_; }
    uint32_t n_experts()     const { return n_experts_; }
    uint32_t num_heads()     const { return num_heads_; }
    PredictorDType dtype()   const { return PredictorDType(dtype_); }

private:
    // Lookup table: head_lookup_[(layer, horizon)] -> head index in heads_
    int32_t head_index(int32_t source_layer, int32_t horizon) const;

    // mmap state
    void *   map_base_   = nullptr;
    size_t   map_size_   = 0;
#ifdef _WIN32
    void *   map_handle_ = nullptr;
    void *   file_handle_= nullptr;
#else
    int      fd_         = -1;
#endif

    // Cached metadata
    uint32_t hidden_dim_   = 0;
    uint32_t hidden_units_ = 0;
    uint32_t n_experts_    = 0;
    uint32_t num_heads_    = 0;
    uint32_t dtype_        = PRED_DTYPE_FP16;

    std::vector<PredictorHead> heads_;
    // Sparse 2D lookup: vector indexed by source_layer, each entry maps horizon -> head index.
    std::vector<std::vector<int32_t>> head_by_layer_horizon_;
};

} // namespace moe_orch
