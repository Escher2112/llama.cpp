// hopfield.h — L2 (per-window) Hopfield expert-routing predictor.
//
// Modern Hopfield retrieval over a calibration-built memory bank of
// (conversation_centroid -> per-layer expert distribution) pairs. At inference,
// the orchestrator queries with the running conversation embedding and gets a
// soft mixture of stored expert distributions back; top-N indices feed Tier 2
// of the slot cache.
//
// Compute per call:
//   sim[p]      = (keys[p] . query) / sqrt(hidden_dim)      for p in [0,n_patterns)
//   weights[p]  = softmax(beta * sim[p])
//   dist[e]     = sum_p weights[p] * values[layer][p][e]    for e in [0,n_experts)
//   return topk(dist, top_n)
//
// Cost: 1 (n_patterns x hidden_dim) GEMV + 1 (n_patterns x n_experts) GEMV +
//       softmax + topk. With n_patterns ~256 and AVX2-fp16, sub-ms per call.
//
// File format produced by scripts/export_hopfield_to_bin.py:
//   Header (32 bytes):
//     char     magic[4]      = "HFLD"
//     uint32   version       = 1
//     uint32   n_patterns
//     uint32   hidden_dim
//     uint32   n_experts
//     uint32   n_layers
//     uint32   dtype          (0 = fp32, 1 = fp16)
//     float    beta
//   layer_indices[n_layers] : int32  (physical layer index per slot)
//   keys[n_patterns][hidden_dim]                  in dtype, row-major
//   values[n_layers][n_patterns][n_experts]       in dtype, row-major

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moe_orch {

enum HopfieldDType : uint32_t {
    HFLD_DTYPE_FP32 = 0,
    HFLD_DTYPE_FP16 = 1,
};

struct HopfieldHeader {
    char     magic[4];          // "HFLD"
    uint32_t version;           // 1
    uint32_t n_patterns;
    uint32_t hidden_dim;
    uint32_t n_experts;
    uint32_t n_layers;
    uint32_t dtype;             // HopfieldDType
    float    beta;
};
static_assert(sizeof(HopfieldHeader) == 32, "HopfieldHeader layout drift");

class HopfieldPredictor {
public:
    bool load(const char * path);
    void close();

    // Compute the soft expert distribution at `layer` for the given query.
    //   query     : float pointer of length hidden_dim_
    //   layer     : physical layer index (must match one of the trained layers)
    //   out_dist  : caller-provided float buffer of length n_experts_
    // Returns false if the layer wasn't in the training set or the load failed.
    bool retrieve(int32_t layer,
                  const float * query,
                  float * out_dist) const;

    // Top-N expert indices to keep resident at `layer`.
    //   out_indices : caller-provided int32 buffer of length top_n
    //   out_weights : OPTIONAL float buffer of length top_n; receives the
    //                 dist values at the selected indices (NOT renormalized
    //                 — these are per-expert mass shares from the memory
    //                 retrieval, summing to <= 1.0).
    bool predict_top_n(int32_t layer,
                       const float * query,
                       int32_t top_n,
                       int32_t * out_indices,
                       float * out_weights = nullptr) const;

    uint32_t n_patterns() const { return n_patterns_; }
    uint32_t hidden_dim() const { return hidden_dim_; }
    uint32_t n_experts()  const { return n_experts_;  }
    uint32_t n_layers()   const { return n_layers_;   }
    HopfieldDType dtype() const { return HopfieldDType(dtype_); }
    float    beta()       const { return beta_; }

    // Returns true if `layer` has a stored values matrix.
    bool has_layer(int32_t layer) const;

private:
    int32_t layer_slot(int32_t physical_layer) const;

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
    uint32_t n_patterns_ = 0;
    uint32_t hidden_dim_ = 0;
    uint32_t n_experts_  = 0;
    uint32_t n_layers_   = 0;
    uint32_t dtype_      = HFLD_DTYPE_FP16;
    float    beta_       = 4.0f;

    // Pointers into the mmap'd blob.
    const void *               keys_      = nullptr;       // [n_patterns_, hidden_dim_]
    std::vector<const void *>  values_;                    // size n_layers_; each [n_patterns_, n_experts_]
    std::vector<int32_t>       layer_indices_;             // size n_layers_; physical layer at each slot

    // Sparse layer -> slot lookup. layer_to_slot_[L] = slot or -1.
    std::vector<int32_t>       layer_to_slot_;
};

} // namespace moe_orch
