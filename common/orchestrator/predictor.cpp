// predictor.cpp — implementation of MLPPredictor for the llama.cpp orchestrator fork.
//
// Reference implementation. Naive scalar GEMMs for clarity; production should
// substitute SIMD or a BLAS call. This is correct, fast enough for inference-loop
// use (sub-ms per call on Turing-era CPUs), and easy to read for review.

#include "predictor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

namespace moe_orch {

// ---------- helpers ----------

static inline float fp16_to_fp32(uint16_t h) {
    // IEEE 754 half -> float, branchless via reinterpret. ggml has a faster
    // version with a lookup table; this is the readable reference.
    uint32_t sign     = (uint32_t(h) & 0x8000u) << 16;
    uint32_t exponent = (uint32_t(h) >> 10) & 0x1Fu;
    uint32_t mantissa = uint32_t(h) & 0x3FFu;
    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) {
            f = sign;
        } else {
            // subnormal — normalize
            int e = -1;
            while ((mantissa & 0x400u) == 0) { mantissa <<= 1; e--; }
            mantissa &= 0x3FFu;
            f = sign | ((127 - 15 + 1 + e) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // inf / nan
        f = sign | 0x7F800000u | (mantissa << 13);
    } else {
        f = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &f, 4);
    return result;
}

static inline float gelu(float x) {
    // gpt-2 approximation; matches torch.nn.GELU(approximate='none') closely
    // enough for our purposes. For exact match use 0.5*x*(1+erf(x/sqrt(2))).
    return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f));
}

// ---------- mmap (cross-platform) ----------

bool MLPPredictor::load(const char * path) {
    close();

#ifdef _WIN32
    file_handle_ = ::CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle_ == INVALID_HANDLE_VALUE) { file_handle_ = nullptr; return false; }
    LARGE_INTEGER size;
    if (!::GetFileSizeEx((HANDLE)file_handle_, &size)) { close(); return false; }
    map_size_ = (size_t)size.QuadPart;
    map_handle_ = ::CreateFileMappingA((HANDLE)file_handle_, nullptr, PAGE_READONLY,
                                       size.HighPart, size.LowPart, nullptr);
    if (!map_handle_) { close(); return false; }
    map_base_ = ::MapViewOfFile((HANDLE)map_handle_, FILE_MAP_READ, 0, 0, 0);
    if (!map_base_) { close(); return false; }
#else
    fd_ = ::open(path, O_RDONLY);
    if (fd_ < 0) return false;
    struct stat st;
    if (::fstat(fd_, &st) != 0) { close(); return false; }
    map_size_ = (size_t)st.st_size;
    map_base_ = ::mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (map_base_ == MAP_FAILED) { map_base_ = nullptr; close(); return false; }
#endif

    if (map_size_ < sizeof(PredictorHeader)) { close(); return false; }
    const PredictorHeader * hdr = (const PredictorHeader *)map_base_;
    if (std::memcmp(hdr->magic, "MPRD", 4) != 0) { close(); return false; }
    if (hdr->version != 1) { close(); return false; }

    hidden_dim_   = hdr->hidden_dim;
    hidden_units_ = hdr->hidden_units;
    n_experts_    = hdr->n_experts;
    num_heads_    = hdr->num_heads;
    dtype_        = hdr->dtype;

    const size_t per_elt   = (dtype_ == PRED_DTYPE_FP16) ? 2 : 4;
    const size_t per_head  = ((size_t)hidden_dim_ * hidden_units_   // w1
                            + hidden_units_                         // b1
                            + (size_t)hidden_units_ * n_experts_    // w2
                            + n_experts_) * per_elt;                // b2

    const PredictorIndexEntry * idx = (const PredictorIndexEntry *)
        ((const uint8_t *)map_base_ + sizeof(PredictorHeader));
    const uint8_t * blob = (const uint8_t *)map_base_;

    heads_.reserve(num_heads_);
    int32_t max_layer = -1;
    int32_t max_horizon = -1;
    for (uint32_t i = 0; i < num_heads_; i++) {
        const auto & e = idx[i];
        PredictorHead h;
        h.source_layer = e.source_layer;
        h.horizon      = e.horizon;
        const uint8_t * p = blob + e.offset_bytes;
        h.w1 = p;                                                       p += (size_t)hidden_dim_   * hidden_units_   * per_elt;
        h.b1 = p;                                                       p += (size_t)hidden_units_                   * per_elt;
        h.w2 = p;                                                       p += (size_t)hidden_units_ * n_experts_      * per_elt;
        h.b2 = p;
        heads_.push_back(h);
        if (e.source_layer > max_layer)   max_layer = e.source_layer;
        if (e.horizon      > max_horizon) max_horizon = e.horizon;
    }

    // Build sparse lookup: head_by_layer_horizon_[layer][horizon] = head index, or -1
    head_by_layer_horizon_.assign((size_t)(max_layer + 1),
                                  std::vector<int32_t>((size_t)(max_horizon + 1), -1));
    for (size_t i = 0; i < heads_.size(); i++) {
        head_by_layer_horizon_[heads_[i].source_layer][heads_[i].horizon] = (int32_t)i;
    }

    // (per_head used in size sanity if you want)
    (void)per_head;

    return true;
}

void MLPPredictor::close() {
#ifdef _WIN32
    if (map_base_)   { ::UnmapViewOfFile(map_base_); map_base_   = nullptr; }
    if (map_handle_) { ::CloseHandle((HANDLE)map_handle_); map_handle_ = nullptr; }
    if (file_handle_){ ::CloseHandle((HANDLE)file_handle_); file_handle_ = nullptr; }
#else
    if (map_base_)   { ::munmap(map_base_, map_size_); map_base_ = nullptr; }
    if (fd_ >= 0)    { ::close(fd_); fd_ = -1; }
#endif
    map_size_ = 0;
    heads_.clear();
    head_by_layer_horizon_.clear();
}

int32_t MLPPredictor::head_index(int32_t source_layer, int32_t horizon) const {
    if (source_layer < 0 || (size_t)source_layer >= head_by_layer_horizon_.size()) return -1;
    const auto & row = head_by_layer_horizon_[source_layer];
    if (horizon < 0 || (size_t)horizon >= row.size()) return -1;
    return row[horizon];
}

// ---------- inference ----------

bool MLPPredictor::predict_top_k(
    int32_t source_layer,
    int32_t horizon,
    const float * hidden_state,
    int32_t top_k,
    int32_t * out_indices) const
{
    int32_t hi = head_index(source_layer, horizon);
    if (hi < 0) return false;
    const PredictorHead & head = heads_[hi];

    const bool is_fp16 = (dtype_ == PRED_DTYPE_FP16);
    auto load_w = [is_fp16](const void * base, size_t i) -> float {
        return is_fp16 ? fp16_to_fp32(((const uint16_t *)base)[i])
                       : ((const float *)base)[i];
    };

    // hidden = GELU(hidden_state @ w1 + b1)
    std::vector<float> hidden(hidden_units_);
    for (uint32_t j = 0; j < hidden_units_; j++) {
        float sum = load_w(head.b1, j);
        for (uint32_t i = 0; i < hidden_dim_; i++) {
            sum += hidden_state[i] * load_w(head.w1, (size_t)i * hidden_units_ + j);
        }
        hidden[j] = gelu(sum);
    }

    // logits = hidden @ w2 + b2
    std::vector<float> logits(n_experts_);
    for (uint32_t j = 0; j < n_experts_; j++) {
        float sum = load_w(head.b2, j);
        for (uint32_t i = 0; i < hidden_units_; i++) {
            sum += hidden[i] * load_w(head.w2, (size_t)i * n_experts_ + j);
        }
        logits[j] = sum;
    }

    // top-K via partial sort (k is small — typically 8 or 16; partial_sort is fine)
    std::vector<std::pair<float, int32_t>> ranked(n_experts_);
    for (uint32_t i = 0; i < n_experts_; i++) ranked[i] = { logits[i], (int32_t)i };
    const int32_t k = std::min<int32_t>(top_k, (int32_t)n_experts_);
    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(),
                      [](const auto & a, const auto & b) { return a.first > b.first; });
    for (int32_t i = 0; i < k; i++) out_indices[i] = ranked[i].second;
    return true;
}

} // namespace moe_orch
