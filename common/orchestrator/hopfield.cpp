// hopfield.cpp — implementation of L2 Hopfield predictor.
//
// AVX2/F16C path mirrors predictor.cpp (same shape: x . W + b style GEMV).
// Scalar fallback retained for non-AVX2 builds.

#include "hopfield.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__AVX2__) || defined(_MSC_VER)
  #define MOE_ORCH_HAVE_AVX2 1
  #include <immintrin.h>
#else
  #define MOE_ORCH_HAVE_AVX2 0
#endif

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

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign     = (uint32_t(h) & 0x8000u) << 16;
    uint32_t exponent = (uint32_t(h) >> 10) & 0x1Fu;
    uint32_t mantissa = uint32_t(h) & 0x3FFu;
    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) {
            f = sign;
        } else {
            int e = -1;
            while ((mantissa & 0x400u) == 0) { mantissa <<= 1; e--; }
            mantissa &= 0x3FFu;
            f = sign | ((127 - 15 + 1 + e) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        f = sign | 0x7F800000u | (mantissa << 13);
    } else {
        f = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &f, 4);
    return result;
}

bool HopfieldPredictor::load(const char * path) {
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

    if (map_size_ < sizeof(HopfieldHeader)) { close(); return false; }
    const HopfieldHeader * hdr = (const HopfieldHeader *)map_base_;
    if (std::memcmp(hdr->magic, "HFLD", 4) != 0) { close(); return false; }
    if (hdr->version != 1) { close(); return false; }

    n_patterns_ = hdr->n_patterns;
    hidden_dim_ = hdr->hidden_dim;
    n_experts_  = hdr->n_experts;
    n_layers_   = hdr->n_layers;
    dtype_      = hdr->dtype;
    beta_       = hdr->beta;

    const size_t per_elt = (dtype_ == HFLD_DTYPE_FP16) ? 2 : 4;

    const uint8_t * blob = (const uint8_t *)map_base_;
    size_t off = sizeof(HopfieldHeader);

    // layer_indices: int32 * n_layers
    if (off + (size_t)n_layers_ * sizeof(int32_t) > map_size_) { close(); return false; }
    layer_indices_.assign(n_layers_, -1);
    std::memcpy(layer_indices_.data(), blob + off, (size_t)n_layers_ * sizeof(int32_t));
    off += (size_t)n_layers_ * sizeof(int32_t);

    // keys: [n_patterns, hidden_dim]
    const size_t keys_bytes = (size_t)n_patterns_ * hidden_dim_ * per_elt;
    if (off + keys_bytes > map_size_) { close(); return false; }
    keys_ = blob + off;
    off += keys_bytes;

    // values: n_layers x [n_patterns, n_experts]
    const size_t values_per_layer_bytes = (size_t)n_patterns_ * n_experts_ * per_elt;
    values_.resize(n_layers_);
    int32_t max_layer = -1;
    for (uint32_t l = 0; l < n_layers_; l++) {
        if (off + values_per_layer_bytes > map_size_) { close(); return false; }
        values_[l] = blob + off;
        off += values_per_layer_bytes;
        if (layer_indices_[l] > max_layer) max_layer = layer_indices_[l];
    }

    // Sparse lookup
    layer_to_slot_.assign((size_t)(max_layer + 1), -1);
    for (uint32_t l = 0; l < n_layers_; l++) {
        if (layer_indices_[l] >= 0) {
            layer_to_slot_[layer_indices_[l]] = (int32_t)l;
        }
    }

    return true;
}

void HopfieldPredictor::close() {
#ifdef _WIN32
    if (map_base_)   { ::UnmapViewOfFile(map_base_); map_base_ = nullptr; }
    if (map_handle_) { ::CloseHandle((HANDLE)map_handle_); map_handle_ = nullptr; }
    if (file_handle_){ ::CloseHandle((HANDLE)file_handle_); file_handle_ = nullptr; }
#else
    if (map_base_)   { ::munmap(map_base_, map_size_); map_base_ = nullptr; }
    if (fd_ >= 0)    { ::close(fd_); fd_ = -1; }
#endif
    map_size_ = 0;
    keys_ = nullptr;
    values_.clear();
    layer_indices_.clear();
    layer_to_slot_.clear();
}

int32_t HopfieldPredictor::layer_slot(int32_t physical_layer) const {
    if (physical_layer < 0 || (size_t)physical_layer >= layer_to_slot_.size()) return -1;
    return layer_to_slot_[physical_layer];
}

bool HopfieldPredictor::has_layer(int32_t layer) const {
    return layer_slot(layer) >= 0;
}

// ---------- inner kernels ----------
//
// Two GEMVs per call:
//   (A) sim[p] = sum_d keys[p, d] * query[d]                  for p in [0,n_patterns)
//   (B) dist[e] = sum_p weights[p] * values[p, e]              for e in [0,n_experts)
//
// (A) has rows = n_patterns_ (~256), inner dim = hidden_dim_ (~2048).
//     Outer over p (one row at a time), inner over d with FMA.
// (B) has rows = n_experts_ (~128), inner dim = n_patterns_ (~256).
//     Loop-reorder same as predictor.cpp: outer over p, inner over e.
//     Lets us read weights[p] as a broadcast scalar and stream
//     contiguous values rows.

#if MOE_ORCH_HAVE_AVX2
static inline float dot_fp16_avx2(const uint16_t * a_fp16,
                                  const float    * b,
                                  uint32_t n) {
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i ah = _mm_loadu_si128((const __m128i *)(a_fp16 + i));
        __m256  af = _mm256_cvtph_ps(ah);
        __m256  bf = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(af, bf, acc);
    }
    // horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result = _mm_cvtss_f32(s);
    for (; i < n; i++) result += fp16_to_fp32(a_fp16[i]) * b[i];
    return result;
}

static inline float dot_fp32_avx2(const float * a, const float * b, uint32_t n) {
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 af = _mm256_loadu_ps(a + i);
        __m256 bf = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(af, bf, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result = _mm_cvtss_f32(s);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

// dist[e] += scalar * row_fp16[e]  for e in [0,n_experts)
static inline void axpy_fp16_avx2(float scalar,
                                  const uint16_t * row_fp16,
                                  float * dist,
                                  uint32_t n_experts) {
    const __m256 sv = _mm256_set1_ps(scalar);
    uint32_t e = 0;
    for (; e + 8 <= n_experts; e += 8) {
        __m128i rh = _mm_loadu_si128((const __m128i *)(row_fp16 + e));
        __m256  rf = _mm256_cvtph_ps(rh);
        __m256  d  = _mm256_loadu_ps(dist + e);
        d = _mm256_fmadd_ps(sv, rf, d);
        _mm256_storeu_ps(dist + e, d);
    }
    for (; e < n_experts; e++) dist[e] += scalar * fp16_to_fp32(row_fp16[e]);
}

static inline void axpy_fp32_avx2(float scalar,
                                  const float * row,
                                  float * dist,
                                  uint32_t n_experts) {
    const __m256 sv = _mm256_set1_ps(scalar);
    uint32_t e = 0;
    for (; e + 8 <= n_experts; e += 8) {
        __m256 rf = _mm256_loadu_ps(row + e);
        __m256 d  = _mm256_loadu_ps(dist + e);
        d = _mm256_fmadd_ps(sv, rf, d);
        _mm256_storeu_ps(dist + e, d);
    }
    for (; e < n_experts; e++) dist[e] += scalar * row[e];
}
#endif

bool HopfieldPredictor::retrieve(int32_t layer,
                                 const float * query,
                                 float * out_dist) const
{
    if (!keys_ || !out_dist || !query) return false;
    const int32_t slot = layer_slot(layer);
    if (slot < 0) return false;
    const void * vals = values_[(size_t)slot];

    const bool is_fp16 = (dtype_ == HFLD_DTYPE_FP16);
    const float inv_sqrt_d = 1.0f / std::sqrt((float)hidden_dim_);

    // ---- (A) sim[p] = (keys[p] . query) / sqrt(hidden_dim) ----
    std::vector<float> sim((size_t)n_patterns_);
    for (uint32_t p = 0; p < n_patterns_; p++) {
        float s;
        if (is_fp16) {
            const uint16_t * row = (const uint16_t *)keys_ + (size_t)p * hidden_dim_;
#if MOE_ORCH_HAVE_AVX2
            s = dot_fp16_avx2(row, query, hidden_dim_);
#else
            s = 0.0f;
            for (uint32_t d = 0; d < hidden_dim_; d++) s += fp16_to_fp32(row[d]) * query[d];
#endif
        } else {
            const float * row = (const float *)keys_ + (size_t)p * hidden_dim_;
#if MOE_ORCH_HAVE_AVX2
            s = dot_fp32_avx2(row, query, hidden_dim_);
#else
            s = 0.0f;
            for (uint32_t d = 0; d < hidden_dim_; d++) s += row[d] * query[d];
#endif
        }
        sim[p] = s * inv_sqrt_d;
    }

    // ---- softmax(beta * sim) ----
    float maxv = -INFINITY;
    for (uint32_t p = 0; p < n_patterns_; p++) {
        sim[p] *= beta_;
        if (sim[p] > maxv) maxv = sim[p];
    }
    float sum = 0.0f;
    for (uint32_t p = 0; p < n_patterns_; p++) {
        sim[p] = std::exp(sim[p] - maxv);
        sum += sim[p];
    }
    const float inv = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
    for (uint32_t p = 0; p < n_patterns_; p++) sim[p] *= inv;

    // ---- (B) dist[e] = sum_p weights[p] * values[p, e] ----
    std::memset(out_dist, 0, (size_t)n_experts_ * sizeof(float));
    for (uint32_t p = 0; p < n_patterns_; p++) {
        const float w = sim[p];
        if (is_fp16) {
            const uint16_t * row = (const uint16_t *)vals + (size_t)p * n_experts_;
#if MOE_ORCH_HAVE_AVX2
            axpy_fp16_avx2(w, row, out_dist, n_experts_);
#else
            for (uint32_t e = 0; e < n_experts_; e++) out_dist[e] += w * fp16_to_fp32(row[e]);
#endif
        } else {
            const float * row = (const float *)vals + (size_t)p * n_experts_;
#if MOE_ORCH_HAVE_AVX2
            axpy_fp32_avx2(w, row, out_dist, n_experts_);
#else
            for (uint32_t e = 0; e < n_experts_; e++) out_dist[e] += w * row[e];
#endif
        }
    }
    return true;
}

bool HopfieldPredictor::predict_top_n(int32_t layer,
                                      const float * query,
                                      int32_t top_n,
                                      int32_t * out_indices,
                                      float * out_weights) const
{
    if (!out_indices || top_n <= 0) return false;
    std::vector<float> dist((size_t)n_experts_);
    if (!retrieve(layer, query, dist.data())) return false;

    std::vector<std::pair<float, int32_t>> ranked((size_t)n_experts_);
    for (uint32_t i = 0; i < n_experts_; i++) ranked[i] = { dist[i], (int32_t)i };
    const int32_t k = std::min<int32_t>(top_n, (int32_t)n_experts_);
    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(),
                      [](const auto & a, const auto & b) { return a.first > b.first; });
    for (int32_t i = 0; i < k; i++) {
        out_indices[i] = ranked[i].second;
        if (out_weights) out_weights[i] = ranked[i].first;
    }
    return true;
}

} // namespace moe_orch
