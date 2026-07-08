#ifndef XQWEN35_RUNTIME_VULKAN_BACKEND_HPP_
#define XQWEN35_RUNTIME_VULKAN_BACKEND_HPP_

#include "xqwen35.h"

#include <cstddef>
#include <string>

namespace xq {
namespace kernels {

struct QuantizedMatrix;

struct VulkanGemvTiming {
    double upload_ms = 0.0;
    double dispatch_ms = 0.0;
    double download_ms = 0.0;
    double total_ms = 0.0;
};

class VulkanBackend {
public:
    VulkanBackend();
    ~VulkanBackend();

    bool initialize(std::string* error);
    bool isInitialized() const;
    const std::string& deviceName() const;

    bool gemvW4A16(const QuantizedMatrix& matrix,
                   const float* x,
                   float* y,
                   VulkanGemvTiming* timing,
                   std::string* error);
    bool rmsNorm(const float* x,
                 const float* weight,
                 int n,
                 float eps,
                 float* y,
                 VulkanGemvTiming* timing,
                 std::string* error);
    bool headRmsNorm(float* x,
                     const float* weight,
                     int heads,
                     int head_dim,
                     float eps,
                     VulkanGemvTiming* timing,
                     std::string* error);
    bool l2NormalizeHeads(float* x, int heads, int head_dim, VulkanGemvTiming* timing, std::string* error);
    bool gatedRmsNorm(const float* x,
                      const float* weight,
                      const float* gate,
                      int heads,
                      int head_dim,
                      float eps,
                      float* y,
                      VulkanGemvTiming* timing,
                      std::string* error);
    bool rope(float* q,
              int q_heads,
              float* k,
              int k_heads,
              int head_dim,
              int rotary_dim,
              int position,
              float theta,
              VulkanGemvTiming* timing,
              std::string* error);
    bool siluGateMul(const float* gate,
                     const float* up,
                     int n,
                     float* y,
                     VulkanGemvTiming* timing,
                     std::string* error);
    bool residualAdd(float* hidden, const float* delta, int n, VulkanGemvTiming* timing, std::string* error);
    bool attentionOutputGate(float* hidden, const float* gate, int n, VulkanGemvTiming* timing, std::string* error);
    bool argmax(const float* logits, int n, int* out_index, float* out_value, VulkanGemvTiming* timing, std::string* error);
    bool gqaDecode(const float* q,
                   const float* k_cache,
                   const float* v_cache,
                   int context,
                   int q_heads,
                   int kv_heads,
                   int head_dim,
                   float* out,
                   VulkanGemvTiming* timing,
                   std::string* error);
    bool linearAttentionConv1d(const float* input,
                               const float* state,
                               const float* weight,
                               int conv_dim,
                               int kernel_dim,
                               float* conv_out,
                               float* new_state,
                               VulkanGemvTiming* timing,
                               std::string* error);
    bool linearAttentionStateUpdate(const float* linear_conv,
                                    const float* linear_a,
                                    const float* linear_b,
                                    const float* a_log,
                                    const float* dt_bias,
                                    float* recurrent_state,
                                    int key_heads,
                                    int value_heads,
                                    int key_dim,
                                    int value_dim,
                                    float* out,
                                    VulkanGemvTiming* timing,
                                    std::string* error);
    bool appendFloatVector(const float* old_data,
                           int old_values,
                           const float* append_data,
                           int append_values,
                           float* out,
                           VulkanGemvTiming* timing,
                           std::string* error);

private:
    struct Impl;
    Impl* impl_;
};

xq_status runVulkanW4A16SelfTestJson(char* json_out, size_t json_capacity);

}  // namespace kernels
}  // namespace xq

#endif  // XQWEN35_RUNTIME_VULKAN_BACKEND_HPP_
