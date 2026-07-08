#ifndef XQWEN35_RUNTIME_CUSTOM_MODEL_HPP_
#define XQWEN35_RUNTIME_CUSTOM_MODEL_HPP_

#include "../kernels/kernels.hpp"

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace xq {

namespace kernels {
class VulkanBackend;
}

struct KernelStat {
    uint64_t calls = 0;
    double total_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    std::string backend = "cpu";
};

class KernelTrace {
public:
    void add(const std::string& name, double ms);
    void add(const std::string& name, double ms, const std::string& backend);
    void reset();
    std::string toJson() const;

private:
    std::map<std::string, KernelStat> stats_;
};

class CustomModel {
public:
    bool load(const std::string& model_dir, std::string* error);
    bool load(const std::string& model_dir, const std::string& backend, std::string* error);
    bool prefill(const int32_t* token_ids, size_t n_tokens, KernelTrace* trace, std::string* error);
    bool decodeOne(int32_t* token_id, size_t position, KernelTrace* trace, std::string* error);
    void resetState();

    int hiddenSize() const { return hidden_size_; }
    int vocabSize() const { return vocab_size_; }
    int layers() const { return static_cast<int>(layers_.size()); }
    std::string coverageSummary() const;
    std::string debugJson(int top_k) const;
    bool copyHidden(float* out_values, size_t value_capacity, size_t* out_size) const;

private:
    struct Layer {
        bool full_attention = false;
        std::vector<float> input_norm;
        std::vector<float> post_norm;
        std::vector<float> q_norm;
        std::vector<float> k_norm;
        std::vector<float> linear_norm;
        kernels::QuantizedMatrix q_proj;
        kernels::QuantizedMatrix k_proj;
        kernels::QuantizedMatrix v_proj;
        kernels::QuantizedMatrix o_proj;
        kernels::QuantizedMatrix linear_in_proj_qkv;
        kernels::QuantizedMatrix linear_in_proj_a;
        kernels::QuantizedMatrix linear_in_proj_b;
        kernels::QuantizedMatrix linear_in_proj_z;
        kernels::QuantizedMatrix linear_out_proj;
        kernels::QuantizedMatrix gate_proj;
        kernels::QuantizedMatrix up_proj;
        kernels::QuantizedMatrix down_proj;
        std::vector<float> linear_a_log;
        std::vector<float> linear_dt_bias;
        std::vector<float> linear_conv_weight;
        std::vector<float> k_cache;
        std::vector<float> v_cache;
        size_t kv_len = 0;
        std::vector<float> linear_conv_state;
        std::vector<float> linear_recurrent_state;
    };

    bool loadLayer(const std::string& model_dir, int index, Layer* layer, std::string* error);
    bool loadEmbeddingRow(int32_t token_id, std::vector<float>* out, KernelTrace* trace, std::string* error);
    void runLayer(Layer& layer, size_t position, KernelTrace* trace);
    void runLinear(const std::string& name,
                   const kernels::QuantizedMatrix& matrix,
                   const std::vector<float>& input,
                   std::vector<float>* output,
                   KernelTrace* trace) const;
    void appendKv(Layer& layer, const std::vector<float>& k, const std::vector<float>& v, KernelTrace* trace);
    void runFullAttention(Layer& layer, size_t position, KernelTrace* trace);
    void runLinearAttention(Layer& layer, KernelTrace* trace);
    void runFinalNorm(KernelTrace* trace);
    bool sampleGreedy(int32_t* token_id, KernelTrace* trace, std::string* error);

    std::string model_dir_;
    int hidden_size_ = 4096;
    int intermediate_size_ = 12288;
    int vocab_size_ = 248320;
    int num_attention_heads_ = 16;
    int num_key_value_heads_ = 4;
    int head_dim_ = 256;
    int rotary_dim_ = 64;
    int linear_key_head_dim_ = 128;
    int linear_value_head_dim_ = 128;
    int linear_num_key_heads_ = 16;
    int linear_num_value_heads_ = 32;
    int linear_conv_kernel_dim_ = 4;
    int linear_key_dim_ = 2048;
    int linear_value_dim_ = 4096;
    int linear_conv_dim_ = 8192;
    float rms_eps_ = 1.0e-6f;
    float rope_theta_ = 10000000.0f;
    std::vector<Layer> layers_;
    kernels::QuantizedMatrix lm_head_;
    std::vector<float> final_norm_;
    std::vector<float> hidden_;
    std::vector<float> scratch_hidden_;
    std::vector<float> normed_;
    std::vector<float> q_;
    std::vector<float> q_query_;
    std::vector<float> attn_gate_;
    std::vector<float> k_;
    std::vector<float> v_;
    std::vector<float> attn_hidden_;
    std::vector<float> attn_scores_;
    std::vector<float> max_scores_;
    std::vector<float> linear_mixed_;
    std::vector<float> linear_a_;
    std::vector<float> linear_b_;
    std::vector<float> linear_z_;
    std::vector<float> linear_conv_;
    std::vector<float> gate_;
    std::vector<float> up_;
    std::vector<float> ffn_;
    std::vector<float> logits_;
    std::ifstream embedding_stream_;
    std::vector<uint16_t> embedding_row_bf16_;
    std::string backend_ = "cpu";
    mutable std::unique_ptr<kernels::VulkanBackend> vulkan_backend_;
    mutable uint64_t vulkan_linear_calls_ = 0;
    mutable uint64_t vulkan_linear_failures_ = 0;
    mutable uint64_t vulkan_tensor_calls_ = 0;
    mutable uint64_t vulkan_tensor_failures_ = 0;
};

}  // namespace xq

#endif  // XQWEN35_RUNTIME_CUSTOM_MODEL_HPP_
