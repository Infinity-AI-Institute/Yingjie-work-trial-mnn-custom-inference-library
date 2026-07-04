#ifndef XQWEN35_RUNTIME_CUSTOM_MODEL_HPP_
#define XQWEN35_RUNTIME_CUSTOM_MODEL_HPP_

#include "../kernels/kernels.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace xq {

struct KernelStat {
    uint64_t calls = 0;
    double total_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

class KernelTrace {
public:
    void add(const std::string& name, double ms);
    void reset();
    std::string toJson() const;

private:
    std::map<std::string, KernelStat> stats_;
};

class CustomModel {
public:
    bool load(const std::string& model_dir, std::string* error);
    bool prefill(const int32_t* token_ids, size_t n_tokens, KernelTrace* trace, std::string* error);
    bool decodeOne(int32_t* token_id, size_t position, KernelTrace* trace, std::string* error);

    int hiddenSize() const { return hidden_size_; }
    int vocabSize() const { return vocab_size_; }
    int layers() const { return static_cast<int>(layers_.size()); }
    std::string coverageSummary() const;

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
        kernels::QuantizedMatrix linear_in_proj_z;
        kernels::QuantizedMatrix linear_out_proj;
        kernels::QuantizedMatrix gate_proj;
        kernels::QuantizedMatrix up_proj;
        kernels::QuantizedMatrix down_proj;
    };

    bool loadLayer(const std::string& model_dir, int index, Layer* layer, std::string* error);
    bool loadEmbeddingRow(int32_t token_id, std::vector<float>* out, KernelTrace* trace, std::string* error) const;
    void runLayer(Layer& layer, size_t position, KernelTrace* trace);
    void runLinear(const std::string& name,
                   const kernels::QuantizedMatrix& matrix,
                   const std::vector<float>& input,
                   std::vector<float>* output,
                   KernelTrace* trace) const;

    std::string model_dir_;
    int hidden_size_ = 4096;
    int intermediate_size_ = 12288;
    int vocab_size_ = 248320;
    int num_attention_heads_ = 16;
    int num_key_value_heads_ = 4;
    int head_dim_ = 256;
    float rms_eps_ = 1.0e-6f;
    float rope_theta_ = 10000000.0f;
    std::vector<Layer> layers_;
    std::vector<float> final_norm_;
    std::vector<float> hidden_;
    std::vector<float> scratch_hidden_;
    std::vector<float> normed_;
    std::vector<float> q_;
    std::vector<float> k_;
    std::vector<float> v_;
    std::vector<float> attn_hidden_;
    std::vector<float> gate_;
    std::vector<float> up_;
    std::vector<float> ffn_;
};

}  // namespace xq

#endif  // XQWEN35_RUNTIME_CUSTOM_MODEL_HPP_
