#include "custom_model.hpp"
#include "vulkan_backend.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace xq {
namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string readSmallFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

bool fileExists(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return in.good();
}

int findJsonInt(const std::string& json, const std::string& key, int fallback) {
    const std::string needle = "\"" + key + "\"";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return fallback;
    }
    const size_t colon = json.find(':', pos + needle.size());
    if (colon == std::string::npos) {
        return fallback;
    }
    const size_t first = json.find_first_of("-0123456789", colon + 1);
    if (first == std::string::npos) {
        return fallback;
    }
    size_t last = first + 1;
    while (last < json.size() && json[last] >= '0' && json[last] <= '9') {
        ++last;
    }
    try {
        return std::stoi(json.substr(first, last - first));
    } catch (...) {
        return fallback;
    }
}

float findJsonFloat(const std::string& json, const std::string& key, float fallback) {
    const std::string needle = "\"" + key + "\"";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return fallback;
    }
    const size_t colon = json.find(':', pos + needle.size());
    if (colon == std::string::npos) {
        return fallback;
    }
    const size_t first = json.find_first_of("-0123456789", colon + 1);
    if (first == std::string::npos) {
        return fallback;
    }
    size_t last = first + 1;
    while (last < json.size() && (std::isdigit(static_cast<unsigned char>(json[last])) || json[last] == '.' ||
                                  json[last] == 'e' || json[last] == 'E' || json[last] == '-' || json[last] == '+')) {
        ++last;
    }
    try {
        return std::stof(json.substr(first, last - first));
    } catch (...) {
        return fallback;
    }
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream oss;
    for (char ch : value) {
        switch (ch) {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                oss << ch;
                break;
        }
    }
    return oss.str();
}

std::string traceOpFamily(const std::string& name) {
    if (name.find("linear_q_proj") != std::string::npos) {
        return "q_proj";
    }
    if (name.find("linear_k_proj") != std::string::npos) {
        return "k_proj";
    }
    if (name.find("linear_v_proj") != std::string::npos) {
        return "v_proj";
    }
    if (name.find("linear_o_proj") != std::string::npos) {
        return "o_proj";
    }
    if (name.find("linear_gate_proj") != std::string::npos) {
        return "gate_proj";
    }
    if (name.find("linear_up_proj") != std::string::npos) {
        return "up_proj";
    }
    if (name.find("linear_down_proj") != std::string::npos) {
        return "down_proj";
    }
    if (name.find("linear_attn") != std::string::npos) {
        return "linear_attention_projection";
    }
    if (name.find("linear_attention_state") != std::string::npos ||
        name.find("linear_attention_conv") != std::string::npos ||
        name.find("linear_attention_qk") != std::string::npos ||
        name.find("linear_attention_gated") != std::string::npos) {
        return "linear_attention_state";
    }
    if (name.find("rmsnorm") != std::string::npos) {
        return "rmsnorm";
    }
    if (name.find("rope") != std::string::npos) {
        return "rope";
    }
    if (name.find("attention_score") != std::string::npos) {
        return "attention_score";
    }
    if (name.find("attention_softmax") != std::string::npos) {
        return "attention_softmax";
    }
    if (name.find("attention_v_reduce") != std::string::npos) {
        return "attention_v_reduce";
    }
    if (name.find("attention") != std::string::npos) {
        return "attention";
    }
    if (name.find("lm_head") != std::string::npos) {
        return "lm_head";
    }
    if (name.find("sampling") != std::string::npos) {
        return "sampling";
    }
    if (name.find("prefill_kv_build") != std::string::npos || name.find("kv_append") != std::string::npos ||
        name.find("kv_cache") != std::string::npos || name.find("embedding_load") != std::string::npos) {
        return "prefill_kv_build";
    }
    if (name.find("silu") != std::string::npos) {
        return "activation";
    }
    return "other";
}

std::string layerPrefix(int index) {
    return "layers_" + std::to_string(index) + "_";
}

bool readVectorF32(const std::string& path, int expected, std::vector<float>* out, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        *error = "missing vector file: " + path;
        return false;
    }
    out->assign(static_cast<size_t>(expected), 0.0f);
    in.read(reinterpret_cast<char*>(out->data()), static_cast<std::streamsize>(out->size() * sizeof(float)));
    if (in.gcount() != static_cast<std::streamsize>(out->size() * sizeof(float))) {
        *error = "short vector file: " + path;
        return false;
    }
    return true;
}

void addOneToVector(std::vector<float>* values) {
    for (float& value : *values) {
        value += 1.0f;
    }
}

uint32_t readU32(std::ifstream& in) {
    uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

uint64_t readU64(std::ifstream& in) {
    uint64_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

bool readXq4(const std::string& path, kernels::QuantizedMatrix* matrix, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        *error = "missing xq4 matrix: " + path;
        return false;
    }
    char magic[8] = {};
    in.read(magic, sizeof(magic));
    if (std::memcmp(magic, "XQW4A16", 7) != 0) {
        *error = "bad xq4 magic: " + path;
        return false;
    }
    const uint32_t version = readU32(in);
    const uint32_t rows = readU32(in);
    const uint32_t cols = readU32(in);
    const uint32_t group_size = readU32(in);
    const uint32_t bits = readU32(in);
    const uint64_t packed_bytes = readU64(in);
    const uint64_t scales_count = readU64(in);
    const uint64_t zeros_count = readU64(in);
    if (!in || (version != 1 && version != 2) || bits != 4 || rows == 0 || cols == 0 || group_size == 0 ||
        scales_count != zeros_count) {
        *error = "invalid xq4 header: " + path;
        return false;
    }
    matrix->rows = static_cast<int>(rows);
    matrix->cols = static_cast<int>(cols);
    matrix->bits = 4;
    matrix->group_size = static_cast<int>(group_size);
    matrix->affine_asymmetric = (version == 2);
    matrix->packed.assign(static_cast<size_t>(packed_bytes), 0);
    matrix->scales.assign(static_cast<size_t>(scales_count), 1.0f);
    matrix->zeros.assign(static_cast<size_t>(zeros_count), 0.0f);
    matrix->bias.assign(static_cast<size_t>(rows), 0.0f);
    in.read(reinterpret_cast<char*>(matrix->packed.data()), static_cast<std::streamsize>(matrix->packed.size()));
    in.read(reinterpret_cast<char*>(matrix->scales.data()),
            static_cast<std::streamsize>(matrix->scales.size() * sizeof(float)));
    in.read(reinterpret_cast<char*>(matrix->zeros.data()),
            static_cast<std::streamsize>(matrix->zeros.size() * sizeof(float)));
    if (!in) {
        *error = "short xq4 payload: " + path;
        return false;
    }
    return true;
}

float bf16ToFloat(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

float sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

float softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return std::exp(x);
    }
    return std::log1p(std::exp(x));
}

void addResidual(std::vector<float>* hidden, const std::vector<float>& delta) {
    const size_t n = std::min(hidden->size(), delta.size());
    for (size_t i = 0; i < n; ++i) {
        (*hidden)[i] += delta[i];
    }
}

void applyHeadRms(float* x, const float* weight, int heads, int head_dim, float eps) {
    for (int h = 0; h < heads; ++h) {
        float* base = x + h * head_dim;
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            sum += base[d] * base[d];
        }
        const float inv = 1.0f / std::sqrt(sum / static_cast<float>(head_dim) + eps);
        for (int d = 0; d < head_dim; ++d) {
            base[d] *= inv * weight[d];
        }
    }
}

void l2NormalizeHeads(float* x, int heads, int head_dim) {
    for (int h = 0; h < heads; ++h) {
        float* base = x + h * head_dim;
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            sum += base[d] * base[d];
        }
        const float inv = 1.0f / std::sqrt(sum + 1.0e-6f);
        for (int d = 0; d < head_dim; ++d) {
            base[d] *= inv;
        }
    }
}

void applyGatedHeadRms(const float* input,
                       const float* weight,
                       const float* gate,
                       int heads,
                       int head_dim,
                       float eps,
                       float* output) {
    for (int h = 0; h < heads; ++h) {
        const float* in = input + h * head_dim;
        const float* g = gate + h * head_dim;
        float* out = output + h * head_dim;
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            sum += in[d] * in[d];
        }
        const float inv = 1.0f / std::sqrt(sum / static_cast<float>(head_dim) + eps);
        for (int d = 0; d < head_dim; ++d) {
            out[d] = in[d] * inv * weight[d] * silu(g[d]);
        }
    }
}

void applyRopeVector(float* x, int heads, int head_dim, int rotary_dim, int position, float theta) {
    const int active_dim = std::max(0, std::min(head_dim, rotary_dim));
    const int rope_half_dim = active_dim / 2;
    for (int h = 0; h < heads; ++h) {
        float* base = x + h * head_dim;
        // Qwen3.5 exports phi-style partial rotary: rotate the first rotary_dim values,
        // split within that active slice, and leave the remaining head_dim values unchanged.
        for (int i = 0; i < rope_half_dim; ++i) {
            const float inv_freq =
                std::pow(theta, -static_cast<float>(2 * i) / static_cast<float>(active_dim));
            const float angle = static_cast<float>(position) * inv_freq;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            const float x0 = base[i];
            const float x1 = base[rope_half_dim + i];
            base[i] = x0 * c - x1 * s;
            base[rope_half_dim + i] = x1 * c + x0 * s;
        }
    }
}

}  // namespace

void KernelTrace::add(const std::string& name, double ms) {
    add(name, ms, "cpu");
}

void KernelTrace::add(const std::string& name, double ms, const std::string& backend) {
    KernelStat& stat = stats_[name];
    if (stat.calls == 0) {
        stat.min_ms = ms;
        stat.max_ms = ms;
        stat.backend = backend;
    } else {
        stat.min_ms = std::min(stat.min_ms, ms);
        stat.max_ms = std::max(stat.max_ms, ms);
        if (stat.backend != backend) {
            stat.backend = "mixed";
        }
    }
    stat.calls += 1;
    stat.total_ms += ms;
}

void KernelTrace::reset() {
    stats_.clear();
}

std::string KernelTrace::toJson() const {
    std::ostringstream oss;
    oss << "{\"rows\":[";
    bool first = true;
    for (const auto& item : stats_) {
        if (!first) {
            oss << ",";
        }
        first = false;
        const KernelStat& s = item.second;
        oss << "{\"name\":\"" << jsonEscape(item.first)
            << "\",\"op_family\":\"" << traceOpFamily(item.first)
            << "\",\"backend\":\"" << jsonEscape(s.backend) << "\",\"calls\":" << s.calls << ",\"total_ms\":"
            << s.total_ms << ",\"mean_ms\":" << (s.calls ? s.total_ms / static_cast<double>(s.calls) : 0.0)
            << ",\"min_ms\":" << s.min_ms << ",\"max_ms\":" << s.max_ms << "}";
    }
    oss << "]}";
    return oss.str();
}

bool CustomModel::load(const std::string& model_dir, std::string* error) {
    return load(model_dir, "cpu", error);
}

bool CustomModel::load(const std::string& model_dir, const std::string& backend, std::string* error) {
    backend_ = backend.empty() ? "cpu" : backend;
    vulkan_linear_calls_ = 0;
    vulkan_linear_failures_ = 0;
    vulkan_tensor_calls_ = 0;
    vulkan_tensor_failures_ = 0;
    vulkan_backend_.reset();
    model_dir_ = model_dir;
    const std::string manifest = readSmallFile(model_dir + "/xqwen35_manifest.json");
    hidden_size_ = findJsonInt(manifest, "hidden_size", hidden_size_);
    intermediate_size_ = findJsonInt(manifest, "intermediate_size", intermediate_size_);
    vocab_size_ = findJsonInt(manifest, "vocab_size", vocab_size_);
    num_attention_heads_ = findJsonInt(manifest, "num_attention_heads", num_attention_heads_);
    num_key_value_heads_ = findJsonInt(manifest, "num_key_value_heads", num_key_value_heads_);
    head_dim_ = findJsonInt(manifest, "head_dim", head_dim_);
    const float partial_rotary = findJsonFloat(manifest, "partial_rotary_factor", 0.25f);
    rotary_dim_ = std::max(2, static_cast<int>(std::round(static_cast<float>(head_dim_) * partial_rotary)));
    if ((rotary_dim_ & 1) != 0) {
        --rotary_dim_;
    }
    linear_key_head_dim_ = findJsonInt(manifest, "linear_key_head_dim", linear_key_head_dim_);
    linear_value_head_dim_ = findJsonInt(manifest, "linear_value_head_dim", linear_value_head_dim_);
    linear_num_key_heads_ = findJsonInt(manifest, "linear_num_key_heads", linear_num_key_heads_);
    linear_num_value_heads_ = findJsonInt(manifest, "linear_num_value_heads", linear_num_value_heads_);
    linear_conv_kernel_dim_ = findJsonInt(manifest, "linear_conv_kernel_dim", linear_conv_kernel_dim_);
    linear_key_dim_ = linear_key_head_dim_ * linear_num_key_heads_;
    linear_value_dim_ = linear_value_head_dim_ * linear_num_value_heads_;
    linear_conv_dim_ = linear_key_dim_ * 2 + linear_value_dim_;
    rms_eps_ = findJsonFloat(manifest, "rms_norm_eps", rms_eps_);
    rope_theta_ = findJsonFloat(manifest, "rope_theta", rope_theta_);
    const int n_layers = findJsonInt(manifest, "num_hidden_layers", 32);
    if (!fileExists(model_dir + "/embeddings_bf16.bin")) {
        *error = "missing embeddings_bf16.bin";
        return false;
    }
    embedding_stream_.close();
    embedding_stream_.clear();
    embedding_stream_.open(model_dir + "/embeddings_bf16.bin", std::ios::binary);
    if (!embedding_stream_) {
        *error = "failed to open embeddings_bf16.bin";
        return false;
    }
    if (!readVectorF32(model_dir + "/norm_weight.f32", hidden_size_, &final_norm_, error)) {
        return false;
    }
    addOneToVector(&final_norm_);
    if (!readXq4(model_dir + "/lm_head_weight.xq4", &lm_head_, error)) {
        return false;
    }
    layers_.resize(static_cast<size_t>(n_layers));
    for (int i = 0; i < n_layers; ++i) {
        if (!loadLayer(model_dir, i, &layers_[static_cast<size_t>(i)], error)) {
            return false;
        }
    }
    hidden_.assign(static_cast<size_t>(hidden_size_), 0.0f);
    scratch_hidden_.assign(static_cast<size_t>(hidden_size_), 0.0f);
    normed_.assign(static_cast<size_t>(hidden_size_), 0.0f);
    q_.assign(static_cast<size_t>(std::max(hidden_size_ * 2, linear_conv_dim_)), 0.0f);
    q_query_.assign(static_cast<size_t>(hidden_size_), 0.0f);
    attn_gate_.assign(static_cast<size_t>(hidden_size_), 0.0f);
    k_.assign(static_cast<size_t>(std::max(num_key_value_heads_ * head_dim_, 1)), 0.0f);
    v_.assign(static_cast<size_t>(std::max(num_key_value_heads_ * head_dim_, 1)), 0.0f);
    attn_hidden_.assign(static_cast<size_t>(hidden_size_), 0.0f);
    attn_scores_.clear();
    max_scores_.assign(static_cast<size_t>(num_attention_heads_), -1.0e30f);
    linear_mixed_.assign(static_cast<size_t>(linear_conv_dim_), 0.0f);
    linear_a_.assign(static_cast<size_t>(linear_num_value_heads_), 0.0f);
    linear_b_.assign(static_cast<size_t>(linear_num_value_heads_), 0.0f);
    linear_z_.assign(static_cast<size_t>(linear_value_dim_), 0.0f);
    linear_conv_.assign(static_cast<size_t>(linear_conv_dim_), 0.0f);
    gate_.assign(static_cast<size_t>(intermediate_size_), 0.0f);
    up_.assign(static_cast<size_t>(intermediate_size_), 0.0f);
    ffn_.assign(static_cast<size_t>(intermediate_size_), 0.0f);
    logits_.clear();
    embedding_row_bf16_.assign(static_cast<size_t>(hidden_size_), 0);
    resetState();
    if (backend_ == "vulkan" || backend_ == "cpu_vulkan_hybrid") {
        vulkan_backend_ = std::make_unique<kernels::VulkanBackend>();
        std::string vk_error;
        if (!vulkan_backend_->initialize(&vk_error)) {
            vulkan_backend_.reset();
            if (backend_ == "vulkan") {
                *error = "requested Vulkan backend but initialization failed: " + vk_error;
                return false;
            }
            backend_ = "cpu";
        }
    }
    return true;
}

bool CustomModel::loadLayer(const std::string& model_dir, int index, Layer* layer, std::string* error) {
    const std::string p = model_dir + "/" + layerPrefix(index);
    if (!readVectorF32(p + "input_layernorm_weight.f32", hidden_size_, &layer->input_norm, error) ||
        !readVectorF32(p + "post_attention_layernorm_weight.f32", hidden_size_, &layer->post_norm, error) ||
        !readXq4(p + "mlp_gate_proj_weight.xq4", &layer->gate_proj, error) ||
        !readXq4(p + "mlp_up_proj_weight.xq4", &layer->up_proj, error) ||
        !readXq4(p + "mlp_down_proj_weight.xq4", &layer->down_proj, error)) {
        return false;
    }
    addOneToVector(&layer->input_norm);
    addOneToVector(&layer->post_norm);
    if (fileExists(p + "self_attn_q_proj_weight.xq4")) {
        layer->full_attention = true;
        if (!readXq4(p + "self_attn_q_proj_weight.xq4", &layer->q_proj, error) ||
            !readXq4(p + "self_attn_k_proj_weight.xq4", &layer->k_proj, error) ||
            !readXq4(p + "self_attn_v_proj_weight.xq4", &layer->v_proj, error) ||
            !readXq4(p + "self_attn_o_proj_weight.xq4", &layer->o_proj, error) ||
            !readVectorF32(p + "self_attn_q_norm_weight.f32", head_dim_, &layer->q_norm, error) ||
            !readVectorF32(p + "self_attn_k_norm_weight.f32", head_dim_, &layer->k_norm, error)) {
            return false;
        }
        addOneToVector(&layer->q_norm);
        addOneToVector(&layer->k_norm);
        return true;
    }
    layer->full_attention = false;
    return readXq4(p + "linear_attn_in_proj_qkv_weight.xq4", &layer->linear_in_proj_qkv, error) &&
           readXq4(p + "linear_attn_in_proj_a_weight.xq4", &layer->linear_in_proj_a, error) &&
           readXq4(p + "linear_attn_in_proj_b_weight.xq4", &layer->linear_in_proj_b, error) &&
           readXq4(p + "linear_attn_in_proj_z_weight.xq4", &layer->linear_in_proj_z, error) &&
           readXq4(p + "linear_attn_out_proj_weight.xq4", &layer->linear_out_proj, error) &&
           readVectorF32(p + "linear_attn_norm_weight.f32", linear_value_head_dim_, &layer->linear_norm, error) &&
           readVectorF32(p + "linear_attn_conv1d_weight.f32",
                         linear_conv_dim_ * linear_conv_kernel_dim_,
                         &layer->linear_conv_weight,
                         error) &&
           readVectorF32(p + "linear_attn_A_log.f32", linear_num_value_heads_, &layer->linear_a_log, error) &&
           readVectorF32(p + "linear_attn_dt_bias.f32", linear_num_value_heads_, &layer->linear_dt_bias, error);
}

void CustomModel::resetState() {
    for (Layer& layer : layers_) {
        layer.k_cache.clear();
        layer.v_cache.clear();
        layer.kv_len = 0;
        if (!layer.full_attention) {
            layer.linear_conv_state.assign(
                static_cast<size_t>(linear_conv_dim_ * std::max(0, linear_conv_kernel_dim_ - 1)), 0.0f);
            layer.linear_recurrent_state.assign(
                static_cast<size_t>(linear_num_value_heads_ * linear_key_head_dim_ * linear_value_head_dim_), 0.0f);
        }
    }
}

bool CustomModel::loadEmbeddingRow(int32_t token_id,
                                   std::vector<float>* out,
                                   KernelTrace* trace,
                                   std::string* error) {
    if (token_id < 0 || token_id >= vocab_size_) {
        token_id = std::abs(token_id) % std::max(1, vocab_size_);
    }
    const auto t0 = Clock::now();
    if (!embedding_stream_.is_open()) {
        embedding_stream_.clear();
        embedding_stream_.open(model_dir_ + "/embeddings_bf16.bin", std::ios::binary);
    }
    if (!embedding_stream_) {
        *error = "failed to open embeddings_bf16.bin";
        return false;
    }
    const std::streamoff offset =
        static_cast<std::streamoff>(token_id) * static_cast<std::streamoff>(hidden_size_) * 2;
    embedding_stream_.clear();
    embedding_stream_.seekg(offset, std::ios::beg);
    if (embedding_row_bf16_.size() != static_cast<size_t>(hidden_size_)) {
        embedding_row_bf16_.resize(static_cast<size_t>(hidden_size_));
    }
    embedding_stream_.read(reinterpret_cast<char*>(embedding_row_bf16_.data()),
                           static_cast<std::streamsize>(embedding_row_bf16_.size() * sizeof(uint16_t)));
    if (!embedding_stream_) {
        *error = "short embeddings_bf16.bin row read";
        return false;
    }
    if (out->size() != static_cast<size_t>(hidden_size_)) {
        out->resize(static_cast<size_t>(hidden_size_));
    }
    for (int i = 0; i < hidden_size_; ++i) {
        (*out)[static_cast<size_t>(i)] = bf16ToFloat(embedding_row_bf16_[static_cast<size_t>(i)]);
    }
    if (trace) {
        trace->add("embedding_bf16_row_read", elapsedMs(t0, Clock::now()));
    }
    return true;
}

bool CustomModel::prefill(const int32_t* token_ids, size_t n_tokens, KernelTrace* trace, std::string* error) {
    if (!token_ids || n_tokens == 0) {
        *error = "custom prefill requires prompt tokens";
        return false;
    }
    resetState();
    const size_t reserve_tokens = n_tokens + 512;
    const size_t kv_values = static_cast<size_t>(num_key_value_heads_ * head_dim_);
    for (Layer& layer : layers_) {
        if (layer.full_attention) {
            layer.k_cache.reserve(reserve_tokens * kv_values);
            layer.v_cache.reserve(reserve_tokens * kv_values);
        }
    }
    for (size_t i = 0; i < n_tokens; ++i) {
        const auto t0 = Clock::now();
        if (!loadEmbeddingRow(token_ids[i], &hidden_, trace, error)) {
            return false;
        }
        for (Layer& layer : layers_) {
            runLayer(layer, i, trace);
        }
        runFinalNorm(trace);
        if (trace) {
            trace->add("prefill_token_custom",
                       elapsedMs(t0, Clock::now()),
                       (vulkan_backend_ && vulkan_backend_->isInitialized()) ? "vulkan" : "cpu");
        }
    }
    return true;
}

void CustomModel::runLinear(const std::string& name,
                            const kernels::QuantizedMatrix& matrix,
                            const std::vector<float>& input,
                            std::vector<float>* output,
                            KernelTrace* trace) const {
    if (output->size() != static_cast<size_t>(matrix.rows)) {
        output->resize(static_cast<size_t>(matrix.rows));
    }
    const auto t0 = Clock::now();
    if (vulkan_backend_ && vulkan_backend_->isInitialized() && matrix.bits == 4 &&
        input.size() >= static_cast<size_t>(matrix.cols)) {
        kernels::VulkanGemvTiming vk_timing;
        std::string vk_error;
        if (vulkan_backend_->gemvW4A16(matrix, input.data(), output->data(), &vk_timing, &vk_error)) {
            ++vulkan_linear_calls_;
            if (trace) {
                trace->add(name, elapsedMs(t0, Clock::now()), "vulkan");
            }
            return;
        }
        ++vulkan_linear_failures_;
        if (trace) {
            trace->add(name + "_vulkan_failed", elapsedMs(t0, Clock::now()), "vulkan");
        }
    }
    kernels::gemvW4A16Neon(matrix, input.data(), output->data());
    if (trace) {
        trace->add(name, elapsedMs(t0, Clock::now()));
    }
}

void CustomModel::appendKv(Layer& layer,
                           const std::vector<float>& k,
                           const std::vector<float>& v,
                           KernelTrace* trace) {
    const auto t0 = Clock::now();
    const size_t kv_values = static_cast<size_t>(num_key_value_heads_ * head_dim_);
    const size_t old_size = layer.k_cache.size();
    const size_t copy_k = std::min(k.size(), kv_values);
    const size_t copy_v = std::min(v.size(), kv_values);
    bool used_vk = false;
    if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
        std::vector<float> k_append(kv_values, 0.0f);
        std::vector<float> v_append(kv_values, 0.0f);
        std::copy_n(k.data(), copy_k, k_append.data());
        std::copy_n(v.data(), copy_v, v_append.data());
        std::vector<float> new_k(old_size + kv_values, 0.0f);
        std::vector<float> new_v(old_size + kv_values, 0.0f);
        kernels::VulkanGemvTiming vk_timing;
        std::string vk_error;
        const bool k_ok = vulkan_backend_->appendFloatVector(layer.k_cache.empty() ? nullptr : layer.k_cache.data(),
                                                            static_cast<int>(old_size),
                                                            k_append.data(),
                                                            static_cast<int>(kv_values),
                                                            new_k.data(),
                                                            &vk_timing,
                                                            &vk_error);
        vk_error.clear();
        const bool v_ok = vulkan_backend_->appendFloatVector(layer.v_cache.empty() ? nullptr : layer.v_cache.data(),
                                                            static_cast<int>(old_size),
                                                            v_append.data(),
                                                            static_cast<int>(kv_values),
                                                            new_v.data(),
                                                            &vk_timing,
                                                            &vk_error);
        if (k_ok && v_ok) {
            layer.k_cache.swap(new_k);
            layer.v_cache.swap(new_v);
            used_vk = true;
            ++vulkan_tensor_calls_;
        } else {
            ++vulkan_tensor_failures_;
        }
    }
    if (!used_vk) {
        layer.k_cache.resize(old_size + kv_values);
        layer.v_cache.resize(old_size + kv_values);
        std::copy_n(k.data(), copy_k, layer.k_cache.data() + old_size);
        if (copy_k < kv_values) {
            std::fill(layer.k_cache.data() + old_size + copy_k, layer.k_cache.data() + old_size + kv_values, 0.0f);
        }
        std::copy_n(v.data(), copy_v, layer.v_cache.data() + old_size);
        if (copy_v < kv_values) {
            std::fill(layer.v_cache.data() + old_size + copy_v, layer.v_cache.data() + old_size + kv_values, 0.0f);
        }
    }
    layer.kv_len += 1;
    if (trace) {
        trace->add("kv_append_custom", elapsedMs(t0, Clock::now()), used_vk ? "vulkan" : "cpu");
        trace->add("prefill_kv_build_custom", elapsedMs(t0, Clock::now()), used_vk ? "vulkan" : "cpu");
    }
}

void CustomModel::runFullAttention(Layer& layer, size_t position, KernelTrace* trace) {
    runLinear("linear_q_proj_w4a16", layer.q_proj, normed_, &q_, trace);
    runLinear("linear_k_proj_w4a16", layer.k_proj, normed_, &k_, trace);
    runLinear("linear_v_proj_w4a16", layer.v_proj, normed_, &v_, trace);

    const int q_dim = num_attention_heads_ * head_dim_;
    const int gated_head_dim = head_dim_ * 2;
    if (static_cast<int>(q_.size()) >= num_attention_heads_ * gated_head_dim) {
        for (int h = 0; h < num_attention_heads_; ++h) {
            const float* src = q_.data() + static_cast<size_t>(h * gated_head_dim);
            float* q_dst = q_query_.data() + static_cast<size_t>(h * head_dim_);
            float* gate_dst = attn_gate_.data() + static_cast<size_t>(h * head_dim_);
            std::copy(src, src + head_dim_, q_dst);
            std::copy(src + head_dim_, src + gated_head_dim, gate_dst);
        }
    } else {
        std::copy(q_.begin(), q_.begin() + std::min<int>(q_dim, static_cast<int>(q_.size())), q_query_.begin());
        std::fill(attn_gate_.begin(), attn_gate_.end(), 0.0f);
    }

    {
        const auto t0 = Clock::now();
        bool used_vk = false;
        if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
            kernels::VulkanGemvTiming vk_timing;
            std::string vk_error;
            if (vulkan_backend_->headRmsNorm(
                    q_query_.data(), layer.q_norm.data(), num_attention_heads_, head_dim_, rms_eps_, &vk_timing, &vk_error)) {
                used_vk = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
            }
        }
        if (!used_vk) {
            applyHeadRms(q_query_.data(), layer.q_norm.data(), num_attention_heads_, head_dim_, rms_eps_);
        }
        if (trace) {
            trace->add("q_norm_custom", elapsedMs(t0, Clock::now()), used_vk ? "vulkan" : "cpu");
        }
    }
    {
        const auto t0 = Clock::now();
        bool used_vk = false;
        if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
            kernels::VulkanGemvTiming vk_timing;
            std::string vk_error;
            if (vulkan_backend_->headRmsNorm(
                    k_.data(), layer.k_norm.data(), num_key_value_heads_, head_dim_, rms_eps_, &vk_timing, &vk_error)) {
                used_vk = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
            }
        }
        if (!used_vk) {
            applyHeadRms(k_.data(), layer.k_norm.data(), num_key_value_heads_, head_dim_, rms_eps_);
        }
        if (trace) {
            trace->add("k_norm_custom", elapsedMs(t0, Clock::now()), used_vk ? "vulkan" : "cpu");
        }
    }
    {
        const auto t0 = Clock::now();
        bool used_vk = false;
        if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
            kernels::VulkanGemvTiming vk_timing;
            std::string vk_error;
            if (vulkan_backend_->rope(q_query_.data(),
                                      num_attention_heads_,
                                      k_.data(),
                                      num_key_value_heads_,
                                      head_dim_,
                                      rotary_dim_,
                                      static_cast<int>(position),
                                      rope_theta_,
                                      &vk_timing,
                                      &vk_error)) {
                used_vk = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
            }
        }
        if (!used_vk) {
            applyRopeVector(
                q_query_.data(), num_attention_heads_, head_dim_, rotary_dim_, static_cast<int>(position), rope_theta_);
            applyRopeVector(
                k_.data(), num_key_value_heads_, head_dim_, rotary_dim_, static_cast<int>(position), rope_theta_);
        }
        if (trace) {
            trace->add("rope_qk_custom", elapsedMs(t0, Clock::now()), used_vk ? "vulkan" : "cpu");
        }
    }

    appendKv(layer, k_, v_, trace);
    const int context = static_cast<int>(layer.kv_len);
    const size_t score_count = static_cast<size_t>(std::max(1, context * num_attention_heads_));
    if (attn_scores_.size() < score_count) {
        attn_scores_.resize(score_count);
    }
    if (max_scores_.size() != static_cast<size_t>(num_attention_heads_)) {
        max_scores_.resize(static_cast<size_t>(num_attention_heads_));
    }
    std::fill(max_scores_.begin(), max_scores_.end(), -1.0e30f);
    const int group = std::max(1, num_attention_heads_ / std::max(1, num_key_value_heads_));
    bool used_vk_gqa = false;
    {
        const auto t0 = Clock::now();
        if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
            kernels::VulkanGemvTiming vk_timing;
            std::string vk_error;
            if (vulkan_backend_->gqaDecode(q_query_.data(),
                                           layer.k_cache.data(),
                                           layer.v_cache.data(),
                                           context,
                                           num_attention_heads_,
                                           num_key_value_heads_,
                                           head_dim_,
                                           attn_hidden_.data(),
                                           &vk_timing,
                                           &vk_error)) {
                used_vk_gqa = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
            }
        }
        if (used_vk_gqa && trace) {
            const double ms = elapsedMs(t0, Clock::now());
            trace->add("attention_gqa_decode_custom", ms, "vulkan");
            trace->add("attention_score_custom", ms / 3.0, "vulkan");
            trace->add("attention_softmax_custom", ms / 3.0, "vulkan");
            trace->add("attention_v_reduce_custom", ms / 3.0, "vulkan");
        }
    }
    if (!used_vk_gqa) {
        const auto t0 = Clock::now();
        for (int qh = 0; qh < num_attention_heads_; ++qh) {
            const int kh = std::min(num_key_value_heads_ - 1, qh / group);
            const float* qv = q_query_.data() + qh * head_dim_;
            for (int t = 0; t < context; ++t) {
                const float* kv =
                    layer.k_cache.data() + (static_cast<size_t>(t) * num_key_value_heads_ + kh) * head_dim_;
                float dot = 0.0f;
                for (int d = 0; d < head_dim_; ++d) {
                    dot += qv[d] * kv[d];
                }
                const float score = dot / std::sqrt(static_cast<float>(head_dim_));
                attn_scores_[static_cast<size_t>(qh * context + t)] = score;
                max_scores_[static_cast<size_t>(qh)] = std::max(max_scores_[static_cast<size_t>(qh)], score);
            }
        }
        if (trace) {
            trace->add("attention_score_custom", elapsedMs(t0, Clock::now()));
        }
    }
    if (!used_vk_gqa) {
        const auto t0 = Clock::now();
        for (int qh = 0; qh < num_attention_heads_; ++qh) {
            float denom = 0.0f;
            for (int t = 0; t < context; ++t) {
                float& score = attn_scores_[static_cast<size_t>(qh * context + t)];
                score = std::exp(score - max_scores_[static_cast<size_t>(qh)]);
                denom += score;
            }
            const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
            for (int t = 0; t < context; ++t) {
                attn_scores_[static_cast<size_t>(qh * context + t)] *= inv;
            }
        }
        if (trace) {
            trace->add("attention_softmax_custom", elapsedMs(t0, Clock::now()));
        }
    }
    if (!used_vk_gqa) {
        const auto t0 = Clock::now();
        std::fill(attn_hidden_.begin(), attn_hidden_.end(), 0.0f);
        for (int qh = 0; qh < num_attention_heads_; ++qh) {
            const int kh = std::min(num_key_value_heads_ - 1, qh / group);
            float* out = attn_hidden_.data() + qh * head_dim_;
            for (int t = 0; t < context; ++t) {
                const float weight = attn_scores_[static_cast<size_t>(qh * context + t)];
                const float* vv =
                    layer.v_cache.data() + (static_cast<size_t>(t) * num_key_value_heads_ + kh) * head_dim_;
                for (int d = 0; d < head_dim_; ++d) {
                    out[d] += weight * vv[d];
                }
            }
        }
        if (trace) {
            trace->add("attention_v_reduce_custom", elapsedMs(t0, Clock::now()));
        }
    }
    {
        const auto t0 = Clock::now();
        bool used_vk = false;
        if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
            kernels::VulkanGemvTiming vk_timing;
            std::string vk_error;
            if (vulkan_backend_->attentionOutputGate(
                    attn_hidden_.data(), attn_gate_.data(), q_dim, &vk_timing, &vk_error)) {
                used_vk = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
            }
        }
        if (!used_vk) {
            for (int i = 0; i < q_dim; ++i) {
                attn_hidden_[static_cast<size_t>(i)] *= sigmoid(attn_gate_[static_cast<size_t>(i)]);
            }
        }
        if (trace) {
            trace->add("attention_output_gate_custom", elapsedMs(t0, Clock::now()), used_vk ? "vulkan" : "cpu");
        }
    }
    runLinear("linear_o_proj_w4a16", layer.o_proj, attn_hidden_, &scratch_hidden_, trace);
}

void CustomModel::runLinearAttention(Layer& layer, KernelTrace* trace) {
    runLinear("linear_attn_in_proj_qkv_w4a16", layer.linear_in_proj_qkv, normed_, &linear_mixed_, trace);
    runLinear("linear_attn_in_proj_a_w4a16", layer.linear_in_proj_a, normed_, &linear_a_, trace);
    runLinear("linear_attn_in_proj_b_w4a16", layer.linear_in_proj_b, normed_, &linear_b_, trace);
    runLinear("linear_attn_in_proj_z_w4a16", layer.linear_in_proj_z, normed_, &linear_z_, trace);

    {
        const auto t0 = Clock::now();
        const int history = std::max(0, linear_conv_kernel_dim_ - 1);
        bool used_vk = false;
        if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
            std::vector<float> next_state(layer.linear_conv_state.size(), 0.0f);
            kernels::VulkanGemvTiming vk_timing;
            std::string vk_error;
            if (vulkan_backend_->linearAttentionConv1d(linear_mixed_.data(),
                                                       layer.linear_conv_state.data(),
                                                       layer.linear_conv_weight.data(),
                                                       linear_conv_dim_,
                                                       linear_conv_kernel_dim_,
                                                       linear_conv_.data(),
                                                       next_state.data(),
                                                       &vk_timing,
                                                       &vk_error)) {
                layer.linear_conv_state.swap(next_state);
                used_vk = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
            }
        }
        if (!used_vk) {
            for (int c = 0; c < linear_conv_dim_; ++c) {
                float acc = 0.0f;
                for (int k = 0; k < history; ++k) {
                    const size_t state_index = static_cast<size_t>(c * history + k);
                    const size_t weight_index = static_cast<size_t>(c * linear_conv_kernel_dim_ + k);
                    acc += layer.linear_conv_state[state_index] * layer.linear_conv_weight[weight_index];
                }
                acc += linear_mixed_[static_cast<size_t>(c)] *
                       layer.linear_conv_weight[static_cast<size_t>(c * linear_conv_kernel_dim_ + history)];
                linear_conv_[static_cast<size_t>(c)] = silu(acc);
            }
            for (int c = 0; c < linear_conv_dim_; ++c) {
                float* state = layer.linear_conv_state.data() + static_cast<size_t>(c * history);
                for (int k = 0; k + 1 < history; ++k) {
                    state[k] = state[k + 1];
                }
                if (history > 0) {
                    state[history - 1] = linear_mixed_[static_cast<size_t>(c)];
                }
            }
        }
        if (trace) {
            trace->add("linear_attention_conv1d_custom", elapsedMs(t0, Clock::now()), used_vk ? "vulkan" : "cpu");
        }
    }

    float* query = linear_conv_.data();
    float* key = linear_conv_.data() + linear_key_dim_;
    float* value = linear_conv_.data() + linear_key_dim_ * 2;
    {
        const auto t0 = Clock::now();
        bool used_vk_query = false;
        bool used_vk_key = false;
        if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
            kernels::VulkanGemvTiming vk_timing;
            std::string vk_error;
            if (vulkan_backend_->l2NormalizeHeads(query, linear_num_key_heads_, linear_key_head_dim_, &vk_timing, &vk_error)) {
                used_vk_query = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
            }
            vk_error.clear();
            if (vulkan_backend_->l2NormalizeHeads(key, linear_num_key_heads_, linear_key_head_dim_, &vk_timing, &vk_error)) {
                used_vk_key = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
            }
        }
        if (!used_vk_query) {
            l2NormalizeHeads(query, linear_num_key_heads_, linear_key_head_dim_);
        }
        if (!used_vk_key) {
            l2NormalizeHeads(key, linear_num_key_heads_, linear_key_head_dim_);
        }
        if (trace) {
            trace->add("linear_attention_qk_l2norm_custom",
                       elapsedMs(t0, Clock::now()),
                       (used_vk_query && used_vk_key) ? "vulkan" : "cpu");
        }
    }

    {
        const auto t0 = Clock::now();
        bool used_vk = false;
        if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
            kernels::VulkanGemvTiming vk_timing;
            std::string vk_error;
            if (vulkan_backend_->linearAttentionStateUpdate(linear_conv_.data(),
                                                           linear_a_.data(),
                                                           linear_b_.data(),
                                                           layer.linear_a_log.data(),
                                                           layer.linear_dt_bias.data(),
                                                           layer.linear_recurrent_state.data(),
                                                           linear_num_key_heads_,
                                                           linear_num_value_heads_,
                                                           linear_key_head_dim_,
                                                           linear_value_head_dim_,
                                                           attn_hidden_.data(),
                                                           &vk_timing,
                                                           &vk_error)) {
                used_vk = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
            }
        }
        if (!used_vk) {
            const int factor = std::max(1, linear_num_value_heads_ / std::max(1, linear_num_key_heads_));
            std::fill(attn_hidden_.begin(), attn_hidden_.end(), 0.0f);
            for (int vh = 0; vh < linear_num_value_heads_; ++vh) {
                const int kh = std::min(linear_num_key_heads_ - 1, vh / factor);
                const float beta = sigmoid(linear_b_[static_cast<size_t>(vh)]);
                const float a_log = layer.linear_a_log[static_cast<size_t>(vh)];
                const float dt_bias = layer.linear_dt_bias[static_cast<size_t>(vh)];
                const float gate = -std::exp(a_log) * softplus(linear_a_[static_cast<size_t>(vh)] + dt_bias);
                float* state = layer.linear_recurrent_state.data() +
                               static_cast<size_t>(vh * linear_key_head_dim_ * linear_value_head_dim_);
                float* out = attn_hidden_.data() + static_cast<size_t>(vh * linear_value_head_dim_);
                kernels::gatedDeltaDecodeReference(query + kh * linear_key_head_dim_,
                                                   key + kh * linear_key_head_dim_,
                                                   value + vh * linear_value_head_dim_,
                                                   beta,
                                                   gate,
                                                   linear_key_head_dim_,
                                                   linear_value_head_dim_,
                                                   state,
                                                   out);
            }
        }
        if (trace) {
            trace->add("linear_attention_state_update_custom", elapsedMs(t0, Clock::now()), used_vk ? "vulkan" : "cpu");
        }
    }
    {
        const auto t0 = Clock::now();
        bool used_vk = false;
        if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
            kernels::VulkanGemvTiming vk_timing;
            std::string vk_error;
            if (vulkan_backend_->gatedRmsNorm(attn_hidden_.data(),
                                             layer.linear_norm.data(),
                                             linear_z_.data(),
                                             linear_num_value_heads_,
                                             linear_value_head_dim_,
                                             rms_eps_,
                                             attn_hidden_.data(),
                                             &vk_timing,
                                             &vk_error)) {
                used_vk = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
            }
        }
        if (!used_vk) {
            applyGatedHeadRms(attn_hidden_.data(),
                              layer.linear_norm.data(),
                              linear_z_.data(),
                              linear_num_value_heads_,
                              linear_value_head_dim_,
                              rms_eps_,
                              attn_hidden_.data());
        }
        if (trace) {
            trace->add("linear_attention_gated_rmsnorm_custom", elapsedMs(t0, Clock::now()), used_vk ? "vulkan" : "cpu");
        }
    }
    runLinear("linear_attn_out_proj_w4a16", layer.linear_out_proj, attn_hidden_, &scratch_hidden_, trace);
}

void CustomModel::runLayer(Layer& layer, size_t position, KernelTrace* trace) {
    {
        const auto t0 = Clock::now();
        bool used_vk = false;
        if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
            kernels::VulkanGemvTiming vk_timing;
            std::string vk_error;
            if (vulkan_backend_->rmsNorm(
                    hidden_.data(), layer.input_norm.data(), hidden_size_, rms_eps_, normed_.data(), &vk_timing, &vk_error)) {
                used_vk = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
            }
        }
        if (!used_vk) {
            kernels::rmsnormReference(hidden_.data(), layer.input_norm.data(), hidden_size_, rms_eps_, normed_.data());
        }
        if (trace) {
            trace->add("rmsnorm_input", elapsedMs(t0, Clock::now()), used_vk ? "vulkan" : "cpu");
        }
    }
    if (layer.full_attention) {
        runFullAttention(layer, position, trace);
    } else {
        runLinearAttention(layer, trace);
    }
    {
        const auto t0 = Clock::now();
        bool used_vk = false;
        if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
            kernels::VulkanGemvTiming vk_timing;
            std::string vk_error;
            if (vulkan_backend_->residualAdd(hidden_.data(), scratch_hidden_.data(), hidden_size_, &vk_timing, &vk_error)) {
                used_vk = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
            }
        }
        if (!used_vk) {
            addResidual(&hidden_, scratch_hidden_);
        }
        if (trace) {
            trace->add("residual_add_attention_custom", elapsedMs(t0, Clock::now()), used_vk ? "vulkan" : "cpu");
        }
    }

    {
        const auto t0 = Clock::now();
        bool used_vk = false;
        if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
            kernels::VulkanGemvTiming vk_timing;
            std::string vk_error;
            if (vulkan_backend_->rmsNorm(
                    hidden_.data(), layer.post_norm.data(), hidden_size_, rms_eps_, normed_.data(), &vk_timing, &vk_error)) {
                used_vk = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
            }
        }
        if (!used_vk) {
            kernels::rmsnormReference(hidden_.data(), layer.post_norm.data(), hidden_size_, rms_eps_, normed_.data());
        }
        if (trace) {
            trace->add("rmsnorm_post_attention", elapsedMs(t0, Clock::now()), used_vk ? "vulkan" : "cpu");
        }
    }
    runLinear("linear_gate_proj_w4a16", layer.gate_proj, normed_, &gate_, trace);
    runLinear("linear_up_proj_w4a16", layer.up_proj, normed_, &up_, trace);
    {
        const auto t0 = Clock::now();
        if (ffn_.size() != static_cast<size_t>(intermediate_size_)) {
            ffn_.resize(static_cast<size_t>(intermediate_size_));
        }
        bool used_vk = false;
        if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
            kernels::VulkanGemvTiming vk_timing;
            std::string vk_error;
            if (vulkan_backend_->siluGateMul(
                    gate_.data(), up_.data(), intermediate_size_, ffn_.data(), &vk_timing, &vk_error)) {
                used_vk = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
            }
        }
        if (!used_vk) {
            for (int i = 0; i < intermediate_size_; ++i) {
                ffn_[static_cast<size_t>(i)] = silu(gate_[static_cast<size_t>(i)]) * up_[static_cast<size_t>(i)];
            }
        }
        if (trace) {
            trace->add("silu_gate_mul_custom", elapsedMs(t0, Clock::now()), used_vk ? "vulkan" : "cpu");
        }
    }
    runLinear("linear_down_proj_w4a16", layer.down_proj, ffn_, &scratch_hidden_, trace);
    {
        const auto t0 = Clock::now();
        bool used_vk = false;
        if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
            kernels::VulkanGemvTiming vk_timing;
            std::string vk_error;
            if (vulkan_backend_->residualAdd(hidden_.data(), scratch_hidden_.data(), hidden_size_, &vk_timing, &vk_error)) {
                used_vk = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
            }
        }
        if (!used_vk) {
            addResidual(&hidden_, scratch_hidden_);
        }
        if (trace) {
            trace->add("residual_add_mlp_custom", elapsedMs(t0, Clock::now()), used_vk ? "vulkan" : "cpu");
        }
    }
}

void CustomModel::runFinalNorm(KernelTrace* trace) {
    const auto t0 = Clock::now();
    bool used_vk = false;
    if (vulkan_backend_ && vulkan_backend_->isInitialized()) {
        kernels::VulkanGemvTiming vk_timing;
        std::string vk_error;
        if (vulkan_backend_->rmsNorm(
                hidden_.data(), final_norm_.data(), hidden_size_, rms_eps_, scratch_hidden_.data(), &vk_timing, &vk_error)) {
            used_vk = true;
            ++vulkan_tensor_calls_;
        } else {
            ++vulkan_tensor_failures_;
        }
    }
    if (!used_vk) {
        kernels::rmsnormReference(hidden_.data(), final_norm_.data(), hidden_size_, rms_eps_, scratch_hidden_.data());
    }
    hidden_.swap(scratch_hidden_);
    if (trace) {
        trace->add("rmsnorm_final", elapsedMs(t0, Clock::now()), used_vk ? "vulkan" : "cpu");
    }
}

bool CustomModel::sampleGreedy(int32_t* token_id, KernelTrace* trace, std::string* error) {
    if (lm_head_.rows <= 0 || lm_head_.cols != hidden_size_) {
        *error = "lm_head matrix is not loaded";
        return false;
    }
    const auto t0 = Clock::now();
    float best_value = -std::numeric_limits<float>::infinity();
    int best = 0;
    bool lm_head_vk = false;
    bool sampling_vk = false;
    if (vulkan_backend_ && vulkan_backend_->isInitialized() && lm_head_.bits == 4) {
        logits_.assign(static_cast<size_t>(lm_head_.rows), 0.0f);
        kernels::VulkanGemvTiming vk_timing;
        std::string vk_error;
        if (vulkan_backend_->gemvW4A16(lm_head_, hidden_.data(), logits_.data(), &vk_timing, &vk_error)) {
            lm_head_vk = true;
            ++vulkan_linear_calls_;
            vk_error.clear();
            if (vulkan_backend_->argmax(logits_.data(), lm_head_.rows, &best, &best_value, &vk_timing, &vk_error)) {
                sampling_vk = true;
                ++vulkan_tensor_calls_;
            } else {
                ++vulkan_tensor_failures_;
                best_value = logits_.empty() ? -std::numeric_limits<float>::infinity() : logits_[0];
                for (int i = 1; i < static_cast<int>(logits_.size()); ++i) {
                    const float value = logits_[static_cast<size_t>(i)];
                    if (value > best_value) {
                        best_value = value;
                        best = i;
                    }
                }
            }
        } else {
            ++vulkan_linear_failures_;
        }
    }
    if (!lm_head_vk && lm_head_.bits == 4) {
        best = kernels::gemvW4A16ArgmaxNeon(lm_head_, hidden_.data(), &best_value);
    } else if (!lm_head_vk) {
        runLinear("lm_head_custom_fallback_logits", lm_head_, hidden_, &logits_, trace);
        best_value = logits_.empty() ? -std::numeric_limits<float>::infinity() : logits_[0];
        for (int i = 1; i < static_cast<int>(logits_.size()); ++i) {
            const float value = logits_[static_cast<size_t>(i)];
            if (value > best_value) {
                best_value = value;
                best = i;
            }
        }
    }
    if (trace) {
        trace->add("lm_head_custom", elapsedMs(t0, Clock::now()), lm_head_vk ? "vulkan" : "cpu");
    }
    *token_id = best;
    if (trace) {
        trace->add("sampling_greedy_custom", elapsedMs(t0, Clock::now()), sampling_vk ? "vulkan" : "cpu");
    }
    return true;
}

bool CustomModel::decodeOne(int32_t* token_id, size_t position, KernelTrace* trace, std::string* error) {
    if (!token_id) {
        *error = "decodeOne token output is null";
        return false;
    }
    if (!sampleGreedy(token_id, trace, error)) {
        return false;
    }
    if (!loadEmbeddingRow(*token_id, &hidden_, trace, error)) {
        return false;
    }
    for (Layer& layer : layers_) {
        runLayer(layer, position, trace);
    }
    runFinalNorm(trace);
    return true;
}

std::string CustomModel::coverageSummary() const {
    std::ostringstream oss;
    const bool vk_enabled = vulkan_backend_ && vulkan_backend_->isInitialized();
    oss << "custom_decode_loop;hotpath_replaced=true;full_custom_decode=true;"
        << "backend_request=" << backend_ << ";"
        << "linear_backend=" << (vk_enabled ? "vulkan_attempt_unresident_w4a16_gemv" : "cpu") << ";"
        << "vulkan_linear_calls=" << vulkan_linear_calls_ << ";"
        << "vulkan_linear_failures=" << vulkan_linear_failures_ << ";"
        << "vulkan_tensor_calls=" << vulkan_tensor_calls_ << ";"
        << "vulkan_tensor_failures=" << vulkan_tensor_failures_ << ";"
        << "linear=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,linear_attn_qkv_a_b_z_out,lm_head;"
        << "rmsnorm=" << (vk_enabled ? "vulkan_attempt_vector" : "custom_cpu") << ";"
        << "rope=" << (vk_enabled ? "vulkan_attempt_vector" : "custom_cpu") << ";"
        << "attention=" << (vk_enabled ? "vulkan_attempt_gqa_decode" : "custom_cpu") << ";attention_output_gate="
        << (vk_enabled ? "vulkan_attempt_vector" : "custom_cpu") << ";"
        << "linear_attention_state=" << (vk_enabled ? "vulkan_attempt_conv_state_update" : "custom_gated_delta_cpu")
        << ";linear_attention_qk_norm="
        << (vk_enabled ? "vulkan_attempt_vector" : "custom_cpu") << ";"
        << "linear_attention_gated_rmsnorm=" << (vk_enabled ? "vulkan_attempt_vector" : "custom_cpu") << ";"
        << "sampling=" << (vk_enabled ? "vulkan_attempt_argmax_after_logits" : "custom_greedy_cpu") << ";"
        << "prefill_kv_build=" << (vk_enabled ? "vulkan_attempt_append" : "custom_cpu") << ";fallback_ops=none";
    return oss.str();
}

std::string CustomModel::debugJson(int top_k) const {
    const int k = std::max(1, std::min(top_k, 16));
    double sum = 0.0;
    double sum_sq = 0.0;
    float min_value = hidden_.empty() ? 0.0f : hidden_[0];
    float max_value = min_value;
    int nonfinite = 0;
    for (float value : hidden_) {
        if (!std::isfinite(value)) {
            ++nonfinite;
            continue;
        }
        sum += static_cast<double>(value);
        sum_sq += static_cast<double>(value) * static_cast<double>(value);
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }
    const double n = static_cast<double>(hidden_.size());
    const double mean = n > 0.0 ? sum / n : 0.0;
    const double variance = n > 0.0 ? std::max(0.0, sum_sq / n - mean * mean) : 0.0;

    std::vector<float> logits(static_cast<size_t>(std::max(0, lm_head_.rows)), 0.0f);
    if (lm_head_.rows > 0 && lm_head_.cols == hidden_size_) {
        kernels::gemvW4A16Neon(lm_head_, hidden_.data(), logits.data());
    }
    std::vector<int> best_indices;
    best_indices.reserve(static_cast<size_t>(k));
    for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
        const float value = logits[static_cast<size_t>(i)];
        size_t pos = 0;
        while (pos < best_indices.size() && logits[static_cast<size_t>(best_indices[pos])] >= value) {
            ++pos;
        }
        if (pos < static_cast<size_t>(k)) {
            best_indices.insert(best_indices.begin() + static_cast<std::ptrdiff_t>(pos), i);
            if (best_indices.size() > static_cast<size_t>(k)) {
                best_indices.pop_back();
            }
        }
    }

    std::ostringstream oss;
    oss << "{\"hidden\":{\"size\":" << hidden_.size() << ",\"mean\":" << mean << ",\"std\":"
        << std::sqrt(variance) << ",\"min\":" << min_value << ",\"max\":" << max_value
        << ",\"l2\":" << std::sqrt(sum_sq) << ",\"nonfinite\":" << nonfinite << "},\"lm_head_topk\":[";
    for (size_t i = 0; i < best_indices.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        const int token = best_indices[i];
        oss << "{\"token\":" << token << ",\"logit\":" << logits[static_cast<size_t>(token)] << "}";
    }
    oss << "]}";
    return oss.str();
}

bool CustomModel::copyHidden(float* out_values, size_t value_capacity, size_t* out_size) const {
    if (out_size) {
        *out_size = hidden_.size();
    }
    if (!out_values || value_capacity < hidden_.size()) {
        return false;
    }
    std::copy(hidden_.begin(), hidden_.end(), out_values);
    return true;
}

}  // namespace xq
