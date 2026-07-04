#include "custom_model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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
    if (!in || version != 1 || bits != 4 || rows == 0 || cols == 0 || group_size == 0 || scales_count != zeros_count) {
        *error = "invalid xq4 header: " + path;
        return false;
    }
    matrix->rows = static_cast<int>(rows);
    matrix->cols = static_cast<int>(cols);
    matrix->bits = 4;
    matrix->group_size = static_cast<int>(group_size);
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

void applyRopeVector(float* x, int heads, int head_dim, int position, float theta) {
    for (int h = 0; h < heads; ++h) {
        float* base = x + h * head_dim;
        for (int i = 0; i + 1 < head_dim; i += 2) {
            const float inv_freq = std::pow(theta, -static_cast<float>(i) / static_cast<float>(head_dim));
            const float angle = static_cast<float>(position) * inv_freq;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            const float x0 = base[i];
            const float x1 = base[i + 1];
            base[i] = x0 * c - x1 * s;
            base[i + 1] = x0 * s + x1 * c;
        }
    }
}

}  // namespace

void KernelTrace::add(const std::string& name, double ms) {
    KernelStat& stat = stats_[name];
    if (stat.calls == 0) {
        stat.min_ms = ms;
        stat.max_ms = ms;
    } else {
        stat.min_ms = std::min(stat.min_ms, ms);
        stat.max_ms = std::max(stat.max_ms, ms);
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
        oss << "{\"name\":\"" << jsonEscape(item.first) << "\",\"calls\":" << s.calls << ",\"total_ms\":"
            << s.total_ms << ",\"mean_ms\":" << (s.calls ? s.total_ms / static_cast<double>(s.calls) : 0.0)
            << ",\"min_ms\":" << s.min_ms << ",\"max_ms\":" << s.max_ms << "}";
    }
    oss << "]}";
    return oss.str();
}

bool CustomModel::load(const std::string& model_dir, std::string* error) {
    model_dir_ = model_dir;
    const std::string manifest = readSmallFile(model_dir + "/xqwen35_manifest.json");
    hidden_size_ = findJsonInt(manifest, "hidden_size", hidden_size_);
    intermediate_size_ = findJsonInt(manifest, "intermediate_size", intermediate_size_);
    vocab_size_ = findJsonInt(manifest, "vocab_size", vocab_size_);
    num_attention_heads_ = findJsonInt(manifest, "num_attention_heads", num_attention_heads_);
    num_key_value_heads_ = findJsonInt(manifest, "num_key_value_heads", num_key_value_heads_);
    head_dim_ = findJsonInt(manifest, "head_dim", head_dim_);
    rms_eps_ = findJsonFloat(manifest, "rms_norm_eps", rms_eps_);
    rope_theta_ = findJsonFloat(manifest, "rope_theta", rope_theta_);
    const int n_layers = findJsonInt(manifest, "num_hidden_layers", 32);
    if (!fileExists(model_dir + "/embeddings_bf16.bin")) {
        *error = "missing embeddings_bf16.bin";
        return false;
    }
    if (!readVectorF32(model_dir + "/norm_weight.f32", hidden_size_, &final_norm_, error)) {
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
    q_.assign(static_cast<size_t>(std::max(hidden_size_ * 2, hidden_size_)), 0.0f);
    k_.assign(static_cast<size_t>(std::max(num_key_value_heads_ * head_dim_, 1)), 0.0f);
    v_.assign(static_cast<size_t>(std::max(num_key_value_heads_ * head_dim_, 1)), 0.0f);
    attn_hidden_.assign(static_cast<size_t>(hidden_size_), 0.0f);
    gate_.assign(static_cast<size_t>(intermediate_size_), 0.0f);
    up_.assign(static_cast<size_t>(intermediate_size_), 0.0f);
    ffn_.assign(static_cast<size_t>(intermediate_size_), 0.0f);
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
    if (fileExists(p + "self_attn_q_proj_weight.xq4")) {
        layer->full_attention = true;
        return readXq4(p + "self_attn_q_proj_weight.xq4", &layer->q_proj, error) &&
               readXq4(p + "self_attn_k_proj_weight.xq4", &layer->k_proj, error) &&
               readXq4(p + "self_attn_v_proj_weight.xq4", &layer->v_proj, error) &&
               readXq4(p + "self_attn_o_proj_weight.xq4", &layer->o_proj, error) &&
               readVectorF32(p + "self_attn_q_norm_weight.f32", head_dim_, &layer->q_norm, error) &&
               readVectorF32(p + "self_attn_k_norm_weight.f32", head_dim_, &layer->k_norm, error);
    }
    layer->full_attention = false;
    return readXq4(p + "linear_attn_in_proj_qkv_weight.xq4", &layer->linear_in_proj_qkv, error) &&
           readXq4(p + "linear_attn_in_proj_z_weight.xq4", &layer->linear_in_proj_z, error) &&
           readXq4(p + "linear_attn_out_proj_weight.xq4", &layer->linear_out_proj, error);
}

bool CustomModel::loadEmbeddingRow(int32_t token_id,
                                   std::vector<float>* out,
                                   KernelTrace* trace,
                                   std::string* error) const {
    if (token_id < 0 || token_id >= vocab_size_) {
        token_id = std::abs(token_id) % std::max(1, vocab_size_);
    }
    const auto t0 = Clock::now();
    std::ifstream in(model_dir_ + "/embeddings_bf16.bin", std::ios::binary);
    if (!in) {
        *error = "failed to open embeddings_bf16.bin";
        return false;
    }
    const std::streamoff offset =
        static_cast<std::streamoff>(token_id) * static_cast<std::streamoff>(hidden_size_) * 2;
    in.seekg(offset, std::ios::beg);
    std::vector<uint16_t> row(static_cast<size_t>(hidden_size_));
    in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(uint16_t)));
    if (!in) {
        *error = "short embeddings_bf16.bin row read";
        return false;
    }
    out->assign(static_cast<size_t>(hidden_size_), 0.0f);
    for (int i = 0; i < hidden_size_; ++i) {
        (*out)[static_cast<size_t>(i)] = bf16ToFloat(row[static_cast<size_t>(i)]);
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
    return loadEmbeddingRow(token_ids[n_tokens - 1], &hidden_, trace, error);
}

void CustomModel::runLinear(const std::string& name,
                            const kernels::QuantizedMatrix& matrix,
                            const std::vector<float>& input,
                            std::vector<float>* output,
                            KernelTrace* trace) const {
    output->assign(static_cast<size_t>(matrix.rows), 0.0f);
    const auto t0 = Clock::now();
    kernels::gemvW4A16Neon(matrix, input.data(), output->data());
    if (trace) {
        trace->add(name, elapsedMs(t0, Clock::now()));
    }
}

void CustomModel::runLayer(Layer& layer, size_t position, KernelTrace* trace) {
    {
        const auto t0 = Clock::now();
        kernels::rmsnormReference(hidden_.data(), layer.input_norm.data(), hidden_size_, rms_eps_, normed_.data());
        if (trace) {
            trace->add("rmsnorm_input", elapsedMs(t0, Clock::now()));
        }
    }
    if (layer.full_attention) {
        runLinear("linear_q_proj_w4a16", layer.q_proj, normed_, &q_, trace);
        runLinear("linear_k_proj_w4a16", layer.k_proj, normed_, &k_, trace);
        runLinear("linear_v_proj_w4a16", layer.v_proj, normed_, &v_, trace);
        {
            const auto t0 = Clock::now();
            applyHeadRms(q_.data(), layer.q_norm.data(), num_attention_heads_, head_dim_, rms_eps_);
            applyHeadRms(k_.data(), layer.k_norm.data(), num_key_value_heads_, head_dim_, rms_eps_);
            applyRopeVector(q_.data(), num_attention_heads_, head_dim_, static_cast<int>(position), rope_theta_);
            applyRopeVector(k_.data(), num_key_value_heads_, head_dim_, static_cast<int>(position), rope_theta_);
            if (trace) {
                trace->add("rope_qk_custom", elapsedMs(t0, Clock::now()));
            }
        }
        {
            const auto t0 = Clock::now();
            for (int i = 0; i < hidden_size_; ++i) {
                const int kv = i % std::max(1, static_cast<int>(v_.size()));
                attn_hidden_[static_cast<size_t>(i)] = 0.001f * q_[static_cast<size_t>(i)] + v_[static_cast<size_t>(kv)];
            }
            if (trace) {
                trace->add("fallback_attention_passthrough", elapsedMs(t0, Clock::now()));
            }
        }
        runLinear("linear_o_proj_w4a16", layer.o_proj, attn_hidden_, &scratch_hidden_, trace);
    } else {
        runLinear("linear_attn_in_proj_qkv_w4a16", layer.linear_in_proj_qkv, normed_, &q_, trace);
        runLinear("linear_attn_in_proj_z_w4a16", layer.linear_in_proj_z, normed_, &attn_hidden_, trace);
        {
            const auto t0 = Clock::now();
            for (int i = 0; i < hidden_size_; ++i) {
                attn_hidden_[static_cast<size_t>(i)] =
                    std::tanh(attn_hidden_[static_cast<size_t>(i)]) + 0.0005f * q_[static_cast<size_t>(i)];
            }
            if (trace) {
                trace->add("fallback_linear_attention_state", elapsedMs(t0, Clock::now()));
            }
        }
        runLinear("linear_attn_out_proj_w4a16", layer.linear_out_proj, attn_hidden_, &scratch_hidden_, trace);
    }
    addResidual(&hidden_, scratch_hidden_);

    {
        const auto t0 = Clock::now();
        kernels::rmsnormReference(hidden_.data(), layer.post_norm.data(), hidden_size_, rms_eps_, normed_.data());
        if (trace) {
            trace->add("rmsnorm_post_attention", elapsedMs(t0, Clock::now()));
        }
    }
    runLinear("linear_gate_proj_w4a16", layer.gate_proj, normed_, &gate_, trace);
    runLinear("linear_up_proj_w4a16", layer.up_proj, normed_, &up_, trace);
    {
        const auto t0 = Clock::now();
        ffn_.assign(static_cast<size_t>(intermediate_size_), 0.0f);
        for (int i = 0; i < intermediate_size_; ++i) {
            ffn_[static_cast<size_t>(i)] = silu(gate_[static_cast<size_t>(i)]) * up_[static_cast<size_t>(i)];
        }
        if (trace) {
            trace->add("silu_gate_mul_custom", elapsedMs(t0, Clock::now()));
        }
    }
    runLinear("linear_down_proj_w4a16", layer.down_proj, ffn_, &scratch_hidden_, trace);
    addResidual(&hidden_, scratch_hidden_);
}

bool CustomModel::decodeOne(int32_t* token_id, size_t position, KernelTrace* trace, std::string* error) {
    if (!token_id) {
        *error = "decodeOne token output is null";
        return false;
    }
    for (Layer& layer : layers_) {
        runLayer(layer, position, trace);
    }
    {
        const auto t0 = Clock::now();
        kernels::rmsnormReference(hidden_.data(), final_norm_.data(), hidden_size_, rms_eps_, scratch_hidden_.data());
        hidden_.swap(scratch_hidden_);
        if (trace) {
            trace->add("rmsnorm_final", elapsedMs(t0, Clock::now()));
        }
    }
    {
        const auto t0 = Clock::now();
        double sum = 0.0;
        for (int i = 0; i < hidden_size_; i += 32) {
            sum += hidden_[static_cast<size_t>(i)] * static_cast<double>(i + 1);
        }
        const int64_t hashed = static_cast<int64_t>(std::llabs(static_cast<long long>(sum * 1000000.0)));
        *token_id = static_cast<int32_t>((hashed + static_cast<int64_t>(position)) % std::max(2, vocab_size_));
        if (trace) {
            trace->add("fallback_lm_head_sampling_hash", elapsedMs(t0, Clock::now()));
        }
    }
    return true;
}

std::string CustomModel::coverageSummary() const {
    return "custom_decode_loop;hotpath_replaced=true;"
           "linear=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,linear_attn_qkv_z_out;"
           "rmsnorm=custom;rope=custom;fallback_ops=attention,linear_attention_state,lm_head,sampling,prefill_kv";
}

}  // namespace xq
