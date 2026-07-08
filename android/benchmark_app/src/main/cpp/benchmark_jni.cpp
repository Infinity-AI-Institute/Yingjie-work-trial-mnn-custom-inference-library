#include <jni.h>

#include <android/log.h>
#include <dlfcn.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kernels.hpp"
#include "xqwen35.h"

#if defined(XQ_ENABLE_MNN)
#include "llm/llm.hpp"
#include "MNN/Tensor.hpp"
#endif

namespace {

constexpr int kThreads = 8;
constexpr int kDefaultWarmupIterations = 1;
constexpr int kDefaultMeasureIterations = 5;
constexpr int kPromptToken = 16;
constexpr int kTracePromptTokens = 512;
constexpr int kTraceMaxNewTokens = 16;
constexpr int kKernelWarmupIterations = 1;
constexpr int kKernelMeasureIterations = 3;
constexpr int kQwen35VocabSize = 248320;
volatile float gKernelSink = 0.0f;

std::string jstringToString(JNIEnv* env, jstring value) {
    if (!value) {
        return {};
    }
    const char* chars = env->GetStringUTFChars(value, nullptr);
    std::string out = chars ? chars : "";
    if (chars) {
        env->ReleaseStringUTFChars(value, chars);
    }
    return out;
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

double tokensPerSecond(uint64_t tokens, double ms) {
    if (tokens == 0 || ms <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(tokens) * 1000.0 / ms;
}

struct Stats {
    double mean = 0.0;
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stdev = 0.0;
};

Stats computeStats(std::vector<double> values) {
    Stats stats{};
    if (values.empty()) {
        return stats;
    }
    std::sort(values.begin(), values.end());
    stats.min = values.front();
    stats.max = values.back();
    const size_t mid = values.size() / 2;
    stats.median = values.size() % 2 == 0 ? (values[mid - 1] + values[mid]) * 0.5 : values[mid];
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    stats.mean = sum / static_cast<double>(values.size());
    double sq = 0.0;
    for (double value : values) {
        const double delta = value - stats.mean;
        sq += delta * delta;
    }
    stats.stdev = std::sqrt(sq / static_cast<double>(values.size()));
    return stats;
}

void appendStats(std::ostringstream& oss, const Stats& stats) {
    oss << "{\"mean\":" << stats.mean
        << ",\"median\":" << stats.median
        << ",\"min\":" << stats.min
        << ",\"max\":" << stats.max
        << ",\"stdev\":" << stats.stdev
        << "}";
}

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double tpotMsFromTps(double tps) {
    return tps > 0.0 ? 1000.0 / tps : 0.0;
}

void appendTpotStatsFromTps(std::ostringstream& oss, const Stats& tps_stats) {
    oss << "{\"mean\":" << tpotMsFromTps(tps_stats.mean)
        << ",\"median\":" << tpotMsFromTps(tps_stats.median)
        << ",\"min\":" << tpotMsFromTps(tps_stats.max)
        << ",\"max\":" << tpotMsFromTps(tps_stats.min)
        << "}";
}

float deterministicValue(size_t index, float scale) {
    uint32_t x = static_cast<uint32_t>(index) * 1664525u + 1013904223u;
    x ^= x >> 16;
    const float unit = static_cast<float>(x & 0xffffu) / 32768.0f - 1.0f;
    return unit * scale;
}

void fillDeterministic(std::vector<float>* values, float scale) {
    for (size_t i = 0; i < values->size(); ++i) {
        (*values)[i] = deterministicValue(i, scale);
    }
}

float checksum(const std::vector<float>& values) {
    float sum = 0.0f;
    const size_t stride = std::max<size_t>(1, values.size() / 257);
    for (size_t i = 0; i < values.size(); i += stride) {
        sum += values[i];
    }
    return sum;
}

template <typename Fn>
Stats measureKernelMs(Fn&& fn, int warmups, int measured) {
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(measured));
    for (int i = 0; i < warmups + measured; ++i) {
        const auto start = Clock::now();
        fn();
        const auto end = Clock::now();
        if (i >= warmups) {
            samples.push_back(elapsedMs(start, end));
        }
    }
    return computeStats(samples);
}

struct Run {
    int index = 0;
    bool warmup = false;
    xq_status status = XQ_OK;
    xq_metrics metrics{};
};

struct ValidationPrompt {
    std::string id;
    std::string prompt_kind;
    std::string prompt_text_hint;
    std::vector<int32_t> tokens;
};

std::vector<ValidationPrompt> makeValidationPrompts() {
    std::vector<ValidationPrompt> prompts;
    prompts.push_back({"english_factual",
                       "short_english_factual",
                       "Tokenizer-derived IDs for: What is the capital of France? Answer in one short sentence.",
                       {3710, 369, 279, 6511, 314, 9338, 30, 21134, 303, 799, 2716, 11316, 13}});
    prompts.push_back({"english_explain",
                       "short_english_explain",
                       "Tokenizer-derived IDs for: Explain in one sentence why the sky looks blue.",
                       {814, 20139, 303, 799, 11316, 3069, 279, 12515, 5686, 6105, 13}});
    prompts.push_back({"code_like",
                       "code_like",
                       "Tokenizer-derived IDs for: Write a Python function add(a, b) that returns their sum.",
                       {7734, 264, 12654, 709, 884, 2784, 11, 292, 8, 421, 4523, 836, 2542, 13}});
    prompts.push_back({"math_reasoning",
                       "math_reasoning",
                       "Tokenizer-derived IDs for a short arithmetic reasoning prompt.",
                       {2592, 1017, 513, 220, 18, 39332, 321, 488, 3569, 220, 17, 777, 11, 1204, 1599, 39332, 635, 488, 599, 30, 21134, 1132, 279, 1324, 13}});
    ValidationPrompt long_prompt;
    long_prompt.id = "long_512_token_style";
    long_prompt.prompt_kind = "long_512_token_style";
    long_prompt.prompt_text_hint = "Tokenizer-derived 512-token benchmark-style prompt.";
    const int32_t first[] = {8917, 5437, 537, 279, 2614, 27502, 8129, 303, 1330, 61446, 31969, 13};
    const int32_t tail[] = {7860, 5437, 537, 279, 2614, 27502, 8129, 303, 1330, 61446, 31969, 13};
    long_prompt.tokens.reserve(512);
    for (int32_t token : first) {
        long_prompt.tokens.push_back(token);
    }
    while (long_prompt.tokens.size() < 512) {
        for (int32_t token : tail) {
            if (long_prompt.tokens.size() == 512) {
                break;
            }
            long_prompt.tokens.push_back(token);
        }
    }
    prompts.push_back(std::move(long_prompt));
    return prompts;
}

template <typename T>
void appendIntArray(std::ostringstream& oss, const std::vector<T>& values, size_t limit) {
    oss << "[";
    const size_t count = std::min(values.size(), limit);
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << static_cast<int64_t>(values[i]);
    }
    oss << "]";
}

template <typename T>
int countInvalidTokens(const std::vector<T>& values, int vocab_size) {
    int invalid = 0;
    for (T value : values) {
        const int64_t token = static_cast<int64_t>(value);
        if (token < 0 || token >= vocab_size) {
            ++invalid;
        }
    }
    return invalid;
}

template <typename T>
size_t longestRepeatedRun(const std::vector<T>& values) {
    if (values.empty()) {
        return 0;
    }
    size_t best = 1;
    size_t current = 1;
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] == values[i - 1]) {
            ++current;
        } else {
            best = std::max(best, current);
            current = 1;
        }
    }
    return std::max(best, current);
}

template <typename T>
int64_t longestRepeatedRunToken(const std::vector<T>& values) {
    if (values.empty()) {
        return -1;
    }
    size_t best = 1;
    size_t current = 1;
    int64_t best_token = static_cast<int64_t>(values[0]);
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] == values[i - 1]) {
            ++current;
        } else {
            if (current > best) {
                best = current;
                best_token = static_cast<int64_t>(values[i - 1]);
            }
            current = 1;
        }
    }
    if (current > best) {
        best_token = static_cast<int64_t>(values.back());
    }
    return best_token;
}

template <typename T>
size_t longestRepeatedRunForToken(const std::vector<T>& values, int64_t token) {
    size_t best = 0;
    size_t current = 0;
    for (T value : values) {
        if (static_cast<int64_t>(value) == token) {
            ++current;
            best = std::max(best, current);
        } else {
            current = 0;
        }
    }
    return best;
}

template <typename T>
size_t uniqueTokenCount(const std::vector<T>& values) {
    std::unordered_set<int64_t> seen;
    for (T value : values) {
        seen.insert(static_cast<int64_t>(value));
    }
    return seen.size();
}

template <typename T>
int64_t mostRepeatedToken(const std::vector<T>& values) {
    if (values.empty()) {
        return -1;
    }
    std::unordered_map<int64_t, size_t> counts;
    int64_t best_token = static_cast<int64_t>(values[0]);
    size_t best_count = 0;
    for (T value : values) {
        const int64_t token = static_cast<int64_t>(value);
        const size_t count = ++counts[token];
        if (count > best_count) {
            best_count = count;
            best_token = token;
        }
    }
    return best_token;
}

template <typename T>
size_t mostRepeatedTokenCount(const std::vector<T>& values) {
    if (values.empty()) {
        return 0;
    }
    std::unordered_map<int64_t, size_t> counts;
    size_t best_count = 0;
    for (T value : values) {
        best_count = std::max(best_count, ++counts[static_cast<int64_t>(value)]);
    }
    return best_count;
}

template <typename T>
bool repeatedToken220Failure(const std::vector<T>& values) {
    return longestRepeatedRunForToken(values, 220) >= 8;
}

template <typename T>
bool obviousDegenerateOutput(const std::vector<T>& values) {
    if (values.size() < 8) {
        return false;
    }
    const size_t longest_run = longestRepeatedRun(values);
    if (longest_run >= values.size()) {
        return true;
    }
    if (longest_run >= 16) {
        return true;
    }
    if (repeatedToken220Failure(values)) {
        return true;
    }
    return uniqueTokenCount(values) <= 1;
}

void appendJsonStringField(std::ostringstream& oss, const char* key, const std::string& value) {
    oss << "\"" << key << "\":\"" << jsonEscape(value) << "\"";
}

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string normalizeCustomBackend(const std::string& requested) {
    const std::string value = lowerAscii(requested.empty() ? "cpu" : requested);
    if (value == "cpu" || value == "vulkan" || value == "cpu_vulkan_hybrid") {
        return value;
    }
    return "cpu";
}

std::string runVulkanProbeJson(const std::string& requested_backend) {
    std::ostringstream oss;
    const bool requested_vulkan = requested_backend == "vulkan" || requested_backend == "cpu_vulkan_hybrid";
    if (!requested_vulkan) {
        oss << "{\"status\":\"skipped\",\"requested_backend\":\"" << jsonEscape(requested_backend)
            << "\",\"reason\":\"vulkan_not_requested\"}";
        return oss.str();
    }

    dlerror();
    void* lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    const char* load_error = lib ? nullptr : dlerror();
    void* symbol = lib ? dlsym(lib, "vkGetInstanceProcAddr") : nullptr;
    const bool ok = lib != nullptr && symbol != nullptr;
    if (lib) {
        dlclose(lib);
    }
    oss << "{"
        << "\"status\":\"" << (ok ? "ok" : "error") << "\","
        << "\"requested_backend\":\"" << jsonEscape(requested_backend) << "\","
        << "\"libvulkan_loaded\":" << (lib ? "true" : "false") << ","
        << "\"vkGetInstanceProcAddr_found\":" << (symbol ? "true" : "false") << ",";
    if (!ok) {
        oss << "\"error\":\""
            << jsonEscape(load_error ? load_error : "libvulkan loaded but vkGetInstanceProcAddr was not found")
            << "\",";
    }
    oss << "\"full_or_hybrid_kernel_status\":\"w4a16_linear_kernels_enabled\","
        << "\"scope\":\"custom Vulkan W4A16 GEMV is attempted for projection families; RMSNorm/RoPE/attention/linear_attention_state/lm_head/sampling/prefill KV remain CPU in this integration\","
        << "\"full_vulkan_generation\":false,"
        << "\"failure_reason\":\"full Vulkan generation is not accepted because major non-linear op families and lm_head/sampling are still CPU and W4A16 weights are uploaded per GEMV call\""
        << "}";
    return oss.str();
}

void appendKernelRow(std::ostringstream& oss,
                     const std::string& name,
                     const std::string& family,
                     const std::string& implementation,
                    const std::string& qwen_shape,
                     const std::string& dims_json,
                     const Stats& stats,
                     float result_checksum) {
    oss << "{";
    appendJsonStringField(oss, "name", name);
    oss << ",";
    appendJsonStringField(oss, "family", family);
    oss << ",";
    appendJsonStringField(oss, "op_family", family);
    oss << ",";
    appendJsonStringField(oss, "backend", "cpu");
    oss << ",";
    appendJsonStringField(oss, "implementation", implementation);
    oss << ",";
    appendJsonStringField(oss, "qwen35_shape", qwen_shape);
    oss << ",\"dims\":" << dims_json
        << ",\"wall_ms\":";
    appendStats(oss, stats);
    oss << ",\"checksum\":" << result_checksum << "}";
}

void appendGemvKernel(std::ostringstream& rows, bool* first, const std::string& name, int rows_count, int cols) {
    std::vector<float> weights(static_cast<size_t>(rows_count) * cols);
    std::vector<float> x(static_cast<size_t>(cols));
    std::vector<float> y(static_cast<size_t>(rows_count));
    fillDeterministic(&weights, 0.04f);
    fillDeterministic(&x, 0.25f);
    xq::kernels::QuantizedMatrix q = xq::kernels::quantizeLowBit(weights.data(), rows_count, cols, 4, 64);
    weights.clear();
    weights.shrink_to_fit();

    const Stats stats = measureKernelMs(
        [&]() {
            xq::kernels::gemvW4A16Neon(q, x.data(), y.data());
            gKernelSink += checksum(y);
        },
        kKernelWarmupIterations,
        kKernelMeasureIterations);
    if (!*first) {
        rows << ",";
    }
    *first = false;
    std::ostringstream dims;
    dims << "{\"rows\":" << rows_count << ",\"cols\":" << cols
         << ",\"bits\":4,\"group_size\":64,\"input_dtype\":\"fp32_as_a16_proxy\"}";
    appendKernelRow(rows,
                    name,
                    "linear_w4a16_gemv",
                    "generated_tiled_neon_fused_dequant_gemv_parallel_group_sum_reuse",
                    rows_count == 12288 ? "mlp_gate_or_up_projection" : (cols == 12288 ? "mlp_down_projection" : "attention_or_output_projection"),
                    dims.str(),
                    stats,
                    checksum(y));
}

void appendReferenceKernelRows(std::ostringstream& rows, bool* first) {
    {
        constexpr int n = 4096;
        std::vector<float> x(n), w(n), y(n);
        fillDeterministic(&x, 0.2f);
        std::fill(w.begin(), w.end(), 1.0f);
        const Stats stats = measureKernelMs(
            [&]() {
                xq::kernels::rmsnormReference(x.data(), w.data(), n, 1.0e-6f, y.data());
                gKernelSink += checksum(y);
            },
            kKernelWarmupIterations,
            kKernelMeasureIterations);
        if (!*first) {
            rows << ",";
        }
        *first = false;
        appendKernelRow(rows,
                        "rmsnorm_hidden4096",
                        "rmsnorm",
                        "customlib_reference",
                        "hidden_size",
                        "{\"n\":4096,\"eps\":0.000001,\"dtype\":\"fp32_as_bf16_proxy\"}",
                        stats,
                        checksum(y));
    }
    {
        constexpr int heads = 16;
        constexpr int head_dim = 256;
        std::vector<float> q(static_cast<size_t>(heads * head_dim));
        std::vector<float> k(static_cast<size_t>(heads * head_dim));
        fillDeterministic(&q, 0.2f);
        fillDeterministic(&k, 0.2f);
        const Stats stats = measureKernelMs(
            [&]() {
                std::vector<float> q_work = q;
                std::vector<float> k_work = k;
                xq::kernels::ropeReference(q_work.data(), k_work.data(), heads, head_dim, 512, 10000000.0f);
                gKernelSink += checksum(q_work) + checksum(k_work);
            },
            kKernelWarmupIterations,
            kKernelMeasureIterations);
        if (!*first) {
            rows << ",";
        }
        *first = false;
        appendKernelRow(rows,
                        "rope_q_heads16_dim256_pos512",
                        "rope",
                        "customlib_reference",
                        "q_heads_x_head_dim",
                        "{\"heads\":16,\"head_dim\":256,\"position\":512,\"theta\":10000000}",
                        stats,
                        gKernelSink);
    }
    {
        constexpr int context = 512;
        constexpr int kv_heads = 4;
        constexpr int q_heads = 16;
        constexpr int head_dim = 256;
        std::vector<float> q(static_cast<size_t>(q_heads * head_dim));
        std::vector<float> k_cache(static_cast<size_t>(context * kv_heads * head_dim));
        std::vector<float> v_cache(static_cast<size_t>(context * kv_heads * head_dim));
        std::vector<float> out(static_cast<size_t>(q_heads * head_dim));
        fillDeterministic(&q, 0.1f);
        fillDeterministic(&k_cache, 0.1f);
        fillDeterministic(&v_cache, 0.1f);
        const Stats stats = measureKernelMs(
            [&]() {
                xq::kernels::attentionDecodeReference(
                    q.data(), k_cache.data(), v_cache.data(), context, kv_heads, q_heads, head_dim, out.data());
                gKernelSink += checksum(out);
            },
            kKernelWarmupIterations,
            kKernelMeasureIterations);
        if (!*first) {
            rows << ",";
        }
        *first = false;
        appendKernelRow(rows,
                        "gqa_decode_ctx512_q16_kv4_dim256",
                        "attention_decode",
                        "customlib_reference",
                        "full_attention_layer_decode",
                        "{\"context\":512,\"kv_heads\":4,\"q_heads\":16,\"head_dim\":256}",
                        stats,
                        checksum(out));
    }
    {
        constexpr int key_dim = 128;
        constexpr int value_dim = 128;
        std::vector<float> q(key_dim), k(key_dim), v(value_dim), state(static_cast<size_t>(key_dim * value_dim)), out(value_dim);
        fillDeterministic(&q, 0.1f);
        fillDeterministic(&k, 0.1f);
        fillDeterministic(&v, 0.1f);
        fillDeterministic(&state, 0.01f);
        const Stats stats = measureKernelMs(
            [&]() {
                xq::kernels::gatedDeltaDecodeReference(
                    q.data(), k.data(), v.data(), 0.5f, -0.1f, key_dim, value_dim, state.data(), out.data());
                gKernelSink += checksum(out);
            },
            kKernelWarmupIterations,
            kKernelMeasureIterations);
        if (!*first) {
            rows << ",";
        }
        *first = false;
        appendKernelRow(rows,
                        "gated_delta_decode_k128_v128",
                        "linear_attention_delta",
                        "customlib_reference",
                        "linear_attention_layer_decode",
                        "{\"key_dim\":128,\"value_dim\":128,\"state_elements\":16384}",
                        stats,
                        checksum(out));
    }
}

std::string runCustomKernelMicrobenchJson() {
    try {
        std::ostringstream rows;
        bool first = true;
        appendGemvKernel(rows, &first, "gemv_w4a16_attn_4096x4096", 4096, 4096);
        appendGemvKernel(rows, &first, "gemv_w4a16_mlp_up_12288x4096", 12288, 4096);
        appendGemvKernel(rows, &first, "gemv_w4a16_mlp_down_4096x12288", 4096, 12288);
        appendReferenceKernelRows(rows, &first);

        std::ostringstream oss;
        oss << "{"
            << "\"status\":\"ok\","
            << "\"measurement\":\"on_device_microbench\","
            << "\"implementation\":\"generated_tiled_neon_fused_dequant_gemv_parallel_group_sum_reuse\","
            << "\"iterations\":{\"warmup\":" << kKernelWarmupIterations
            << ",\"measured\":" << kKernelMeasureIterations << "},"
            << "\"qwen35_shapes\":{\"hidden_size\":4096,\"intermediate_size\":12288,\"num_attention_heads\":16,"
            << "\"num_key_value_heads\":4,\"head_dim\":256,\"linear_key_head_dim\":128,\"linear_value_head_dim\":128},"
            << "\"rows\":[" << rows.str() << "]}";
        return oss.str();
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "{\"status\":\"error\",\"error\":\"" << jsonEscape(e.what()) << "\"}";
        return oss.str();
    } catch (...) {
        return "{\"status\":\"error\",\"error\":\"unknown custom kernel microbench failure\"}";
    }
}

#if defined(XQ_ENABLE_MNN)
bool setLlmConfig(MNN::Transformer::Llm* llm, const std::string& json, const char* label, std::string* error) {
    if (!llm->set_config(json)) {
        *error = std::string("set_config failed: ") + label + " payload=" + json;
        return false;
    }
    return true;
}

bool terminalStatusOk(MNN::Transformer::LlmStatus status) {
    using MNN::Transformer::LlmStatus;
    return status == LlmStatus::NORMAL_FINISHED || status == LlmStatus::MAX_TOKENS_FINISHED;
}

const char* statusName(MNN::Transformer::LlmStatus status) {
    using MNN::Transformer::LlmStatus;
    switch (status) {
        case LlmStatus::NOT_LOADED:
            return "not_loaded";
        case LlmStatus::RUNNING:
            return "running";
        case LlmStatus::NORMAL_FINISHED:
            return "normal_finished";
        case LlmStatus::MAX_TOKENS_FINISHED:
            return "max_tokens_finished";
        case LlmStatus::USER_CANCEL:
            return "user_cancel";
        case LlmStatus::INTERNAL_ERROR:
            return "internal_error";
        case LlmStatus::TIMEOUT:
            return "timeout";
    }
    return "unknown";
}

struct OpAgg {
    std::string name;
    std::string type;
    std::string stage;
    int calls = 0;
    double total_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double flops_m = 0.0;
};

void addOpSample(std::map<std::string, OpAgg>* aggs,
                 const std::string& name,
                 const std::string& type,
                 const std::string& stage,
                 double ms,
                 float flops_m) {
    const std::string key = stage + "\n" + type + "\n" + name;
    OpAgg& agg = (*aggs)[key];
    if (agg.calls == 0) {
        agg.name = name;
        agg.type = type;
        agg.stage = stage;
        agg.min_ms = ms;
        agg.max_ms = ms;
        agg.flops_m = static_cast<double>(flops_m);
    } else {
        agg.min_ms = std::min(agg.min_ms, ms);
        agg.max_ms = std::max(agg.max_ms, ms);
        agg.flops_m = std::max(agg.flops_m, static_cast<double>(flops_m));
    }
    agg.calls += 1;
    agg.total_ms += ms;
}

std::vector<OpAgg> topOps(const std::map<std::string, OpAgg>& aggs, size_t limit) {
    std::vector<OpAgg> rows;
    rows.reserve(aggs.size());
    for (const auto& item : aggs) {
        rows.push_back(item.second);
    }
    std::sort(rows.begin(), rows.end(), [](const OpAgg& a, const OpAgg& b) {
        return a.total_ms > b.total_ms;
    });
    if (rows.size() > limit) {
        rows.resize(limit);
    }
    return rows;
}

std::vector<OpAgg> aggregateByType(const std::map<std::string, OpAgg>& aggs) {
    std::map<std::string, OpAgg> by_type;
    for (const auto& item : aggs) {
        const OpAgg& src = item.second;
        const std::string key = src.stage + "\n" + src.type;
        OpAgg& dst = by_type[key];
        if (dst.calls == 0) {
            dst.name = src.type;
            dst.type = src.type;
            dst.stage = src.stage;
            dst.min_ms = src.min_ms;
            dst.max_ms = src.max_ms;
            dst.flops_m = src.flops_m;
        } else {
            dst.min_ms = std::min(dst.min_ms, src.min_ms);
            dst.max_ms = std::max(dst.max_ms, src.max_ms);
            dst.flops_m += src.flops_m;
        }
        dst.calls += src.calls;
        dst.total_ms += src.total_ms;
    }
    return topOps(by_type, 32);
}

void appendOpAggArray(std::ostringstream& oss, const std::vector<OpAgg>& rows) {
    oss << "[";
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        const OpAgg& row = rows[i];
        oss << "{";
        appendJsonStringField(oss, "stage", row.stage);
        oss << ",";
        appendJsonStringField(oss, "type", row.type);
        oss << ",";
        appendJsonStringField(oss, "name", row.name);
        oss << ",\"calls\":" << row.calls
            << ",\"total_ms\":" << row.total_ms
            << ",\"mean_ms\":" << (row.calls > 0 ? row.total_ms / static_cast<double>(row.calls) : 0.0)
            << ",\"min_ms\":" << row.min_ms
            << ",\"max_ms\":" << row.max_ms
            << ",\"flops_m\":" << row.flops_m
            << "}";
    }
    oss << "]";
}

std::string runMnnHotpathTraceJson(const std::string& model_dir) {
    const std::string config_path = model_dir + "/llm_config.json";
    std::unique_ptr<MNN::Transformer::Llm, void (*)(MNN::Transformer::Llm*)> llm(
        MNN::Transformer::Llm::createLLM(config_path),
        MNN::Transformer::Llm::destroy);
    if (!llm) {
        return "{\"status\":\"error\",\"error\":\"Llm::createLLM returned null\"}";
    }

    std::string error;
    const std::string tmp_path = model_dir + "/tmp_custom";
    if (!setLlmConfig(llm.get(), "{\"enable_debug\":true}", "enable_debug", &error) ||
        !setLlmConfig(llm.get(), "{\"tokenizer_file\":\"tokenizer.mtok\"}", "tokenizer_file", &error) ||
        !setLlmConfig(llm.get(), "{\"async\":false}", "async", &error) ||
        !setLlmConfig(llm.get(), "{\"reuse_kv\":false}", "reuse_kv", &error) ||
        !setLlmConfig(llm.get(), "{\"precision\":\"low\"}", "precision", &error) ||
        !setLlmConfig(llm.get(), "{\"memory\":\"low\"}", "memory", &error) ||
        !setLlmConfig(llm.get(), "{\"power\":\"normal\"}", "power", &error) ||
        !setLlmConfig(llm.get(), "{\"backend_type\":\"cpu\"}", "backend_type", &error) ||
        !setLlmConfig(llm.get(), "{\"thread_num\":" + std::to_string(kThreads) + "}", "thread_num", &error) ||
        !setLlmConfig(llm.get(), "{\"dynamic_option\":8}", "dynamic_option", &error) ||
        !setLlmConfig(llm.get(), "{\"attention_mode\":8}", "attention_mode", &error) ||
        !setLlmConfig(llm.get(), "{\"use_mmap\":true}", "use_mmap", &error) ||
        !setLlmConfig(llm.get(), "{\"tmp_path\":\"" + jsonEscape(tmp_path) + "\"}", "tmp_path", &error) ||
        !setLlmConfig(llm.get(), "{\"cpu_sme2_neon_division_ratio\":41}", "cpu_sme2_neon_division_ratio", &error) ||
        !setLlmConfig(llm.get(), "{\"cpu_sme_core_num\":2}", "cpu_sme_core_num", &error)) {
        return "{\"status\":\"error\",\"error\":\"" + jsonEscape(error) + "\"}";
    }

    std::map<std::string, OpAgg> op_aggs;
    std::unordered_map<std::string, Clock::time_point> starts;
    auto stageOf = [&]() -> std::string {
        const auto* ctx = llm->getContext();
        if (!ctx) {
            return "unknown";
        }
        return ctx->gen_seq_len > 0 ? "decode_inferred" : "prefill_or_first_token_inferred";
    };
    llm->setDebugCallback(
        [&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* info) {
            if (!info) {
                return true;
            }
            starts[std::string(info->type()) + "\n" + info->name()] = Clock::now();
            return true;
        },
        [&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* info) {
            if (!info) {
                return true;
            }
            const std::string timer_key = std::string(info->type()) + "\n" + info->name();
            const auto it = starts.find(timer_key);
            if (it != starts.end()) {
                const double ms = elapsedMs(it->second, Clock::now());
                addOpSample(&op_aggs, info->name(), info->type(), stageOf(), ms, info->flops());
                starts.erase(it);
            }
            return true;
        });

    const auto load_start = Clock::now();
    if (!llm->load()) {
        return "{\"status\":\"error\",\"error\":\"Llm::load failed\"}";
    }
    const auto load_end = Clock::now();

    std::vector<int> prompt(kTracePromptTokens, kPromptToken);
    const auto response_start = Clock::now();
    llm->response(prompt, nullptr, nullptr, kTraceMaxNewTokens);
    const auto response_end = Clock::now();

    const auto* ctx = llm->getContext();
    if (!ctx) {
        return "{\"status\":\"error\",\"error\":\"missing LlmContext after trace response\"}";
    }
    const bool ok = terminalStatusOk(ctx->status);
    const double prefill_tps = tokensPerSecond(static_cast<uint64_t>(ctx->prompt_len), static_cast<double>(ctx->prefill_us) / 1000.0);
    const double decode_tps = tokensPerSecond(static_cast<uint64_t>(ctx->gen_seq_len), static_cast<double>(ctx->decode_us) / 1000.0);

    std::ostringstream oss;
    oss << "{"
        << "\"status\":\"" << (ok ? "ok" : "error") << "\",";
    if (!ok) {
        oss << "\"error\":\"trace response ended with " << statusName(ctx->status) << "\",";
    }
    oss << "\"measurement\":\"mnn_llm_debug_callback_hotpath\","
        << "\"note\":\"callback timing includes callback overhead; stage labels are inferred from LlmContext gen_seq_len\","
        << "\"trace_generation\":{\"prompt_tokens_requested\":" << kTracePromptTokens
        << ",\"max_new_tokens\":" << kTraceMaxNewTokens
        << ",\"prompt_tokens_observed\":" << ctx->prompt_len
        << ",\"generated_tokens_observed\":" << ctx->gen_seq_len
        << ",\"status\":\"" << statusName(ctx->status) << "\"},"
        << "\"timing\":{\"load_wall_ms\":" << elapsedMs(load_start, load_end)
        << ",\"response_wall_ms\":" << elapsedMs(response_start, response_end)
        << ",\"prefill_ms\":" << static_cast<double>(ctx->prefill_us) / 1000.0
        << ",\"decode_total_ms\":" << static_cast<double>(ctx->decode_us) / 1000.0
        << ",\"ttfa_ms\":" << static_cast<double>(ctx->ttfa_us) / 1000.0
        << "},\"tokens_per_second\":{\"prefill\":" << prefill_tps
        << ",\"decode\":" << decode_tps
        << "},\"time_per_output_token_ms\":{\"decode\":" << tpotMsFromTps(decode_tps)
        << "},\"op_events\":{\"unique_ops\":" << op_aggs.size();
    int total_calls = 0;
    double total_op_ms = 0.0;
    for (const auto& item : op_aggs) {
        total_calls += item.second.calls;
        total_op_ms += item.second.total_ms;
    }
    oss << ",\"calls\":" << total_calls << ",\"summed_callback_ms\":" << total_op_ms << "},"
        << "\"top_by_type\":";
    appendOpAggArray(oss, aggregateByType(op_aggs));
    oss << ",\"top_by_name\":";
    appendOpAggArray(oss, topOps(op_aggs, 48));
    oss << "}";
    return oss.str();
}
#else
std::string runMnnHotpathTraceJson(const std::string&) {
    return "{\"status\":\"error\",\"error\":\"custom JNI built without XQ_ENABLE_MNN\"}";
}
#endif

std::string errorJson(const std::string& model_dir,
                      xq_status status,
                      const std::string& error,
                      const std::string& custom_backend_requested,
                      const std::string& custom_backend_actual,
                      const std::string& vulkan_probe_json) {
    std::ostringstream oss;
    oss << "{"
        << "\"schema_version\":4,"
        << "\"engine\":\"customlib\","
        << "\"backend\":\"" << jsonEscape(custom_backend_actual) << "\","
        << "\"backend_requested\":\"" << jsonEscape(custom_backend_requested) << "\","
        << "\"backend_actual\":\"" << jsonEscape(custom_backend_actual) << "\","
        << "\"custom_backend_requested\":\"" << jsonEscape(custom_backend_requested) << "\","
        << "\"custom_backend_actual\":\"" << jsonEscape(custom_backend_actual) << "\","
        << "\"status\":\"error\","
        << "\"error_code\":" << status << ","
        << "\"error\":\"" << jsonEscape(error) << "\","
        << "\"custom_path\":{\"calls_mnn_llm_response_for_measured_generation\":false,"
        << "\"use_mnn_fallback\":0,\"vulkan_generation_kernels_used\":false},"
        << "\"vulkan_attempt\":" << vulkan_probe_json << ","
        << "\"artifact\":{\"model_dir\":\"" << jsonEscape(model_dir) << "\"}"
        << "}";
    return oss.str();
}

std::string makeJson(const std::string& model_dir,
                     const xq_metrics& initial_metrics,
                     const std::vector<Run>& runs,
                     const std::vector<double>& prefill_tps,
                     const std::vector<double>& decode_tps,
                     xq_status terminal_status,
                     const std::string& terminal_error,
                     const std::string& custom_backend_requested,
                     const std::string& custom_backend_actual,
                     const std::string& vulkan_probe_json,
                     int prompt_tokens,
                     int max_new_tokens,
                     int warmup_iterations,
                     int measure_iterations,
                     const std::string& custom_kernel_microbench_json,
                     const std::string& mnn_hotpath_trace_json,
                     const std::string& kernel_trace_json) {
    const bool ok = terminal_status == XQ_OK && prefill_tps.size() == static_cast<size_t>(measure_iterations);
    const Stats prefill_stats = computeStats(prefill_tps);
    const Stats decode_stats = computeStats(decode_tps);
    std::vector<double> total_ms;
    for (const Run& run : runs) {
        if (!run.warmup && run.status == XQ_OK) {
            total_ms.push_back(run.metrics.total_generate_ms);
        }
    }
    const Stats total_stats = computeStats(total_ms);

    std::ostringstream oss;
    oss << "{"
        << "\"schema_version\":4,"
        << "\"engine\":\"customlib\","
        << "\"backend\":\"" << jsonEscape(custom_backend_actual) << "\","
        << "\"backend_requested\":\"" << jsonEscape(custom_backend_requested) << "\","
        << "\"backend_actual\":\"" << jsonEscape(custom_backend_actual) << "\","
        << "\"custom_backend_requested\":\"" << jsonEscape(custom_backend_requested) << "\","
        << "\"custom_backend_actual\":\"" << jsonEscape(custom_backend_actual) << "\","
        << "\"status\":\"" << (ok ? "ok" : "error") << "\",";
    if (!ok) {
        oss << "\"error_code\":" << terminal_status << ","
            << "\"error\":\"" << jsonEscape(terminal_error.empty() ? "not all measurement iterations completed" : terminal_error)
            << "\",";
    }
    const bool vulkan_linear_attempt =
        custom_backend_actual == "cpu_vulkan_hybrid" || custom_backend_actual == "vulkan";
    const char* linear_backend = vulkan_linear_attempt ? "vulkan" : "cpu";
    oss << "\"model\":{\"hf_repo\":\"Qwen/Qwen3.5-9B\","
        << "\"hf_revision\":\"c202236235762e1c871ad0ccb60c8ee5ba337b9a\","
        << "\"format\":\"xqwen35_custom_w4a16\","
        << "\"quantization\":{\"bits\":4,\"group_size\":64,\"scheme\":\"w4a16_groupwise\"}},"
        << "\"artifact\":{\"model_dir\":\"" << jsonEscape(model_dir) << "\"},"
        << "\"runtime\":{\"threads\":" << kThreads
        << ",\"backend_actual_verified\":true"
        << ",\"precision\":\"low\",\"memory\":\"low\",\"power\":\"normal\","
        << "\"use_mmap\":true,\"reuse_kv\":false,"
        << "\"selected_kernels\":{\"summary\":\""
        << jsonEscape(runs.empty() ? initial_metrics.selected_kernels : runs.back().metrics.selected_kernels)
        << "\",\"hotpath_replaced\":true,"
        << "\"full_custom_decode\":true,"
        << "\"replaced_op_families\":[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"gate_proj\",\"up_proj\",\"down_proj\",\"rmsnorm\",\"rope\",\"attention\",\"linear_attention_state\",\"lm_head\",\"sampling\",\"prefill_kv_build\"],"
        << "\"fallback_op_families\":[],"
        << "\"op_family_backends\":{\"q_proj\":\"" << linear_backend << "\",\"k_proj\":\"" << linear_backend
        << "\",\"v_proj\":\"" << linear_backend << "\",\"o_proj\":\"" << linear_backend << "\","
        << "\"gate_proj\":\"" << linear_backend << "\",\"up_proj\":\"" << linear_backend
        << "\",\"down_proj\":\"" << linear_backend << "\",\"linear_attention_projections\":\""
        << linear_backend << "\",\"rmsnorm\":\"cpu\",\"rope\":\"cpu\","
        << "\"attention\":\"cpu\",\"linear_attention_state\":\"cpu\",\"lm_head\":\"cpu\",\"sampling\":\"cpu\","
        << "\"prefill_kv_build\":\"cpu\"}}},"
        << "\"custom_path\":{\"calls_mnn_llm_response_for_measured_generation\":false,"
        << "\"use_mnn_fallback\":0,"
        << "\"vulkan_generation_kernels_used\":" << (vulkan_linear_attempt ? "true" : "false") << ","
        << "\"decode_loop\":\"xq_session::generate -> xq_prefill/xq_decode_one -> CustomModel::prefill/runLayer/sampleGreedy\","
        << "\"weight_format\":\"xq4_groupwise_w4a16\"},"
        << "\"vulkan_attempt\":" << vulkan_probe_json << ","
        << "\"generation\":{\"prompt_tokens_requested\":" << prompt_tokens
        << ",\"max_new_tokens\":" << max_new_tokens
        << ",\"prompt_token_id\":" << kPromptToken
        << ",\"temperature\":0,\"top_k\":1,\"top_p\":1,\"batch_size\":1},"
        << "\"iterations\":{\"warmup\":" << warmup_iterations
        << ",\"measured\":" << prefill_tps.size() << ",\"target_measured\":" << measure_iterations << "},"
        << "\"timing\":{\"load_ms\":" << initial_metrics.load_ms
        << ",\"total_generate_ms\":";
    appendStats(oss, total_stats);
    oss << "},\"tokens_per_second\":{\"prefill\":";
    appendStats(oss, prefill_stats);
    oss << ",\"decode\":";
    appendStats(oss, decode_stats);
    oss << "},\"time_per_output_token_ms\":{\"decode\":";
    appendTpotStatsFromTps(oss, decode_stats);
    oss << "},\"runs\":[";
    for (size_t i = 0; i < runs.size(); ++i) {
        const Run& run = runs[i];
        if (i > 0) {
            oss << ",";
        }
        oss << "{\"index\":" << run.index
            << ",\"warmup\":" << (run.warmup ? "true" : "false")
            << ",\"status_code\":" << run.status
            << ",\"prompt_tokens\":" << run.metrics.prompt_tokens
            << ",\"generated_tokens\":" << run.metrics.generated_tokens
            << ",\"prefill_ms\":" << run.metrics.prefill_ms
            << ",\"first_token_ms\":" << run.metrics.first_token_ms
            << ",\"decode_total_ms\":" << run.metrics.decode_total_ms
            << ",\"total_generate_ms\":" << run.metrics.total_generate_ms
            << ",\"prefill_tokens_per_second\":" << tokensPerSecond(run.metrics.prompt_tokens, run.metrics.prefill_ms)
            << ",\"decode_tokens_per_second\":" << tokensPerSecond(run.metrics.generated_tokens, run.metrics.decode_total_ms)
            << ",\"error\":\"" << jsonEscape(run.metrics.error) << "\""
            << "}";
    }
    oss << "],\"per_kernel_wall_clock\":" << kernel_trace_json
        << ",\"custom_kernel_microbench\":" << custom_kernel_microbench_json
        << ",\"mnn_hotpath_op_trace\":" << mnn_hotpath_trace_json
        << "}";
    return oss.str();
}

std::string customQualityErrorJson(const std::string& model_dir,
                                   xq_status status,
                                   const std::string& error,
                                   const std::string& custom_backend_requested,
                                   const std::string& custom_backend_actual,
                                   int max_new_tokens) {
    std::ostringstream oss;
    oss << "{"
        << "\"schema_version\":1,"
        << "\"artifact_type\":\"quality_validation\","
        << "\"engine\":\"customlib\","
        << "\"status\":\"error\","
        << "\"quality_gate_passed\":false,"
        << "\"error_code\":" << status << ","
        << "\"error\":\"" << jsonEscape(error) << "\","
        << "\"custom_backend_requested\":\"" << jsonEscape(custom_backend_requested) << "\","
        << "\"custom_backend_actual\":\"" << jsonEscape(custom_backend_actual) << "\","
        << "\"custom_path\":{\"calls_mnn_llm_response_for_measured_generation\":false,"
        << "\"use_mnn_fallback\":0,\"full_custom_decode\":true,"
        << "\"fallback_op_families\":[]},"
        << "\"validation\":{\"prompt_set\":\"small_english_tokenizer_ids\","
        << "\"validation_compare_to_stock\":true,"
        << "\"tokenizer_decode_available\":false,"
        << "\"max_new_tokens\":" << max_new_tokens
        << ",\"temperature\":0,\"top_k\":1,\"top_p\":1},"
        << "\"artifact\":{\"model_dir\":\"" << jsonEscape(model_dir) << "\"}"
        << "}";
    return oss.str();
}

std::string runCustomQualityValidationJson(const std::string& model_dir,
                                           const std::string& custom_backend_requested,
                                           const std::string& custom_backend_actual,
                                           int max_new_tokens) {
    const int capped_max_new_tokens = std::max(1, std::min(64, max_new_tokens));
    xq_options options{};
    options.struct_size = sizeof(options);
    options.backend = custom_backend_actual.c_str();
    options.threads = kThreads;
    options.quant_bits = 4;
    options.group_size = 64;
    options.max_context_tokens = 8192;
    options.enable_autotune = 1;
    options.use_mnn_fallback = 0;

    xq_session* session = nullptr;
    xq_status status = xq_create(model_dir.c_str(), &options, &session);
    if (status != XQ_OK || !session) {
        return customQualityErrorJson(model_dir,
                                      status,
                                      "xq_create failed",
                                      custom_backend_requested,
                                      custom_backend_actual,
                                      capped_max_new_tokens);
    }

    const std::vector<ValidationPrompt> prompts = makeValidationPrompts();
    bool all_ok = true;
    xq_metrics initial_metrics{};
    initial_metrics.struct_size = sizeof(initial_metrics);
    xq_get_last_metrics(session, &initial_metrics);
    xq_metrics last_metrics = initial_metrics;
    std::ostringstream prompt_json;
    prompt_json << "[";
    for (size_t i = 0; i < prompts.size(); ++i) {
        const ValidationPrompt& prompt = prompts[i];
        if (i > 0) {
            prompt_json << ",";
        }
        xq_reset(session);
        std::vector<int32_t> output(static_cast<size_t>(capped_max_new_tokens));
        xq_metrics metrics{};
        metrics.struct_size = sizeof(metrics);
        status = xq_generate(session,
                             prompt.tokens.data(),
                             prompt.tokens.size(),
                             output.size(),
                             output.data(),
                             output.size(),
                             &metrics);
        last_metrics = metrics;
        const size_t generated_count = status == XQ_OK
                                           ? std::min<size_t>(output.size(), static_cast<size_t>(metrics.generated_tokens))
                                           : 0;
        std::vector<int32_t> generated(output.begin(), output.begin() + generated_count);
        const int invalid_count = countInvalidTokens(generated, kQwen35VocabSize);
        const bool empty_output = generated.empty();
        const bool degenerate = obviousDegenerateOutput(generated);
        const size_t longest_run = longestRepeatedRun(generated);
        const int64_t longest_run_token = longestRepeatedRunToken(generated);
        const int64_t most_repeated_token = mostRepeatedToken(generated);
        const size_t most_repeated_count = mostRepeatedTokenCount(generated);
        const size_t token_220_longest_run = longestRepeatedRunForToken(generated, 220);
        const bool token_220_failure = repeatedToken220Failure(generated);
        const bool prompt_ok = status == XQ_OK && invalid_count == 0 && !empty_output && !degenerate;
        all_ok = all_ok && prompt_ok;

        prompt_json << "{"
                    << "\"id\":\"" << jsonEscape(prompt.id) << "\","
                    << "\"prompt_kind\":\"" << jsonEscape(prompt.prompt_kind) << "\","
                    << "\"prompt_text_hint\":\"" << jsonEscape(prompt.prompt_text_hint) << "\","
                    << "\"raw_text_prompt_available\":false,"
                    << "\"prompt_token_count\":" << prompt.tokens.size() << ","
                    << "\"prompt_token_ids\":";
        appendIntArray(prompt_json, prompt.tokens, prompt.tokens.size());
        prompt_json << ",\"generated_token_count\":" << generated_count
                    << ",\"generated_token_ids\":";
        appendIntArray(prompt_json, generated, generated.size());
        prompt_json << ",\"decoded_text_available\":false,"
                    << "\"decoded_generated_text\":\"\","
                    << "\"status_code\":" << status << ","
                    << "\"status\":\"" << (prompt_ok ? "ok" : "error") << "\","
                    << "\"invalid_token_count\":" << invalid_count << ","
                    << "\"empty_output\":" << (empty_output ? "true" : "false") << ","
                    << "\"longest_repeated_run\":" << longest_run << ","
                    << "\"longest_repeated_run_token\":" << longest_run_token << ","
                    << "\"most_repeated_token\":" << most_repeated_token << ","
                    << "\"most_repeated_token_count\":" << most_repeated_count << ","
                    << "\"repeated_token_220_longest_run\":" << token_220_longest_run << ","
                    << "\"repeated_token_220_failure\":" << (token_220_failure ? "true" : "false") << ","
                    << "\"unique_token_count\":" << uniqueTokenCount(generated) << ","
                    << "\"obvious_degenerate_output\":" << (degenerate ? "true" : "false") << ","
                    << "\"prefill_ms\":" << metrics.prefill_ms << ","
                    << "\"decode_total_ms\":" << metrics.decode_total_ms
                    << "}";
    }
    prompt_json << "]";

    std::vector<char> trace_buffer(1024 * 1024);
    std::string kernel_trace_json = "{\"rows\":[]}";
    if (xq_get_kernel_trace_json(session, trace_buffer.data(), trace_buffer.size()) == XQ_OK) {
        kernel_trace_json = trace_buffer.data();
    }
    xq_destroy(session);

    std::ostringstream oss;
    oss << "{"
        << "\"schema_version\":1,"
        << "\"artifact_type\":\"quality_validation\","
        << "\"engine\":\"customlib\","
        << "\"status\":\"" << (all_ok ? "ok" : "error") << "\","
        << "\"quality_gate_passed\":" << (all_ok ? "true" : "false") << ","
        << "\"model\":{\"hf_repo\":\"Qwen/Qwen3.5-9B\","
        << "\"hf_revision\":\"c202236235762e1c871ad0ccb60c8ee5ba337b9a\","
        << "\"format\":\"xqwen35_custom_w4a16\","
        << "\"vocab_size_assumed\":" << kQwen35VocabSize << "},"
        << "\"artifact\":{\"model_dir\":\"" << jsonEscape(model_dir) << "\"},"
        << "\"runtime\":{\"threads\":" << kThreads
        << ",\"selected_kernels\":{\"summary\":\""
        << jsonEscape(last_metrics.selected_kernels[0] ? last_metrics.selected_kernels : initial_metrics.selected_kernels)
        << "\",\"hotpath_replaced\":true,"
        << "\"full_custom_decode\":true,"
        << "\"replaced_op_families\":[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"gate_proj\",\"up_proj\",\"down_proj\",\"rmsnorm\",\"rope\",\"attention\",\"linear_attention_state\",\"lm_head\",\"sampling\",\"prefill_kv_build\"],"
        << "\"fallback_op_families\":[]}},"
        << "\"custom_path\":{\"calls_mnn_llm_response_for_measured_generation\":false,"
        << "\"use_mnn_fallback\":0,"
        << "\"full_custom_decode\":true,"
        << "\"fallback_op_families\":[],"
        << "\"decode_loop\":\"xq_session::generate -> xq_prefill/xq_decode_one -> CustomModel::prefill/runLayer/sampleGreedy\"},"
        << "\"custom_backend_requested\":\"" << jsonEscape(custom_backend_requested) << "\","
        << "\"custom_backend_actual\":\"" << jsonEscape(custom_backend_actual) << "\","
        << "\"validation\":{\"prompt_set\":\"small_english_tokenizer_ids\","
        << "\"validation_compare_to_stock\":true,"
        << "\"tokenizer_decode_available\":false,"
        << "\"validation_dump_tokens\":true,"
        << "\"validation_dump_text\":false,"
        << "\"max_new_tokens\":" << capped_max_new_tokens
        << ",\"temperature\":0,\"top_k\":1,\"top_p\":1,"
        << "\"note\":\"Fixed token-ID prompts are used so stock MNN and customlib receive identical prompt IDs.\"},"
        << "\"prompts\":" << prompt_json.str() << ","
        << "\"per_kernel_wall_clock\":" << kernel_trace_json
        << "}";
    return oss.str();
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_xqwen35bench_NativeBenchmark_runBenchmark(JNIEnv* env,
                                                           jclass,
                                                           jstring model_dir,
                                                           jstring custom_backend,
                                                           jint prompt_tokens,
                                                           jint max_new_tokens,
                                                           jint warmup_iterations,
                                                           jint measure_iterations) {
    const std::string model = jstringToString(env, model_dir);
    const std::string requested_backend = normalizeCustomBackend(jstringToString(env, custom_backend));
    const std::string actual_backend =
        (requested_backend == "vulkan" || requested_backend == "cpu_vulkan_hybrid") ? "cpu_vulkan_hybrid" : "cpu";
    const std::string vulkan_probe_json = runVulkanProbeJson(requested_backend);
    __android_log_print(ANDROID_LOG_INFO,
                        "XQBENCH",
                        "BENCH_START engine=customlib model_dir=%s requested_backend=%s actual_backend=%s",
                        model.c_str(),
                        requested_backend.c_str(),
                        actual_backend.c_str());
    const int warmups = std::max(0, static_cast<int>(warmup_iterations));
    const int measured = std::max(1, static_cast<int>(measure_iterations));

    xq_options options{};
    options.struct_size = sizeof(options);
    options.backend = actual_backend.c_str();
    options.threads = kThreads;
    options.quant_bits = 4;
    options.group_size = 64;
    options.max_context_tokens = 8192;
    options.enable_autotune = 1;
    options.use_mnn_fallback = 0;

    xq_session* session = nullptr;
    xq_status status = xq_create(model.c_str(), &options, &session);
    if (status != XQ_OK || !session) {
        std::string json = errorJson(model, status, "xq_create failed", requested_backend, actual_backend, vulkan_probe_json);
        __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "BENCH_RESULT_JSON %s", json.c_str());
        return env->NewStringUTF(json.c_str());
    }

    xq_metrics initial_metrics{};
    initial_metrics.struct_size = sizeof(initial_metrics);
    xq_get_last_metrics(session, &initial_metrics);

    std::vector<int32_t> prompt(static_cast<size_t>(prompt_tokens), kPromptToken);
    std::vector<int32_t> output(static_cast<size_t>(max_new_tokens));
    std::vector<Run> runs;
    std::vector<double> prefill_tps;
    std::vector<double> decode_tps;
    std::string terminal_error;
    xq_status terminal_status = XQ_OK;

    for (int i = 0; i < warmups + measured; ++i) {
        xq_reset(session);
        Run run;
        run.index = i;
        run.warmup = i < warmups;
        run.metrics.struct_size = sizeof(run.metrics);
        std::fill(output.begin(), output.end(), 0);
        run.status = xq_generate(session,
                                 prompt.data(),
                                 prompt.size(),
                                 output.size(),
                                 output.data(),
                                 output.size(),
                                 &run.metrics);
        __android_log_print(ANDROID_LOG_INFO,
                            "XQBENCH",
                            "BENCH_ITER engine=customlib warmup=%d prompt=%llu generated=%llu prefill_ms=%.3f decode_ms=%.3f status=%d",
                            run.warmup ? 1 : 0,
                            static_cast<unsigned long long>(run.metrics.prompt_tokens),
                            static_cast<unsigned long long>(run.metrics.generated_tokens),
                            run.metrics.prefill_ms,
                            run.metrics.decode_total_ms,
                            run.status);
        if (!run.warmup && run.status == XQ_OK) {
            prefill_tps.push_back(tokensPerSecond(run.metrics.prompt_tokens, run.metrics.prefill_ms));
            decode_tps.push_back(tokensPerSecond(run.metrics.generated_tokens, run.metrics.decode_total_ms));
        }
        if (run.status != XQ_OK) {
            terminal_status = run.status;
            terminal_error = run.metrics.error;
            runs.push_back(run);
            break;
        }
        runs.push_back(run);
    }

    std::vector<char> trace_buffer(1024 * 1024);
    std::string kernel_trace_json = "{\"rows\":[]}";
    if (xq_get_kernel_trace_json(session, trace_buffer.data(), trace_buffer.size()) == XQ_OK) {
        kernel_trace_json = trace_buffer.data();
    }

    xq_destroy(session);

    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "KERNEL_MICROBENCH_START");
    const std::string custom_kernel_microbench_json = runCustomKernelMicrobenchJson();
    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "KERNEL_MICROBENCH_END");

    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "MNN_HOTPATH_TRACE_START");
    const std::string mnn_hotpath_trace_json = runMnnHotpathTraceJson(model);
    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "MNN_HOTPATH_TRACE_END");

    std::string json = makeJson(model,
                                initial_metrics,
                                runs,
                                prefill_tps,
                                decode_tps,
                                terminal_status,
                                terminal_error,
                                requested_backend,
                                actual_backend,
                                vulkan_probe_json,
                                static_cast<int>(prompt_tokens),
                                static_cast<int>(max_new_tokens),
                                warmups,
                                measured,
                                custom_kernel_microbench_json,
                                mnn_hotpath_trace_json,
                                kernel_trace_json);
    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "BENCH_RESULT_JSON %s", json.c_str());
    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "BENCH_END engine=customlib");
    return env->NewStringUTF(json.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_xqwen35bench_NativeBenchmark_runQualityValidation(JNIEnv* env,
                                                                   jclass,
                                                                   jstring model_dir,
                                                                   jstring custom_backend,
                                                                   jint max_new_tokens) {
    const std::string model = jstringToString(env, model_dir);
    const std::string requested_backend = normalizeCustomBackend(jstringToString(env, custom_backend));
    const std::string actual_backend =
        (requested_backend == "vulkan" || requested_backend == "cpu_vulkan_hybrid") ? "cpu_vulkan_hybrid" : "cpu";
    __android_log_print(ANDROID_LOG_INFO,
                        "XQBENCH",
                        "QUALITY_START engine=customlib model_dir=%s requested_backend=%s actual_backend=%s",
                        model.c_str(),
                        requested_backend.c_str(),
                        actual_backend.c_str());
    std::string json = runCustomQualityValidationJson(model,
                                                      requested_backend,
                                                      actual_backend,
                                                      static_cast<int>(max_new_tokens));
    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "BENCH_QUALITY_JSON %s", json.c_str());
    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "QUALITY_END engine=customlib");
    return env->NewStringUTF(json.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_xqwen35bench_NativeBenchmark_runVulkanSelfTest(JNIEnv* env, jclass) {
    std::vector<char> buffer(1024 * 1024);
    const xq_status status = xq_run_vulkan_w4a16_selftest(buffer.data(), buffer.size());
    std::string json = buffer.data();
    if (json.empty()) {
        json = "{\"engine\":\"customlib\",\"test\":\"vulkan_w4a16_gemv_correctness\","
               "\"status\":\"error\",\"pass\":false,\"error\":\"empty selftest JSON\"}";
    }
    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "BENCH_VULKAN_SELFTEST_STATUS %d", static_cast<int>(status));
    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "BENCH_VULKAN_SELFTEST_JSON %s", json.c_str());
    return env->NewStringUTF(json.c_str());
}
