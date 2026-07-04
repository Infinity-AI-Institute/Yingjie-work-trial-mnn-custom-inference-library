#include <jni.h>

#include <android/log.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "kernels.hpp"
#include "xqwen35.h"

#if defined(XQ_ENABLE_MNN)
#include "llm/llm.hpp"
#include "MNN/Tensor.hpp"
#endif

namespace {

constexpr int kThreads = 8;
constexpr int kWarmupIterations = 1;
constexpr int kMeasureIterations = 5;
constexpr int kPromptToken = 16;
constexpr int kTracePromptTokens = 512;
constexpr int kTraceMaxNewTokens = 16;
constexpr int kKernelWarmupIterations = 1;
constexpr int kKernelMeasureIterations = 3;
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

void appendJsonStringField(std::ostringstream& oss, const char* key, const std::string& value) {
    oss << "\"" << key << "\":\"" << jsonEscape(value) << "\"";
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
                    "generated_tiled_neon_fused_dequant_gemv",
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
            << "\"implementation\":\"generated_tiled_neon_fused_dequant_gemv\","
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

std::string errorJson(const std::string& model_dir, xq_status status, const std::string& error) {
    std::ostringstream oss;
    oss << "{"
        << "\"schema_version\":2,"
        << "\"engine\":\"customlib\","
        << "\"backend\":\"mnn_fallback_cpu\","
        << "\"status\":\"error\","
        << "\"error_code\":" << status << ","
        << "\"error\":\"" << jsonEscape(error) << "\","
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
                     int prompt_tokens,
                     int max_new_tokens,
                     const std::string& custom_kernel_microbench_json,
                     const std::string& mnn_hotpath_trace_json,
                     const std::string& kernel_trace_json) {
    const bool ok = terminal_status == XQ_OK && prefill_tps.size() == static_cast<size_t>(kMeasureIterations);
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
        << "\"schema_version\":3,"
        << "\"engine\":\"customlib\","
        << "\"backend\":\"" << jsonEscape(runs.empty() ? initial_metrics.backend : runs.back().metrics.backend) << "\","
        << "\"status\":\"" << (ok ? "ok" : "error") << "\",";
    if (!ok) {
        oss << "\"error_code\":" << terminal_status << ","
            << "\"error\":\"" << jsonEscape(terminal_error.empty() ? "not all measurement iterations completed" : terminal_error)
            << "\",";
    }
    oss << "\"model\":{\"hf_repo\":\"Qwen/Qwen3.5-9B\","
        << "\"hf_revision\":\"c202236235762e1c871ad0ccb60c8ee5ba337b9a\","
        << "\"format\":\"mnn_skip_weight\","
        << "\"quantization\":{\"bits\":4,\"group_size\":64,\"scheme\":\"w4a16_groupwise\"}},"
        << "\"artifact\":{\"model_dir\":\"" << jsonEscape(model_dir) << "\"},"
        << "\"runtime\":{\"threads\":" << kThreads
        << ",\"precision\":\"low\",\"memory\":\"low\",\"power\":\"normal\","
        << "\"use_mmap\":true,\"reuse_kv\":false,"
        << "\"selected_kernels\":{\"summary\":\""
        << jsonEscape(runs.empty() ? initial_metrics.selected_kernels : runs.back().metrics.selected_kernels)
        << "\",\"hotpath_replaced\":true,"
        << "\"replaced_op_families\":[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"gate_proj\",\"up_proj\",\"down_proj\",\"rmsnorm\",\"rope\",\"linear_attn_qkv_z_out\"],"
        << "\"fallback_op_families\":[\"attention\",\"linear_attention_state\",\"lm_head\",\"sampling\",\"prefill_kv_build\"]}},"
        << "\"custom_path\":{\"calls_mnn_llm_response_for_measured_generation\":false,"
        << "\"decode_loop\":\"xq_session::generate -> xq_prefill/xq_decode_one -> CustomModel::runLayer\","
        << "\"weight_format\":\"xq4_groupwise_w4a16\"},"
        << "\"generation\":{\"prompt_tokens_requested\":" << prompt_tokens
        << ",\"max_new_tokens\":" << max_new_tokens
        << ",\"prompt_token_id\":" << kPromptToken
        << ",\"temperature\":0,\"top_k\":1,\"top_p\":1,\"batch_size\":1},"
        << "\"iterations\":{\"warmup\":" << kWarmupIterations
        << ",\"measured\":" << prefill_tps.size() << ",\"target_measured\":" << kMeasureIterations << "},"
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

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_xqwen35bench_NativeBenchmark_runBenchmark(JNIEnv* env,
                                                           jclass,
                                                           jstring model_dir,
                                                           jint prompt_tokens,
                                                           jint max_new_tokens) {
    const std::string model = jstringToString(env, model_dir);
    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "BENCH_START engine=customlib model_dir=%s", model.c_str());

    xq_options options{};
    options.struct_size = sizeof(options);
    options.backend = "cpu";
    options.threads = kThreads;
    options.quant_bits = 4;
    options.group_size = 64;
    options.max_context_tokens = 8192;
    options.enable_autotune = 1;
    options.use_mnn_fallback = 0;

    xq_session* session = nullptr;
    xq_status status = xq_create(model.c_str(), &options, &session);
    if (status != XQ_OK || !session) {
        std::string json = errorJson(model, status, "xq_create failed");
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

    for (int i = 0; i < kWarmupIterations + kMeasureIterations; ++i) {
        xq_reset(session);
        Run run;
        run.index = i;
        run.warmup = i < kWarmupIterations;
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
                                static_cast<int>(prompt_tokens),
                                static_cast<int>(max_new_tokens),
                                custom_kernel_microbench_json,
                                mnn_hotpath_trace_json,
                                kernel_trace_json);
    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "BENCH_RESULT_JSON %s", json.c_str());
    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "BENCH_END engine=customlib");
    return env->NewStringUTF(json.c_str());
}
