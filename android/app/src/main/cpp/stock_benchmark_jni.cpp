#include <jni.h>

#include <android/log.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if defined(XQ_ENABLE_MNN)
#include "llm/llm.hpp"
#endif

namespace {

constexpr int kThreads = 8;
constexpr int kWarmupIterations = 1;
constexpr int kMeasureIterations = 5;
constexpr int kPromptToken = 16;

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

bool fileExists(const std::string& path) {
    std::ifstream fs(path, std::ios::binary);
    return fs.good();
}

void ensureDir(const std::string& path) {
    if (!path.empty()) {
        mkdir(path.c_str(), 0775);
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
                if (static_cast<unsigned char>(ch) < 0x20) {
                    oss << "\\u00";
                    const char* hex = "0123456789abcdef";
                    oss << hex[(ch >> 4) & 0x0f] << hex[ch & 0x0f];
                } else {
                    oss << ch;
                }
                break;
        }
    }
    return oss.str();
}

double tokensPerSecond(int tokens, int64_t usec) {
    if (tokens <= 0 || usec <= 0) {
        return 0.0;
    }
    return 1000000.0 * static_cast<double>(tokens) / static_cast<double>(usec);
}

double tpotMsFromTps(double tps) {
    return tps > 0.0 ? 1000.0 / tps : 0.0;
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
    stats.median = (values.size() % 2 == 0) ? (values[mid - 1] + values[mid]) * 0.5 : values[mid];
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

void appendTpotStatsFromTps(std::ostringstream& oss, const Stats& tps_stats) {
    oss << "{\"mean\":" << tpotMsFromTps(tps_stats.mean)
        << ",\"median\":" << tpotMsFromTps(tps_stats.median)
        << ",\"min\":" << tpotMsFromTps(tps_stats.max)
        << ",\"max\":" << tpotMsFromTps(tps_stats.min)
        << "}";
}

std::string errorJson(const std::string& model_dir,
                      const std::string& error,
                      int prompt_tokens,
                      int max_new_tokens) {
    const std::string config_path = model_dir + "/llm_config.json";
    std::ostringstream oss;
    oss << "{"
        << "\"schema_version\":2,"
        << "\"engine\":\"mnn_stock\","
        << "\"backend\":\"cpu\","
        << "\"status\":\"error\","
        << "\"error\":\"" << jsonEscape(error) << "\","
        << "\"model_dir\":\"" << jsonEscape(model_dir) << "\","
        << "\"required_files\":{\"llm_config_json\":" << (fileExists(config_path) ? "true" : "false")
        << ",\"llm_mnn\":" << (fileExists(model_dir + "/llm.mnn") ? "true" : "false")
        << ",\"llm_mnn_weight\":" << (fileExists(model_dir + "/llm.mnn.weight") ? "true" : "false")
        << ",\"tokenizer_mtok\":" << (fileExists(model_dir + "/tokenizer.mtok") ? "true" : "false")
        << ",\"embeddings_bf16_bin\":" << (fileExists(model_dir + "/embeddings_bf16.bin") ? "true" : "false") << "},"
        << "\"generation\":{\"prompt_tokens\":" << prompt_tokens
        << ",\"max_new_tokens\":" << max_new_tokens << "}"
        << "}";
    return oss.str();
}

#if defined(XQ_ENABLE_MNN)
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

bool terminalStatusOk(MNN::Transformer::LlmStatus status) {
    using MNN::Transformer::LlmStatus;
    return status == LlmStatus::NORMAL_FINISHED || status == LlmStatus::MAX_TOKENS_FINISHED;
}

bool setLlmConfig(MNN::Transformer::Llm* llm, const std::string& json, const char* label, std::string* error) {
    if (!llm->set_config(json)) {
        *error = std::string("set_config failed: ") + label + " payload=" + json;
        return false;
    }
    return true;
}

struct IterationResult {
    int index = 0;
    bool warmup = false;
    int prompt_len = 0;
    int generated_tokens = 0;
    int64_t prefill_us = 0;
    int64_t decode_us = 0;
    int64_t sample_us = 0;
    int64_t ttfa_us = 0;
    double wall_ms = 0.0;
    std::string status;
};

std::string runStockBenchmark(const std::string& model_dir, int prompt_tokens, int max_new_tokens) {
    const std::string config_path = model_dir + "/llm_config.json";
    if (model_dir.empty()) {
        return errorJson(model_dir, "model_dir is empty", prompt_tokens, max_new_tokens);
    }
    if (!fileExists(config_path)) {
        return errorJson(model_dir, "missing llm_config.json", prompt_tokens, max_new_tokens);
    }

    const std::string tmp_path = model_dir + "/tmp";
    ensureDir(tmp_path);

    std::unique_ptr<MNN::Transformer::Llm, void (*)(MNN::Transformer::Llm*)> llm(
        MNN::Transformer::Llm::createLLM(config_path),
        MNN::Transformer::Llm::destroy);
    if (!llm) {
        return errorJson(model_dir, "Llm::createLLM returned null", prompt_tokens, max_new_tokens);
    }

    std::string error;
    if (!setLlmConfig(llm.get(), "{\"tokenizer_file\":\"tokenizer.mtok\"}", "tokenizer_file", &error) ||
        !setLlmConfig(llm.get(), "{\"async\":false}", "async", &error) ||
        !setLlmConfig(llm.get(), "{\"reuse_kv\":false}", "reuse_kv", &error) ||
        !setLlmConfig(llm.get(), "{\"precision\":\"low\"}", "precision", &error) ||
        !setLlmConfig(llm.get(), "{\"memory\":\"low\"}", "memory", &error) ||
        !setLlmConfig(llm.get(), "{\"power\":\"normal\"}", "power", &error) ||
        !setLlmConfig(llm.get(), "{\"backend_type\":\"cpu\"}", "backend_type", &error) ||
        !setLlmConfig(llm.get(), "{\"thread_num\":8}", "thread_num", &error) ||
        !setLlmConfig(llm.get(), "{\"dynamic_option\":8}", "dynamic_option", &error) ||
        !setLlmConfig(llm.get(), "{\"attention_mode\":8}", "attention_mode", &error) ||
        !setLlmConfig(llm.get(), "{\"use_mmap\":true}", "use_mmap", &error) ||
        !setLlmConfig(llm.get(), "{\"tmp_path\":\"" + jsonEscape(tmp_path) + "\"}", "tmp_path", &error) ||
        !setLlmConfig(llm.get(), "{\"cpu_sme2_neon_division_ratio\":41}", "cpu_sme2_neon_division_ratio", &error) ||
        !setLlmConfig(llm.get(), "{\"cpu_sme_core_num\":2}", "cpu_sme_core_num", &error)) {
        return errorJson(model_dir, error, prompt_tokens, max_new_tokens);
    }

    const auto load_start = std::chrono::steady_clock::now();
    if (!llm->load()) {
        return errorJson(model_dir, "Llm::load failed", prompt_tokens, max_new_tokens);
    }
    const auto load_end = std::chrono::steady_clock::now();
    const double load_wall_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    const int64_t load_us = llm->getContext() ? llm->getContext()->load_us : 0;

    std::vector<IterationResult> iterations;
    std::vector<double> prefill_tps;
    std::vector<double> decode_tps;
    std::vector<double> wall_ms_values;
    std::string terminal_error;

    std::vector<int> prompt(static_cast<size_t>(prompt_tokens), kPromptToken);
    for (int i = 0; i < kWarmupIterations + kMeasureIterations; ++i) {
        llm->reset();
        IterationResult result;
        result.index = i;
        result.warmup = i < kWarmupIterations;

        const auto start = std::chrono::steady_clock::now();
        llm->response(prompt, nullptr, nullptr, max_new_tokens);
        const auto end = std::chrono::steady_clock::now();

        const MNN::Transformer::LlmContext* context = llm->getContext();
        result.wall_ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (context) {
            result.prompt_len = context->prompt_len;
            result.generated_tokens = context->gen_seq_len;
            result.prefill_us = context->prefill_us;
            result.decode_us = context->decode_us;
            result.sample_us = context->sample_us;
            result.ttfa_us = context->ttfa_us;
            result.status = statusName(context->status);
            if (!terminalStatusOk(context->status)) {
                terminal_error = std::string("generation ended with status ") + result.status;
            }
        } else {
            result.status = "missing_context";
            terminal_error = "generation returned without context";
        }

        if (!result.warmup) {
            prefill_tps.push_back(tokensPerSecond(result.prompt_len, result.prefill_us));
            decode_tps.push_back(tokensPerSecond(result.generated_tokens, result.decode_us));
            wall_ms_values.push_back(result.wall_ms);
        }
        iterations.push_back(result);

        __android_log_print(ANDROID_LOG_INFO,
                            "XQBENCH",
                            "BENCH_ITER engine=mnn_stock warmup=%d prompt=%d generated=%d prefill_us=%lld decode_us=%lld status=%s",
                            result.warmup ? 1 : 0,
                            result.prompt_len,
                            result.generated_tokens,
                            static_cast<long long>(result.prefill_us),
                            static_cast<long long>(result.decode_us),
                            result.status.c_str());
        if (!terminal_error.empty()) {
            break;
        }
    }

    const Stats prefill_stats = computeStats(prefill_tps);
    const Stats decode_stats = computeStats(decode_tps);
    const Stats wall_stats = computeStats(wall_ms_values);
    const bool ok = terminal_error.empty() && prefill_tps.size() == static_cast<size_t>(kMeasureIterations);

    std::ostringstream oss;
    oss << "{"
        << "\"schema_version\":2,"
        << "\"engine\":\"mnn_stock\","
        << "\"backend\":\"cpu\","
        << "\"status\":\"" << (ok ? "ok" : "error") << "\",";
    if (!ok) {
        oss << "\"error\":\"" << jsonEscape(terminal_error.empty() ? "not all measurement iterations completed" : terminal_error) << "\",";
    }
    oss << "\"model\":{\"hf_repo\":\"Qwen/Qwen3.5-9B\","
        << "\"hf_revision\":\"c202236235762e1c871ad0ccb60c8ee5ba337b9a\","
        << "\"format\":\"mnn_skip_weight\","
        << "\"quantization\":{\"bits\":4,\"group_size\":64,\"scheme\":\"w4a16_groupwise\"}},"
        << "\"artifact\":{\"model_dir\":\"" << jsonEscape(model_dir) << "\","
        << "\"config_path\":\"" << jsonEscape(config_path) << "\"},"
        << "\"runtime\":{\"threads\":" << kThreads
        << ",\"precision\":\"low\",\"memory\":\"low\",\"power\":\"normal\","
        << "\"use_mmap\":true,\"reuse_kv\":false,\"dynamic_option\":8,"
        << "\"attention_mode\":8,\"cpu_sme2_neon_division_ratio\":41,\"cpu_sme_core_num\":2},"
        << "\"generation\":{\"prompt_tokens_requested\":" << prompt_tokens
        << ",\"max_new_tokens\":" << max_new_tokens
        << ",\"prompt_token_id\":" << kPromptToken
        << ",\"temperature\":0,\"top_k\":1,\"top_p\":1,\"batch_size\":1},"
        << "\"iterations\":{\"warmup\":" << kWarmupIterations
        << ",\"measured\":" << prefill_tps.size() << ",\"target_measured\":" << kMeasureIterations << "},"
        << "\"timing\":{\"load_us\":" << load_us
        << ",\"load_wall_ms\":" << load_wall_ms
        << ",\"wall_ms\":";
    appendStats(oss, wall_stats);
    oss << "},\"tokens_per_second\":{\"prefill\":";
    appendStats(oss, prefill_stats);
    oss << ",\"decode\":";
    appendStats(oss, decode_stats);
    oss << "},\"time_per_output_token_ms\":{\"decode\":";
    appendTpotStatsFromTps(oss, decode_stats);
    oss << "},\"runs\":[";
    for (size_t i = 0; i < iterations.size(); ++i) {
        const IterationResult& run = iterations[i];
        if (i > 0) {
            oss << ",";
        }
        oss << "{\"index\":" << run.index
            << ",\"warmup\":" << (run.warmup ? "true" : "false")
            << ",\"status\":\"" << run.status << "\""
            << ",\"prompt_tokens\":" << run.prompt_len
            << ",\"generated_tokens\":" << run.generated_tokens
            << ",\"prefill_us\":" << run.prefill_us
            << ",\"decode_us\":" << run.decode_us
            << ",\"sample_us\":" << run.sample_us
            << ",\"ttfa_us\":" << run.ttfa_us
            << ",\"wall_ms\":" << run.wall_ms
            << ",\"prefill_tokens_per_second\":" << tokensPerSecond(run.prompt_len, run.prefill_us)
            << ",\"decode_tokens_per_second\":" << tokensPerSecond(run.generated_tokens, run.decode_us)
            << "}";
    }
    oss << "]}";
    return oss.str();
}
#endif

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_xqwen35stock_NativeStockBenchmark_runBenchmark(JNIEnv* env,
                                                                jclass,
                                                                jstring model_dir,
                                                                jint prompt_tokens,
                                                                jint max_new_tokens) {
    const std::string model = jstringToString(env, model_dir);
    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "BENCH_START engine=mnn_stock model_dir=%s", model.c_str());

#if defined(XQ_ENABLE_MNN)
    std::string json = runStockBenchmark(model, static_cast<int>(prompt_tokens), static_cast<int>(max_new_tokens));
#else
    std::string json = errorJson(model, "stock JNI was built without XQ_ENABLE_MNN", prompt_tokens, max_new_tokens);
#endif

    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "BENCH_RESULT_JSON %s", json.c_str());
    __android_log_print(ANDROID_LOG_INFO, "XQBENCH", "BENCH_END engine=mnn_stock");
    return env->NewStringUTF(json.c_str());
}
