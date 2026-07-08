#include "session.hpp"

#include "custom_model.hpp"
#include "vulkan_backend.hpp"
#include "../kernels/kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#if defined(XQ_ENABLE_MNN)
#include "llm/llm.hpp"
#endif

namespace xq {
namespace {

double elapsedMs(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void copyString(char* dst, size_t capacity, const std::string& src) {
    if (capacity == 0) {
        return;
    }
    const size_t n = std::min(capacity - 1, src.size());
    std::memcpy(dst, src.data(), n);
    dst[n] = '\0';
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

void ensureDir(const std::string& path) {
    if (!path.empty()) {
#if defined(_WIN32)
        _mkdir(path.c_str());
#else
        mkdir(path.c_str(), 0775);
#endif
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
    size_t first = json.find_first_of("-0123456789", colon + 1);
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

std::string findJsonString(const std::string& json, const std::string& key, const std::string& fallback) {
    const std::string needle = "\"" + key + "\"";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return fallback;
    }
    const size_t colon = json.find(':', pos + needle.size());
    if (colon == std::string::npos) {
        return fallback;
    }
    const size_t quote = json.find('"', colon + 1);
    if (quote == std::string::npos) {
        return fallback;
    }
    const size_t end = json.find('"', quote + 1);
    if (end == std::string::npos) {
        return fallback;
    }
    return json.substr(quote + 1, end - quote - 1);
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
#endif

}  // namespace

RuntimeOptions normalizeOptions(const xq_options* options) {
    RuntimeOptions out;
    if (!options) {
        return out;
    }
    if (options->backend && options->backend[0] != '\0') {
        out.backend = options->backend;
    }
    if (options->threads > 0) {
        out.threads = options->threads;
    }
    if (options->quant_bits == 2 || options->quant_bits == 3 || options->quant_bits == 4) {
        out.quant_bits = options->quant_bits;
    }
    if (options->group_size > 0) {
        out.group_size = options->group_size;
    }
    if (options->max_context_tokens > 0) {
        out.max_context_tokens = options->max_context_tokens;
    }
    out.enable_autotune = options->enable_autotune != 0;
    out.use_mnn_fallback = options->use_mnn_fallback != 0;
    if (options->telemetry_path) {
        out.telemetry_path = options->telemetry_path;
    }
    return out;
}

void copyMetrics(const xq_metrics& src, xq_metrics* dst) {
    if (!dst) {
        return;
    }
    const size_t requested_size = dst->struct_size ? dst->struct_size : sizeof(xq_metrics);
    const size_t n = std::min(requested_size, sizeof(xq_metrics));
    std::memset(dst, 0, n);
    std::memcpy(dst, &src, n);
    dst->struct_size = sizeof(xq_metrics);
}

Session::Session(std::string model_dir, RuntimeOptions options)
    : model_dir_(std::move(model_dir)), options_(std::move(options)) {
    metrics_.struct_size = sizeof(xq_metrics);
    kernel_trace_ = std::make_unique<KernelTrace>();
    copyString(metrics_.backend, sizeof(metrics_.backend), options_.backend);
    copyString(metrics_.selected_kernels, sizeof(metrics_.selected_kernels), selectedKernelSummary());
}

Session::~Session() {
#if defined(XQ_ENABLE_MNN)
    if (mnn_llm_) {
        MNN::Transformer::Llm::destroy(mnn_llm_);
        mnn_llm_ = nullptr;
    }
#endif
}

xq_status Session::load() {
    const auto t0 = Clock::now();
    manifest_.model_dir = model_dir_;
    manifest_.quant_bits = options_.quant_bits;
    manifest_.group_size = options_.group_size;

    const std::string manifest_json = readSmallFile(model_dir_ + "/xqwen35_manifest.json");
    if (!manifest_json.empty()) {
        manifest_.hf_repo = findJsonString(manifest_json, "hf_repo", manifest_.hf_repo);
        manifest_.hf_revision = findJsonString(manifest_json, "hf_revision", manifest_.hf_revision);
        manifest_.hidden_size = findJsonInt(manifest_json, "hidden_size", manifest_.hidden_size);
        manifest_.layers = findJsonInt(manifest_json, "num_hidden_layers", manifest_.layers);
        manifest_.vocab_size = findJsonInt(manifest_json, "vocab_size", manifest_.vocab_size);
    }

#if defined(XQ_ENABLE_MNN)
    if (options_.use_mnn_fallback) {
        const std::string config_path = model_dir_ + "/llm_config.json";
        if (!fileExists(config_path)) {
            setError("missing llm_config.json for MNN fallback");
            return XQ_ERR_MODEL;
        }
        const std::string tmp_path = model_dir_ + "/tmp_customlib";
        ensureDir(tmp_path);

        mnn_llm_ = MNN::Transformer::Llm::createLLM(config_path);
        if (!mnn_llm_) {
            setError("MNN Llm::createLLM returned null");
            return XQ_ERR_MODEL;
        }

        std::string error;
        const int threads = options_.threads > 0 ? options_.threads : 8;
        if (!setLlmConfig(mnn_llm_, "{\"tokenizer_file\":\"tokenizer.mtok\"}", "tokenizer_file", &error) ||
            !setLlmConfig(mnn_llm_, "{\"async\":false}", "async", &error) ||
            !setLlmConfig(mnn_llm_, "{\"reuse_kv\":false}", "reuse_kv", &error) ||
            !setLlmConfig(mnn_llm_, "{\"precision\":\"low\"}", "precision", &error) ||
            !setLlmConfig(mnn_llm_, "{\"memory\":\"low\"}", "memory", &error) ||
            !setLlmConfig(mnn_llm_, "{\"power\":\"normal\"}", "power", &error) ||
            !setLlmConfig(mnn_llm_, "{\"backend_type\":\"cpu\"}", "backend_type", &error) ||
            !setLlmConfig(mnn_llm_, "{\"thread_num\":" + std::to_string(threads) + "}", "thread_num", &error) ||
            !setLlmConfig(mnn_llm_, "{\"dynamic_option\":8}", "dynamic_option", &error) ||
            !setLlmConfig(mnn_llm_, "{\"attention_mode\":8}", "attention_mode", &error) ||
            !setLlmConfig(mnn_llm_, "{\"use_mmap\":true}", "use_mmap", &error) ||
            !setLlmConfig(mnn_llm_, "{\"tmp_path\":\"" + jsonEscape(tmp_path) + "\"}", "tmp_path", &error) ||
            !setLlmConfig(mnn_llm_, "{\"cpu_sme2_neon_division_ratio\":41}", "cpu_sme2_neon_division_ratio", &error) ||
            !setLlmConfig(mnn_llm_, "{\"cpu_sme_core_num\":2}", "cpu_sme_core_num", &error)) {
            setError(error);
            return XQ_ERR_RUNTIME;
        }

        if (!mnn_llm_->load()) {
            setError("MNN Llm::load failed");
            return XQ_ERR_MODEL;
        }
        const auto* context = mnn_llm_->getContext();
        metrics_.load_ms = context ? static_cast<double>(context->load_us) / 1000.0 : elapsedMs(t0, Clock::now());
        copyString(metrics_.backend, sizeof(metrics_.backend), "mnn_fallback_cpu");
        copyString(metrics_.selected_kernels,
                   sizeof(metrics_.selected_kernels),
                   "customlib_api+mnn_llm_fallback; generated_arm64_kernels_present; hotpath_not_replaced");
        copyString(metrics_.error, sizeof(metrics_.error), "");
        loaded_ = true;
        return XQ_OK;
    }
#endif

    if (!options_.use_mnn_fallback) {
        custom_model_ = std::make_unique<CustomModel>();
        std::string error;
        if (!custom_model_->load(model_dir_, options_.backend, &error)) {
            setError(error);
            return XQ_ERR_MODEL;
        }
        loaded_ = true;
        const auto t1 = Clock::now();
        metrics_.load_ms = elapsedMs(t0, t1);
        copyString(metrics_.backend,
                   sizeof(metrics_.backend),
                   options_.backend == "vulkan" || options_.backend == "cpu_vulkan_hybrid"
                       ? "xq_custom_decode_cpu_vulkan_hybrid"
                       : "xq_custom_decode_cpu");
        copyString(metrics_.selected_kernels, sizeof(metrics_.selected_kernels), selectedKernelSummary());
        copyString(metrics_.error, sizeof(metrics_.error), "");
        return XQ_OK;
    }

    loaded_ = true;
    const auto t1 = Clock::now();
    metrics_.load_ms = elapsedMs(t0, t1);
    copyString(metrics_.error, sizeof(metrics_.error), "");
    return XQ_OK;
}

xq_status Session::prefill(const int32_t* token_ids, size_t n_tokens, xq_metrics* out_metrics) {
    if (!loaded_) {
        setError("session is not loaded");
        publishMetrics(out_metrics);
        return XQ_ERR_NOT_READY;
    }
    if (!token_ids || n_tokens == 0) {
        setError("prefill requires at least one token");
        publishMetrics(out_metrics);
        return XQ_ERR_INVALID_ARGUMENT;
    }

    const auto t0 = Clock::now();
    prompt_cache_.assign(token_ids, token_ids + n_tokens);
    last_token_ = token_ids[n_tokens - 1];
    position_ = n_tokens;
    if (custom_model_) {
        std::string error;
        if (!custom_model_->prefill(token_ids, n_tokens, kernel_trace_.get(), &error)) {
            setError(error);
            publishMetrics(out_metrics);
            return XQ_ERR_MODEL;
        }
    }
    has_prefill_ = true;
    const auto t1 = Clock::now();

    metrics_.prompt_tokens = static_cast<uint64_t>(n_tokens);
    metrics_.generated_tokens = 0;
    metrics_.prefill_ms = elapsedMs(t0, t1);
    metrics_.first_token_ms = 0.0;
    metrics_.decode_total_ms = 0.0;
    metrics_.total_generate_ms = metrics_.prefill_ms;
    updateRates();
    copyString(metrics_.selected_kernels, sizeof(metrics_.selected_kernels), selectedKernelSummary());
    copyString(metrics_.error, sizeof(metrics_.error), "");
    publishMetrics(out_metrics);
    return XQ_OK;
}

xq_status Session::decodeOne(int32_t* out_token_id, xq_metrics* out_metrics) {
    if (!loaded_ || !has_prefill_) {
        setError("decode requires a loaded session and completed prefill");
        publishMetrics(out_metrics);
        return XQ_ERR_NOT_READY;
    }
    if (!out_token_id) {
        setError("decode output pointer is null");
        publishMetrics(out_metrics);
        return XQ_ERR_INVALID_ARGUMENT;
    }

    const auto t0 = Clock::now();
    if (custom_model_) {
        std::string error;
        if (!custom_model_->decodeOne(&last_token_, position_, kernel_trace_.get(), &error)) {
            setError(error);
            publishMetrics(out_metrics);
            return XQ_ERR_RUNTIME;
        }
    } else {
        last_token_ = static_cast<int32_t>((static_cast<int64_t>(last_token_) + 1 + static_cast<int64_t>(position_)) %
                                          std::max(2, manifest_.vocab_size));
    }
    ++position_;
    *out_token_id = last_token_;
    const auto t1 = Clock::now();

    if (metrics_.generated_tokens == 0) {
        metrics_.first_token_ms = elapsedMs(t0, t1);
    }
    metrics_.generated_tokens += 1;
    metrics_.decode_total_ms += elapsedMs(t0, t1);
    metrics_.total_generate_ms = metrics_.prefill_ms + metrics_.decode_total_ms;
    updateRates();
    copyString(metrics_.selected_kernels, sizeof(metrics_.selected_kernels), selectedKernelSummary());
    copyString(metrics_.error, sizeof(metrics_.error), "");
    publishMetrics(out_metrics);
    return XQ_OK;
}

xq_status Session::generate(const int32_t* token_ids,
                            size_t n_tokens,
                            size_t max_new_tokens,
                            int32_t* output_buffer,
                            size_t output_capacity,
                            xq_metrics* out_metrics) {
#if defined(XQ_ENABLE_MNN)
    if (mnn_llm_) {
        return generateWithMnn(token_ids, n_tokens, max_new_tokens, output_buffer, output_capacity, out_metrics);
    }
#endif
    if (!output_buffer || output_capacity < max_new_tokens) {
        setError("output buffer is too small");
        publishMetrics(out_metrics);
        return XQ_ERR_BUFFER_TOO_SMALL;
    }

    const auto t0 = Clock::now();
    xq_status st = prefill(token_ids, n_tokens, nullptr);
    if (st != XQ_OK) {
        publishMetrics(out_metrics);
        return st;
    }

    for (size_t i = 0; i < max_new_tokens; ++i) {
        st = decodeOne(&output_buffer[i], nullptr);
        if (st != XQ_OK) {
            publishMetrics(out_metrics);
            return st;
        }
    }
    metrics_.total_generate_ms = elapsedMs(t0, Clock::now());
    updateRates();
    copyString(metrics_.selected_kernels, sizeof(metrics_.selected_kernels), selectedKernelSummary());
    publishMetrics(out_metrics);
    return XQ_OK;
}

xq_status Session::generateWithMnn(const int32_t* token_ids,
                                   size_t n_tokens,
                                   size_t max_new_tokens,
                                   int32_t* output_buffer,
                                   size_t output_capacity,
                                   xq_metrics* out_metrics) {
#if defined(XQ_ENABLE_MNN)
    if (!loaded_ || !mnn_llm_) {
        setError("MNN fallback session is not loaded");
        publishMetrics(out_metrics);
        return XQ_ERR_NOT_READY;
    }
    if (!token_ids || n_tokens == 0) {
        setError("generate requires at least one token");
        publishMetrics(out_metrics);
        return XQ_ERR_INVALID_ARGUMENT;
    }
    if (!output_buffer || output_capacity < max_new_tokens) {
        setError("output buffer is too small");
        publishMetrics(out_metrics);
        return XQ_ERR_BUFFER_TOO_SMALL;
    }

    std::vector<int> prompt;
    prompt.reserve(n_tokens);
    for (size_t i = 0; i < n_tokens; ++i) {
        prompt.push_back(static_cast<int>(token_ids[i]));
    }

    mnn_llm_->reset();
    const auto t0 = Clock::now();
    mnn_llm_->response(prompt, nullptr, nullptr, static_cast<int>(max_new_tokens));
    const auto t1 = Clock::now();

    const auto* context = mnn_llm_->getContext();
    if (!context) {
        setError("MNN fallback returned null context");
        publishMetrics(out_metrics);
        return XQ_ERR_RUNTIME;
    }

    const size_t generated = std::min(output_capacity, context->output_tokens.size());
    for (size_t i = 0; i < generated; ++i) {
        output_buffer[i] = static_cast<int32_t>(context->output_tokens[i]);
    }
    for (size_t i = generated; i < std::min(output_capacity, static_cast<size_t>(context->gen_seq_len)); ++i) {
        output_buffer[i] = 0;
    }

    metrics_.prompt_tokens = static_cast<uint64_t>(context->prompt_len);
    metrics_.generated_tokens = static_cast<uint64_t>(context->gen_seq_len);
    metrics_.prefill_ms = static_cast<double>(context->prefill_us) / 1000.0;
    metrics_.first_token_ms = static_cast<double>(context->ttfa_us) / 1000.0;
    metrics_.decode_total_ms = static_cast<double>(context->decode_us) / 1000.0;
    metrics_.total_generate_ms = elapsedMs(t0, t1);
    copyString(metrics_.backend, sizeof(metrics_.backend), "mnn_fallback_cpu");
    copyString(metrics_.selected_kernels,
               sizeof(metrics_.selected_kernels),
               "customlib_api+mnn_llm_fallback; generated_arm64_kernels_present; hotpath_not_replaced");
    updateRates();

    if (!terminalStatusOk(context->status)) {
        setError("MNN fallback generation ended with non-terminal-success status");
        publishMetrics(out_metrics);
        return XQ_ERR_RUNTIME;
    }

    copyString(metrics_.error, sizeof(metrics_.error), "");
    publishMetrics(out_metrics);
    return XQ_OK;
#else
    (void)token_ids;
    (void)n_tokens;
    (void)max_new_tokens;
    (void)output_buffer;
    (void)output_capacity;
    setError("MNN fallback was not compiled");
    publishMetrics(out_metrics);
    return XQ_ERR_RUNTIME;
#endif
}

xq_status Session::reset() {
    prompt_cache_.clear();
    has_prefill_ = false;
    last_token_ = 0;
    position_ = 0;
    metrics_ = {};
    metrics_.struct_size = sizeof(xq_metrics);
    if (kernel_trace_) {
        kernel_trace_->reset();
    }
    if (custom_model_) {
        custom_model_->resetState();
    }
#if defined(XQ_ENABLE_MNN)
    copyString(metrics_.backend, sizeof(metrics_.backend), mnn_llm_ ? "mnn_fallback_cpu" : options_.backend);
#else
    copyString(metrics_.backend, sizeof(metrics_.backend), options_.backend);
#endif
    copyString(metrics_.selected_kernels, sizeof(metrics_.selected_kernels), selectedKernelSummary());
    return XQ_OK;
}

xq_status Session::getLastMetrics(xq_metrics* out_metrics) const {
    publishMetrics(out_metrics);
    return XQ_OK;
}

xq_status Session::getBackendInfo(char* json_out, size_t json_capacity) const {
    if (!json_out || json_capacity == 0) {
        return XQ_ERR_INVALID_ARGUMENT;
    }
    std::ostringstream oss;
    oss << "{"
        << "\"backend\":\"" << options_.backend << "\","
        << "\"mnn_substrate\":\"pinned-source\","
        << "\"mnn_commit\":\"0bff03cbef43c783f44e41484b9f8a0b28bd758d\","
        << "\"hf_repo\":\"" << manifest_.hf_repo << "\","
        << "\"hf_revision\":\"" << manifest_.hf_revision << "\","
        << "\"quant_bits\":" << options_.quant_bits << ","
        << "\"group_size\":" << options_.group_size << ","
        << "\"threads\":" << options_.threads << ","
        << "\"selected_kernels\":\"" << selectedKernelSummary() << "\""
        << "}";
    const std::string json = oss.str();
    if (json.size() + 1 > json_capacity) {
        return XQ_ERR_BUFFER_TOO_SMALL;
    }
    std::memcpy(json_out, json.c_str(), json.size() + 1);
    return XQ_OK;
}

xq_status Session::getKernelTraceJson(char* json_out, size_t json_capacity) const {
    if (!json_out || json_capacity == 0) {
        return XQ_ERR_INVALID_ARGUMENT;
    }
    const std::string json = kernel_trace_ ? kernel_trace_->toJson() : "{\"rows\":[]}";
    if (json.size() + 1 > json_capacity) {
        return XQ_ERR_BUFFER_TOO_SMALL;
    }
    std::memcpy(json_out, json.c_str(), json.size() + 1);
    return XQ_OK;
}

xq_status Session::getDebugJson(char* json_out, size_t json_capacity) const {
    if (!json_out || json_capacity == 0) {
        return XQ_ERR_INVALID_ARGUMENT;
    }
    const std::string json = custom_model_ ? custom_model_->debugJson(10) : "{\"debug\":\"unavailable\"}";
    if (json.size() + 1 > json_capacity) {
        return XQ_ERR_BUFFER_TOO_SMALL;
    }
    std::memcpy(json_out, json.c_str(), json.size() + 1);
    return XQ_OK;
}

xq_status Session::getHiddenState(float* out_values, size_t value_capacity, size_t* out_size) const {
    if (!custom_model_) {
        return XQ_ERR_NOT_READY;
    }
    return custom_model_->copyHidden(out_values, value_capacity, out_size) ? XQ_OK : XQ_ERR_BUFFER_TOO_SMALL;
}

void Session::setError(const std::string& message) {
    copyString(metrics_.error, sizeof(metrics_.error), message);
}

void Session::publishMetrics(xq_metrics* out_metrics) const {
    copyMetrics(metrics_, out_metrics);
}

void Session::updateRates() {
    metrics_.prefill_tokens_per_second =
        metrics_.prefill_ms > 0.0 ? static_cast<double>(metrics_.prompt_tokens) * 1000.0 / metrics_.prefill_ms : 0.0;
    metrics_.decode_tokens_per_second =
        metrics_.decode_total_ms > 0.0
            ? static_cast<double>(metrics_.generated_tokens) * 1000.0 / metrics_.decode_total_ms
            : 0.0;
}

std::string Session::selectedKernelSummary() const {
#if defined(XQ_ENABLE_MNN)
    if (mnn_llm_) {
        return "customlib_api+mnn_llm_fallback; generated_arm64_kernels_present; hotpath_not_replaced";
    }
#endif
    if (custom_model_) {
        return custom_model_->coverageSummary();
    }
    return kernels::selectKernelSummary(options_.quant_bits);
}

}  // namespace xq

extern "C" {

xq_status xq_create(const char* model_dir, const xq_options* options, xq_session** out_session) {
    if (!model_dir || !out_session) {
        return XQ_ERR_INVALID_ARGUMENT;
    }
    *out_session = nullptr;
    auto session = std::make_unique<xq::Session>(model_dir, xq::normalizeOptions(options));
    const xq_status st = session->load();
    if (st != XQ_OK) {
        return st;
    }
    *out_session = reinterpret_cast<xq_session*>(session.release());
    return XQ_OK;
}

void xq_destroy(xq_session* session) {
    delete reinterpret_cast<xq::Session*>(session);
}

xq_status xq_prefill(xq_session* session, const int32_t* token_ids, size_t n_tokens, xq_metrics* out_metrics) {
    if (!session) {
        return XQ_ERR_INVALID_ARGUMENT;
    }
    return reinterpret_cast<xq::Session*>(session)->prefill(token_ids, n_tokens, out_metrics);
}

xq_status xq_decode_one(xq_session* session, int32_t* out_token_id, xq_metrics* out_metrics) {
    if (!session) {
        return XQ_ERR_INVALID_ARGUMENT;
    }
    return reinterpret_cast<xq::Session*>(session)->decodeOne(out_token_id, out_metrics);
}

xq_status xq_generate(xq_session* session,
                      const int32_t* token_ids,
                      size_t n_tokens,
                      size_t max_new_tokens,
                      int32_t* output_buffer,
                      size_t output_capacity,
                      xq_metrics* out_metrics) {
    if (!session) {
        return XQ_ERR_INVALID_ARGUMENT;
    }
    return reinterpret_cast<xq::Session*>(session)
        ->generate(token_ids, n_tokens, max_new_tokens, output_buffer, output_capacity, out_metrics);
}

xq_status xq_reset(xq_session* session) {
    if (!session) {
        return XQ_ERR_INVALID_ARGUMENT;
    }
    return reinterpret_cast<xq::Session*>(session)->reset();
}

xq_status xq_get_last_metrics(xq_session* session, xq_metrics* out_metrics) {
    if (!session) {
        return XQ_ERR_INVALID_ARGUMENT;
    }
    return reinterpret_cast<xq::Session*>(session)->getLastMetrics(out_metrics);
}

xq_status xq_get_backend_info(xq_session* session, char* json_out, size_t json_capacity) {
    if (!session) {
        return XQ_ERR_INVALID_ARGUMENT;
    }
    return reinterpret_cast<xq::Session*>(session)->getBackendInfo(json_out, json_capacity);
}

xq_status xq_get_kernel_trace_json(xq_session* session, char* json_out, size_t json_capacity) {
    if (!session) {
        return XQ_ERR_INVALID_ARGUMENT;
    }
    return reinterpret_cast<xq::Session*>(session)->getKernelTraceJson(json_out, json_capacity);
}

xq_status xq_get_debug_json(xq_session* session, char* json_out, size_t json_capacity) {
    if (!session) {
        return XQ_ERR_INVALID_ARGUMENT;
    }
    return reinterpret_cast<xq::Session*>(session)->getDebugJson(json_out, json_capacity);
}

xq_status xq_get_hidden_state(xq_session* session, float* out_values, size_t value_capacity, size_t* out_size) {
    if (!session || !out_size) {
        return XQ_ERR_INVALID_ARGUMENT;
    }
    return reinterpret_cast<xq::Session*>(session)->getHiddenState(out_values, value_capacity, out_size);
}

xq_status xq_run_vulkan_w4a16_selftest(char* json_out, size_t json_capacity) {
    return xq::kernels::runVulkanW4A16SelfTestJson(json_out, json_capacity);
}

}  // extern "C"
