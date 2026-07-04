#ifndef XQWEN35_RUNTIME_SESSION_HPP_
#define XQWEN35_RUNTIME_SESSION_HPP_

#include "xqwen35.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if defined(XQ_ENABLE_MNN)
namespace MNN {
namespace Transformer {
class Llm;
}
}
#endif

namespace xq {

struct ModelManifest {
    std::string model_dir;
    std::string hf_repo = "Qwen/Qwen3.5-9B";
    std::string hf_revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a";
    int vocab_size = 248320;
    int hidden_size = 0;
    int layers = 0;
    int quant_bits = 4;
    int group_size = 64;
};

struct RuntimeOptions {
    std::string backend = "cpu";
    int threads = 4;
    int quant_bits = 4;
    int group_size = 64;
    int max_context_tokens = 8192;
    bool enable_autotune = true;
    bool use_mnn_fallback = true;
    std::string telemetry_path;
};

class Session {
public:
    Session(std::string model_dir, RuntimeOptions options);
    ~Session();

    xq_status load();
    xq_status prefill(const int32_t* token_ids, size_t n_tokens, xq_metrics* out_metrics);
    xq_status decodeOne(int32_t* out_token_id, xq_metrics* out_metrics);
    xq_status generate(const int32_t* token_ids,
                       size_t n_tokens,
                       size_t max_new_tokens,
                       int32_t* output_buffer,
                       size_t output_capacity,
                       xq_metrics* out_metrics);
    xq_status reset();
    xq_status getLastMetrics(xq_metrics* out_metrics) const;
    xq_status getBackendInfo(char* json_out, size_t json_capacity) const;

private:
    using Clock = std::chrono::steady_clock;

    void setError(const std::string& message);
    void publishMetrics(xq_metrics* out_metrics) const;
    void updateRates();
    std::string selectedKernelSummary() const;
    xq_status generateWithMnn(const int32_t* token_ids,
                              size_t n_tokens,
                              size_t max_new_tokens,
                              int32_t* output_buffer,
                              size_t output_capacity,
                              xq_metrics* out_metrics);

    std::string model_dir_;
    RuntimeOptions options_;
    ModelManifest manifest_;
    xq_metrics metrics_{};
    bool loaded_ = false;
    bool has_prefill_ = false;
    int32_t last_token_ = 0;
    size_t position_ = 0;
    std::vector<int32_t> prompt_cache_;
#if defined(XQ_ENABLE_MNN)
    MNN::Transformer::Llm* mnn_llm_ = nullptr;
#endif
};

RuntimeOptions normalizeOptions(const xq_options* options);
void copyMetrics(const xq_metrics& src, xq_metrics* dst);

}  // namespace xq

#endif  // XQWEN35_RUNTIME_SESSION_HPP_
