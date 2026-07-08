#ifndef XQWEN35_H_
#define XQWEN35_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(XQWEN35_BUILD)
#define XQ_API __declspec(dllexport)
#elif defined(_WIN32)
#define XQ_API __declspec(dllimport)
#else
#define XQ_API __attribute__((visibility("default")))
#endif

typedef struct xq_session xq_session;

typedef enum xq_status {
    XQ_OK = 0,
    XQ_ERR_INVALID_ARGUMENT = -1,
    XQ_ERR_NOT_READY = -2,
    XQ_ERR_MODEL = -3,
    XQ_ERR_RUNTIME = -4,
    XQ_ERR_BUFFER_TOO_SMALL = -5
} xq_status;

typedef struct xq_options {
    size_t struct_size;
    const char* backend;
    int threads;
    int quant_bits;
    int group_size;
    int max_context_tokens;
    int enable_autotune;
    int use_mnn_fallback;
    const char* telemetry_path;
} xq_options;

typedef struct xq_metrics {
    size_t struct_size;
    uint64_t prompt_tokens;
    uint64_t generated_tokens;
    double load_ms;
    double prefill_ms;
    double first_token_ms;
    double decode_total_ms;
    double total_generate_ms;
    double prefill_tokens_per_second;
    double decode_tokens_per_second;
    uint64_t rss_kb;
    uint64_t pss_kb;
    char backend[64];
    char selected_kernels[512];
    char error[256];
} xq_metrics;

XQ_API xq_status xq_create(const char* model_dir, const xq_options* options, xq_session** out_session);
XQ_API void xq_destroy(xq_session* session);
XQ_API xq_status xq_prefill(xq_session* session, const int32_t* token_ids, size_t n_tokens, xq_metrics* out_metrics);
XQ_API xq_status xq_decode_one(xq_session* session, int32_t* out_token_id, xq_metrics* out_metrics);
XQ_API xq_status xq_generate(xq_session* session,
                             const int32_t* token_ids,
                             size_t n_tokens,
                             size_t max_new_tokens,
                             int32_t* output_buffer,
                             size_t output_capacity,
                             xq_metrics* out_metrics);
XQ_API xq_status xq_reset(xq_session* session);
XQ_API xq_status xq_get_last_metrics(xq_session* session, xq_metrics* out_metrics);
XQ_API xq_status xq_get_backend_info(xq_session* session, char* json_out, size_t json_capacity);
XQ_API xq_status xq_get_kernel_trace_json(xq_session* session, char* json_out, size_t json_capacity);
XQ_API xq_status xq_get_debug_json(xq_session* session, char* json_out, size_t json_capacity);
XQ_API xq_status xq_get_hidden_state(xq_session* session, float* out_values, size_t value_capacity, size_t* out_size);
XQ_API xq_status xq_run_vulkan_w4a16_selftest(char* json_out, size_t json_capacity);

#ifdef __cplusplus
}
#endif

#endif  // XQWEN35_H_
