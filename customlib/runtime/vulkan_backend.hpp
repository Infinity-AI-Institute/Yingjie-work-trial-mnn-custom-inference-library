#ifndef XQWEN35_RUNTIME_VULKAN_BACKEND_HPP_
#define XQWEN35_RUNTIME_VULKAN_BACKEND_HPP_

#include "xqwen35.h"

#include <cstddef>
#include <string>

namespace xq {
namespace kernels {

struct QuantizedMatrix;

struct VulkanGemvTiming {
    double upload_ms = 0.0;
    double dispatch_ms = 0.0;
    double download_ms = 0.0;
    double total_ms = 0.0;
};

class VulkanBackend {
public:
    VulkanBackend();
    ~VulkanBackend();

    bool initialize(std::string* error);
    bool isInitialized() const;
    const std::string& deviceName() const;

    bool gemvW4A16(const QuantizedMatrix& matrix,
                   const float* x,
                   float* y,
                   VulkanGemvTiming* timing,
                   std::string* error);

private:
    struct Impl;
    Impl* impl_;
};

xq_status runVulkanW4A16SelfTestJson(char* json_out, size_t json_capacity);

}  // namespace kernels
}  // namespace xq

#endif  // XQWEN35_RUNTIME_VULKAN_BACKEND_HPP_
