#include "../kernels.hpp"

#include <sstream>

namespace xq {
namespace kernels {

std::string detectCpuFeatures() {
#if defined(__aarch64__)
    std::string features = "arm64,neon";
#if defined(__ARM_FEATURE_DOTPROD)
    features += ",dotprod";
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    features += ",i8mm";
#endif
    return features;
#elif defined(_M_ARM64)
    return "arm64,neon";
#else
    return "host-scalar";
#endif
}

std::string selectKernelSummary(int quant_bits) {
    std::ostringstream oss;
    oss << "linear=w" << quant_bits << "a16_";
    if (quant_bits == 4) {
        oss << "generated_tiled_neon_fused_dequant";
    } else if (quant_bits == 3) {
        oss << "neon_bitpack_ref_verified";
    } else if (quant_bits == 2) {
        oss << "neon_experimental_ref_verified";
    } else {
        oss << "unsupported";
    }
    oss << ";rmsnorm=fp32_ref;rope=fp32_ref;attention=gqa_decode_ref;delta=gated_delta_ref";
    return oss.str();
}

}  // namespace kernels
}  // namespace xq
