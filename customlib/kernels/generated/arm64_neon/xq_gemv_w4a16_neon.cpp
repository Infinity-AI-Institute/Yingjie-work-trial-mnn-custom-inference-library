#include "../../kernels.hpp"

#include <algorithm>
#include <cstddef>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace xq {
namespace kernels {
namespace {

int groupsPerRow(int cols, int group_size) {
    return (cols + group_size - 1) / group_size;
}

float dotGroupScalar(const uint8_t* packed, const float* x, int count, float zero) {
    float code_dot = 0.0f;
    float x_sum = 0.0f;
    for (int c = 0; c < count; c += 2) {
        const uint8_t byte = packed[c >> 1];
        const float x0 = x[c];
        const float code0 = static_cast<float>(byte & 0x0fu);
        code_dot += code0 * x0;
        x_sum += x0;
        if (c + 1 < count) {
            const float x1 = x[c + 1];
            const float code1 = static_cast<float>(byte >> 4);
            code_dot += code1 * x1;
            x_sum += x1;
        }
    }
    return code_dot - zero * x_sum;
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
float horizontalAdd(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    const float32x2_t lo = vget_low_f32(v);
    const float32x2_t hi = vget_high_f32(v);
    const float32x2_t sum2 = vadd_f32(lo, hi);
    return vget_lane_f32(sum2, 0) + vget_lane_f32(sum2, 1);
#endif
}

float dotGroupNeon(const uint8_t* packed, const float* x, int count, float zero) {
    float32x4_t code_dot0 = vdupq_n_f32(0.0f);
    float32x4_t code_dot1 = vdupq_n_f32(0.0f);
    float32x4_t x_sum0 = vdupq_n_f32(0.0f);
    float32x4_t x_sum1 = vdupq_n_f32(0.0f);
    int c = 0;
    for (; c + 15 < count; c += 16) {
        const uint8x8_t bytes = vld1_u8(packed + (c >> 1));
        const uint8x8_t lo = vand_u8(bytes, vdup_n_u8(0x0f));
        const uint8x8_t hi = vshr_n_u8(bytes, 4);
        const uint8x8x2_t interleaved = vzip_u8(lo, hi);

        const uint16x8_t codes01 = vmovl_u8(interleaved.val[0]);
        const uint16x8_t codes23 = vmovl_u8(interleaved.val[1]);
        const float32x4_t codes0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(codes01)));
        const float32x4_t codes1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(codes01)));
        const float32x4_t codes2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(codes23)));
        const float32x4_t codes3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(codes23)));

        const float32x4_t x0 = vld1q_f32(x + c);
        const float32x4_t x1 = vld1q_f32(x + c + 4);
        const float32x4_t x2 = vld1q_f32(x + c + 8);
        const float32x4_t x3 = vld1q_f32(x + c + 12);

        code_dot0 = vmlaq_f32(code_dot0, codes0, x0);
        code_dot1 = vmlaq_f32(code_dot1, codes1, x1);
        code_dot0 = vmlaq_f32(code_dot0, codes2, x2);
        code_dot1 = vmlaq_f32(code_dot1, codes3, x3);
        x_sum0 = vaddq_f32(x_sum0, x0);
        x_sum1 = vaddq_f32(x_sum1, x1);
        x_sum0 = vaddq_f32(x_sum0, x2);
        x_sum1 = vaddq_f32(x_sum1, x3);
    }
    const float code_dot = horizontalAdd(vaddq_f32(code_dot0, code_dot1));
    const float x_sum = horizontalAdd(vaddq_f32(x_sum0, x_sum1));
    return (code_dot - zero * x_sum) + dotGroupScalar(packed + (c >> 1), x + c, count - c, zero);
}
#endif

float dotGroupW4(const uint8_t* packed, const float* x, int count, float zero) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return dotGroupNeon(packed, x, count, zero);
#else
    return dotGroupScalar(packed, x, count, zero);
#endif
}

}  // namespace

void gemvW4A16Neon(const QuantizedMatrix& matrix, const float* x, float* y) {
    if (matrix.bits != 4 || matrix.rows <= 0 || matrix.cols <= 0 || matrix.group_size <= 0) {
        if (matrix.rows > 0) {
            std::fill(y, y + matrix.rows, 0.0f);
        }
        return;
    }

    const int gpr = groupsPerRow(matrix.cols, matrix.group_size);
    const size_t row_stride_bytes = (static_cast<size_t>(matrix.cols) + 1u) / 2u;
    constexpr int kRowTile = 4;
    for (int r0 = 0; r0 < matrix.rows; r0 += kRowTile) {
        const int r1 = std::min(matrix.rows, r0 + kRowTile);
        float acc[kRowTile] = {};
        for (int r = r0; r < r1; ++r) {
            acc[r - r0] = matrix.bias.empty() ? 0.0f : matrix.bias[static_cast<size_t>(r)];
        }
        for (int g = 0; g < gpr; ++g) {
            const int c0 = g * matrix.group_size;
            const int count = std::min(matrix.group_size, matrix.cols - c0);
            for (int r = r0; r < r1; ++r) {
                const size_t meta = static_cast<size_t>(r * gpr + g);
                const uint8_t* row_bytes = matrix.packed.data() + static_cast<size_t>(r) * row_stride_bytes + (c0 >> 1);
                acc[r - r0] += matrix.scales[meta] * dotGroupW4(row_bytes, x + c0, count, matrix.zeros[meta]);
            }
        }
        for (int r = r0; r < r1; ++r) {
            y[r] = acc[r - r0];
        }
    }
}

}  // namespace kernels
}  // namespace xq
