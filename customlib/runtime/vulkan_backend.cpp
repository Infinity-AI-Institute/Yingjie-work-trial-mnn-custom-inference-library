#include "vulkan_backend.hpp"

#include "../kernels/kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
#include "../kernels/generated/vulkan/w4a16_gemv_spv.inc"
#include <vulkan/vulkan.h>
#endif

namespace xq {
namespace kernels {
namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
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

void copyJson(char* dst, size_t capacity, const std::string& json) {
    if (!dst || capacity == 0) {
        return;
    }
    const size_t n = std::min(capacity - 1, json.size());
    std::memcpy(dst, json.data(), n);
    dst[n] = '\0';
}

QuantizedMatrix makeDeterministicMatrix(int rows, int cols, int group_size) {
    std::vector<float> dense(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const float a = std::sin(static_cast<float>((r * 17 + c * 13) % 997) * 0.013f);
            const float b = std::cos(static_cast<float>((r * 7 + c * 19) % 769) * 0.017f);
            dense[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] =
                0.20f * a + 0.05f * b;
        }
    }
    return quantizeLowBit(dense.data(), rows, cols, 4, group_size);
}

std::vector<float> makeDeterministicInput(int cols) {
    std::vector<float> x(static_cast<size_t>(cols));
    for (int c = 0; c < cols; ++c) {
        x[static_cast<size_t>(c)] =
            0.15f * std::sin(static_cast<float>(c % 1021) * 0.011f) +
            0.03f * std::cos(static_cast<float>(c % 577) * 0.019f);
    }
    return x;
}

struct DiffStats {
    double max_abs = 0.0;
    double mean_abs = 0.0;
};

DiffStats compareVectors(const std::vector<float>& expected, const std::vector<float>& got) {
    DiffStats stats;
    const size_t n = std::min(expected.size(), got.size());
    if (n == 0) {
        return stats;
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::abs(static_cast<double>(expected[i]) - static_cast<double>(got[i]));
        stats.max_abs = std::max(stats.max_abs, d);
        sum += d;
    }
    stats.mean_abs = sum / static_cast<double>(n);
    return stats;
}

}  // namespace

#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)

const char* vkResultName(VkResult r) {
    switch (r) {
        case VK_SUCCESS:
            return "VK_SUCCESS";
        case VK_NOT_READY:
            return "VK_NOT_READY";
        case VK_TIMEOUT:
            return "VK_TIMEOUT";
        case VK_EVENT_SET:
            return "VK_EVENT_SET";
        case VK_EVENT_RESET:
            return "VK_EVENT_RESET";
        case VK_INCOMPLETE:
            return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY:
            return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:
            return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST:
            return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED:
            return "VK_ERROR_MEMORY_MAP_FAILED";
        default:
            return "VK_ERROR_UNKNOWN";
    }
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize bytes = 0;
};

struct VulkanBackend::Impl {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::string device_name;
    bool initialized = false;

    void destroyBuffer(Buffer* b) {
        if (b->buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, b->buffer, nullptr);
            b->buffer = VK_NULL_HANDLE;
        }
        if (b->memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, b->memory, nullptr);
            b->memory = VK_NULL_HANDLE;
        }
        b->bytes = 0;
    }

    uint32_t findMemoryType(uint32_t type_bits, VkMemoryPropertyFlags flags, std::string* error) const {
        VkPhysicalDeviceMemoryProperties props{};
        vkGetPhysicalDeviceMemoryProperties(physical_device, &props);
        for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
            if ((type_bits & (1u << i)) != 0u &&
                (props.memoryTypes[i].propertyFlags & flags) == flags) {
                return i;
            }
        }
        *error = "no compatible Vulkan memory type";
        return UINT32_MAX;
    }

    bool createHostBuffer(VkDeviceSize bytes, Buffer* out, std::string* error) {
        if (bytes == 0) {
            bytes = 4;
        }
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = bytes;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult vr = vkCreateBuffer(device, &buffer_info, nullptr, &out->buffer);
        if (vr != VK_SUCCESS) {
            *error = std::string("vkCreateBuffer failed: ") + vkResultName(vr);
            return false;
        }
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, out->buffer, &req);
        const uint32_t type = findMemoryType(
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            error);
        if (type == UINT32_MAX) {
            destroyBuffer(out);
            return false;
        }
        VkMemoryAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = type;
        vr = vkAllocateMemory(device, &alloc, nullptr, &out->memory);
        if (vr != VK_SUCCESS) {
            *error = std::string("vkAllocateMemory failed: ") + vkResultName(vr);
            destroyBuffer(out);
            return false;
        }
        vr = vkBindBufferMemory(device, out->buffer, out->memory, 0);
        if (vr != VK_SUCCESS) {
            *error = std::string("vkBindBufferMemory failed: ") + vkResultName(vr);
            destroyBuffer(out);
            return false;
        }
        out->bytes = bytes;
        return true;
    }

    bool upload(const Buffer& b, const void* src, size_t bytes, std::string* error) const {
        void* mapped = nullptr;
        VkResult vr = vkMapMemory(device, b.memory, 0, b.bytes, 0, &mapped);
        if (vr != VK_SUCCESS) {
            *error = std::string("vkMapMemory(upload) failed: ") + vkResultName(vr);
            return false;
        }
        std::memset(mapped, 0, static_cast<size_t>(b.bytes));
        if (src && bytes > 0) {
            std::memcpy(mapped, src, std::min(static_cast<size_t>(b.bytes), bytes));
        }
        vkUnmapMemory(device, b.memory);
        return true;
    }

    bool download(const Buffer& b, void* dst, size_t bytes, std::string* error) const {
        void* mapped = nullptr;
        VkResult vr = vkMapMemory(device, b.memory, 0, b.bytes, 0, &mapped);
        if (vr != VK_SUCCESS) {
            *error = std::string("vkMapMemory(download) failed: ") + vkResultName(vr);
            return false;
        }
        std::memcpy(dst, mapped, std::min(static_cast<size_t>(b.bytes), bytes));
        vkUnmapMemory(device, b.memory);
        return true;
    }
};

struct GemvPushConstants {
    uint32_t rows;
    uint32_t cols;
    uint32_t group_size;
    uint32_t groups_per_row;
    uint32_t row_stride_bytes;
    uint32_t affine_asymmetric;
    uint32_t has_bias;
    uint32_t pad0;
};

#else

struct VulkanBackend::Impl {
    std::string device_name;
    bool initialized = false;
};

#endif

VulkanBackend::VulkanBackend() : impl_(new Impl()) {}

VulkanBackend::~VulkanBackend() {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (impl_->device != VK_NULL_HANDLE) {
        if (impl_->pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(impl_->device, impl_->pipeline, nullptr);
        }
        if (impl_->pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(impl_->device, impl_->pipeline_layout, nullptr);
        }
        if (impl_->descriptor_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(impl_->device, impl_->descriptor_layout, nullptr);
        }
        if (impl_->command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(impl_->device, impl_->command_pool, nullptr);
        }
        vkDestroyDevice(impl_->device, nullptr);
    }
    if (impl_->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(impl_->instance, nullptr);
    }
#endif
    delete impl_;
}

bool VulkanBackend::initialize(std::string* error) {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (impl_->initialized) {
        return true;
    }
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "xqwen35_custom_vulkan";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.pEngineName = "xq_customlib";
    app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    VkResult vr = vkCreateInstance(&instance_info, nullptr, &impl_->instance);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkCreateInstance failed: ") + vkResultName(vr);
        return false;
    }

    uint32_t physical_count = 0;
    vkEnumeratePhysicalDevices(impl_->instance, &physical_count, nullptr);
    if (physical_count == 0) {
        *error = "vkEnumeratePhysicalDevices returned zero devices";
        return false;
    }
    std::vector<VkPhysicalDevice> physical_devices(physical_count);
    vkEnumeratePhysicalDevices(impl_->instance, &physical_count, physical_devices.data());
    for (VkPhysicalDevice pd : physical_devices) {
        uint32_t q_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &q_count, nullptr);
        std::vector<VkQueueFamilyProperties> q_props(q_count);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &q_count, q_props.data());
        for (uint32_t i = 0; i < q_count; ++i) {
            if ((q_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u) {
                impl_->physical_device = pd;
                impl_->queue_family = i;
                break;
            }
        }
        if (impl_->physical_device != VK_NULL_HANDLE) {
            break;
        }
    }
    if (impl_->physical_device == VK_NULL_HANDLE) {
        *error = "no Vulkan compute queue family found";
        return false;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(impl_->physical_device, &props);
    impl_->device_name = props.deviceName;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = impl_->queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    vr = vkCreateDevice(impl_->physical_device, &device_info, nullptr, &impl_->device);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkCreateDevice failed: ") + vkResultName(vr);
        return false;
    }
    vkGetDeviceQueue(impl_->device, impl_->queue_family, 0, &impl_->queue);

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = impl_->queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vr = vkCreateCommandPool(impl_->device, &pool_info, nullptr, &impl_->command_pool);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkCreateCommandPool failed: ") + vkResultName(vr);
        return false;
    }

    VkDescriptorSetLayoutBinding bindings[6]{};
    for (uint32_t i = 0; i < 6; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 6;
    layout_info.pBindings = bindings;
    vr = vkCreateDescriptorSetLayout(impl_->device, &layout_info, nullptr, &impl_->descriptor_layout);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkCreateDescriptorSetLayout failed: ") + vkResultName(vr);
        return false;
    }

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(GemvPushConstants);
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &impl_->descriptor_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
    vr = vkCreatePipelineLayout(impl_->device, &pipeline_layout_info, nullptr, &impl_->pipeline_layout);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkCreatePipelineLayout failed: ") + vkResultName(vr);
        return false;
    }

    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = vulkan_shader::kW4A16GemvSpvSize;
    shader_info.pCode = vulkan_shader::kW4A16GemvSpv;
    VkShaderModule shader = VK_NULL_HANDLE;
    vr = vkCreateShaderModule(impl_->device, &shader_info, nullptr, &shader);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkCreateShaderModule failed: ") + vkResultName(vr);
        return false;
    }
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage;
    pipeline_info.layout = impl_->pipeline_layout;
    vr = vkCreateComputePipelines(impl_->device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &impl_->pipeline);
    vkDestroyShaderModule(impl_->device, shader, nullptr);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkCreateComputePipelines failed: ") + vkResultName(vr);
        return false;
    }
    impl_->initialized = true;
    return true;
#else
    if (error) {
        *error = "custom Vulkan backend is only compiled for Android builds";
    }
    return false;
#endif
}

bool VulkanBackend::isInitialized() const {
    return impl_->initialized;
}

const std::string& VulkanBackend::deviceName() const {
    return impl_->device_name;
}

bool VulkanBackend::gemvW4A16(const QuantizedMatrix& matrix,
                              const float* x,
                              float* y,
                              VulkanGemvTiming* timing,
                              std::string* error) {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (!initialize(error)) {
        return false;
    }
    if (matrix.bits != 4 || matrix.rows <= 0 || matrix.cols <= 0 || matrix.group_size <= 0) {
        *error = "Vulkan W4A16 GEMV requires a non-empty 4-bit matrix";
        return false;
    }
    const auto total0 = Clock::now();
    const int gpr = (matrix.cols + matrix.group_size - 1) / matrix.group_size;
    const size_t row_stride_bytes = (static_cast<size_t>(matrix.cols) + 1u) / 2u;

    std::vector<uint32_t> packed_words((matrix.packed.size() + 3u) / 4u, 0);
    std::memcpy(packed_words.data(), matrix.packed.data(), matrix.packed.size());
    std::vector<float> zero_bias(1, 0.0f);
    const std::vector<float>& bias = matrix.bias.empty() ? zero_bias : matrix.bias;
    std::vector<float> y_zero(static_cast<size_t>(matrix.rows), 0.0f);

    Buffer packed_b, scales_b, zeros_b, x_b, bias_b, y_b;
    auto cleanup = [&]() {
        impl_->destroyBuffer(&packed_b);
        impl_->destroyBuffer(&scales_b);
        impl_->destroyBuffer(&zeros_b);
        impl_->destroyBuffer(&x_b);
        impl_->destroyBuffer(&bias_b);
        impl_->destroyBuffer(&y_b);
    };
    if (!impl_->createHostBuffer(static_cast<VkDeviceSize>(packed_words.size() * sizeof(uint32_t)), &packed_b, error) ||
        !impl_->createHostBuffer(static_cast<VkDeviceSize>(matrix.scales.size() * sizeof(float)), &scales_b, error) ||
        !impl_->createHostBuffer(static_cast<VkDeviceSize>(matrix.zeros.size() * sizeof(float)), &zeros_b, error) ||
        !impl_->createHostBuffer(static_cast<VkDeviceSize>(static_cast<size_t>(matrix.cols) * sizeof(float)), &x_b, error) ||
        !impl_->createHostBuffer(static_cast<VkDeviceSize>(bias.size() * sizeof(float)), &bias_b, error) ||
        !impl_->createHostBuffer(static_cast<VkDeviceSize>(static_cast<size_t>(matrix.rows) * sizeof(float)), &y_b, error)) {
        cleanup();
        return false;
    }

    const auto upload0 = Clock::now();
    if (!impl_->upload(packed_b, packed_words.data(), packed_words.size() * sizeof(uint32_t), error) ||
        !impl_->upload(scales_b, matrix.scales.data(), matrix.scales.size() * sizeof(float), error) ||
        !impl_->upload(zeros_b, matrix.zeros.data(), matrix.zeros.size() * sizeof(float), error) ||
        !impl_->upload(x_b, x, static_cast<size_t>(matrix.cols) * sizeof(float), error) ||
        !impl_->upload(bias_b, bias.data(), bias.size() * sizeof(float), error) ||
        !impl_->upload(y_b, y_zero.data(), y_zero.size() * sizeof(float), error)) {
        cleanup();
        return false;
    }
    const auto upload1 = Clock::now();

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 6;
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkResult vr = vkCreateDescriptorPool(impl_->device, &pool_info, nullptr, &descriptor_pool);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkCreateDescriptorPool failed: ") + vkResultName(vr);
        cleanup();
        return false;
    }
    VkDescriptorSetAllocateInfo set_info{};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &impl_->descriptor_layout;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    vr = vkAllocateDescriptorSets(impl_->device, &set_info, &descriptor_set);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkAllocateDescriptorSets failed: ") + vkResultName(vr);
        vkDestroyDescriptorPool(impl_->device, descriptor_pool, nullptr);
        cleanup();
        return false;
    }
    Buffer* bufs[6] = {&packed_b, &scales_b, &zeros_b, &x_b, &bias_b, &y_b};
    VkDescriptorBufferInfo infos[6]{};
    VkWriteDescriptorSet writes[6]{};
    for (uint32_t i = 0; i < 6; ++i) {
        infos[i].buffer = bufs[i]->buffer;
        infos[i].offset = 0;
        infos[i].range = bufs[i]->bytes;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(impl_->device, 6, writes, 0, nullptr);

    VkCommandBufferAllocateInfo cmd_alloc{};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = impl_->command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vr = vkAllocateCommandBuffers(impl_->device, &cmd_alloc, &cmd);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkAllocateCommandBuffers failed: ") + vkResultName(vr);
        vkDestroyDescriptorPool(impl_->device, descriptor_pool, nullptr);
        cleanup();
        return false;
    }

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vr = vkBeginCommandBuffer(cmd, &begin);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkBeginCommandBuffer failed: ") + vkResultName(vr);
        vkFreeCommandBuffers(impl_->device, impl_->command_pool, 1, &cmd);
        vkDestroyDescriptorPool(impl_->device, descriptor_pool, nullptr);
        cleanup();
        return false;
    }
    GemvPushConstants pc{};
    pc.rows = static_cast<uint32_t>(matrix.rows);
    pc.cols = static_cast<uint32_t>(matrix.cols);
    pc.group_size = static_cast<uint32_t>(matrix.group_size);
    pc.groups_per_row = static_cast<uint32_t>(gpr);
    pc.row_stride_bytes = static_cast<uint32_t>(row_stride_bytes);
    pc.affine_asymmetric = matrix.affine_asymmetric ? 1u : 0u;
    pc.has_bias = matrix.bias.empty() ? 0u : 1u;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->pipeline);
    vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
    vkCmdPushConstants(cmd, impl_->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, static_cast<uint32_t>((matrix.rows + 63) / 64), 1, 1);
    vr = vkEndCommandBuffer(cmd);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkEndCommandBuffer failed: ") + vkResultName(vr);
        vkFreeCommandBuffers(impl_->device, impl_->command_pool, 1, &cmd);
        vkDestroyDescriptorPool(impl_->device, descriptor_pool, nullptr);
        cleanup();
        return false;
    }

    const auto dispatch0 = Clock::now();
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vr = vkQueueSubmit(impl_->queue, 1, &submit, VK_NULL_HANDLE);
    if (vr == VK_SUCCESS) {
        vr = vkQueueWaitIdle(impl_->queue);
    }
    const auto dispatch1 = Clock::now();
    if (vr != VK_SUCCESS) {
        *error = std::string("Vulkan GEMV dispatch failed: ") + vkResultName(vr);
        vkFreeCommandBuffers(impl_->device, impl_->command_pool, 1, &cmd);
        vkDestroyDescriptorPool(impl_->device, descriptor_pool, nullptr);
        cleanup();
        return false;
    }

    const auto download0 = Clock::now();
    if (!impl_->download(y_b, y, static_cast<size_t>(matrix.rows) * sizeof(float), error)) {
        vkFreeCommandBuffers(impl_->device, impl_->command_pool, 1, &cmd);
        vkDestroyDescriptorPool(impl_->device, descriptor_pool, nullptr);
        cleanup();
        return false;
    }
    const auto download1 = Clock::now();
    if (timing) {
        timing->upload_ms = elapsedMs(upload0, upload1);
        timing->dispatch_ms = elapsedMs(dispatch0, dispatch1);
        timing->download_ms = elapsedMs(download0, download1);
        timing->total_ms = elapsedMs(total0, Clock::now());
    }
    vkFreeCommandBuffers(impl_->device, impl_->command_pool, 1, &cmd);
    vkDestroyDescriptorPool(impl_->device, descriptor_pool, nullptr);
    cleanup();
    return true;
#else
    (void)matrix;
    (void)x;
    (void)y;
    (void)timing;
    if (error) {
        *error = "custom Vulkan GEMV is only compiled for Android builds";
    }
    return false;
#endif
}

xq_status runVulkanW4A16SelfTestJson(char* json_out, size_t json_capacity) {
    std::ostringstream json;
    VulkanBackend backend;
    std::string error;
    const auto start = Clock::now();
    const bool init_ok = backend.initialize(&error);
    json << "{\"engine\":\"customlib\",\"test\":\"vulkan_w4a16_gemv_correctness\","
         << "\"custom_backend_requested\":\"vulkan\","
         << "\"custom_backend_actual\":\"" << (init_ok ? "vulkan" : "unavailable") << "\","
         << "\"vulkan_runtime_initialized\":" << (init_ok ? "true" : "false") << ","
         << "\"vulkan_compute_queue_found\":" << (init_ok ? "true" : "false") << ","
         << "\"vulkan_generation_kernels_used\":false,"
         << "\"vulkan_device_name\":\"" << jsonEscape(backend.deviceName()) << "\"";
    if (!init_ok) {
        json << ",\"status\":\"error\",\"pass\":false,\"error\":\"" << jsonEscape(error) << "\"}";
        copyJson(json_out, json_capacity, json.str());
        return XQ_ERR_RUNTIME;
    }

    struct Shape {
        int rows;
        int cols;
        const char* name;
        double tolerance;
    };
    const Shape shapes[] = {
        {256, 256, "small_correctness", 1.0e-3},
        {4096, 4096, "qwen_4096x4096", 2.0e-2},
    };
    bool all_pass = true;
    json << ",\"results\":[";
    bool first = true;
    for (const Shape& shape : shapes) {
        QuantizedMatrix q = makeDeterministicMatrix(shape.rows, shape.cols, 64);
        std::vector<float> x = makeDeterministicInput(shape.cols);
        std::vector<float> cpu(static_cast<size_t>(shape.rows), 0.0f);
        std::vector<float> vk(static_cast<size_t>(shape.rows), 0.0f);
        const auto cpu0 = Clock::now();
        gemvW4A16Neon(q, x.data(), cpu.data());
        const double cpu_ms = elapsedMs(cpu0, Clock::now());
        VulkanGemvTiming timing;
        std::string kernel_error;
        const bool kernel_ok = backend.gemvW4A16(q, x.data(), vk.data(), &timing, &kernel_error);
        DiffStats diff = kernel_ok ? compareVectors(cpu, vk) : DiffStats{};
        const bool pass = kernel_ok && diff.max_abs <= shape.tolerance && std::isfinite(diff.max_abs);
        all_pass = all_pass && pass;
        if (!first) {
            json << ",";
        }
        first = false;
        json << "{\"name\":\"" << shape.name << "\",\"shape\":\"" << shape.rows << "x" << shape.cols
             << "\",\"backend\":\"vulkan\",\"rows\":" << shape.rows << ",\"cols\":" << shape.cols
             << ",\"group_size\":64,\"cpu_reference_ms\":" << cpu_ms
             << ",\"vulkan_upload_ms\":" << timing.upload_ms
             << ",\"vulkan_dispatch_ms\":" << timing.dispatch_ms
             << ",\"vulkan_download_ms\":" << timing.download_ms
             << ",\"vulkan_total_ms\":" << timing.total_ms
             << ",\"max_abs_error\":" << diff.max_abs
             << ",\"mean_abs_error\":" << diff.mean_abs
             << ",\"tolerance\":" << shape.tolerance
             << ",\"pass\":" << (pass ? "true" : "false");
        if (!kernel_ok) {
            json << ",\"error\":\"" << jsonEscape(kernel_error) << "\"";
        }
        json << "}";
    }
    json << "],\"wall_clock_ms\":" << elapsedMs(start, Clock::now())
         << ",\"status\":\"" << (all_pass ? "ok" : "error") << "\",\"pass\":" << (all_pass ? "true" : "false")
         << "}";
    copyJson(json_out, json_capacity, json.str());
    return all_pass ? XQ_OK : XQ_ERR_RUNTIME;
}

}  // namespace kernels
}  // namespace xq
