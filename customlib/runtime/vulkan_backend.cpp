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
#include "../kernels/generated/vulkan/vector_ops_spv.inc"
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

struct VectorPushConstants {
    uint32_t op;
    uint32_t n;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t q_heads;
    uint32_t k_heads;
    uint32_t rotary_dim;
    uint32_t position;
    float eps;
    float theta;
    uint32_t reserved0;
    uint32_t reserved1;
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
    VkDescriptorSetLayout vector_descriptor_layout = VK_NULL_HANDLE;
    VkPipelineLayout vector_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline vector_pipeline = VK_NULL_HANDLE;
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

    bool runVector(const VectorPushConstants& pc,
                   const void* a_data,
                   size_t a_bytes,
                   const void* b_data,
                   size_t b_bytes,
                   const void* c_data,
                   size_t c_bytes,
                   const void* d_init,
                   size_t d_bytes,
                   void* d_out,
                   void* c_out,
                   VulkanGemvTiming* timing,
                   uint32_t dispatch_x,
                   uint32_t dispatch_y,
                   std::string* error) {
        const auto total0 = Clock::now();
        float zero = 0.0f;
        Buffer bufs[4];
        auto cleanup = [&]() {
            for (Buffer& b : bufs) {
                destroyBuffer(&b);
            }
        };
        const size_t sizes[4] = {
            std::max<size_t>(a_bytes, sizeof(float)),
            std::max<size_t>(b_bytes, sizeof(float)),
            std::max<size_t>(c_bytes, sizeof(float)),
            std::max<size_t>(d_bytes, sizeof(float)),
        };
        for (int i = 0; i < 4; ++i) {
            if (!createHostBuffer(static_cast<VkDeviceSize>(sizes[i]), &bufs[i], error)) {
                cleanup();
                return false;
            }
        }
        const auto upload0 = Clock::now();
        if (!upload(bufs[0], a_data ? a_data : &zero, a_data ? a_bytes : sizeof(float), error) ||
            !upload(bufs[1], b_data ? b_data : &zero, b_data ? b_bytes : sizeof(float), error) ||
            !upload(bufs[2], c_data ? c_data : &zero, c_data ? c_bytes : sizeof(float), error) ||
            !upload(bufs[3], d_init ? d_init : &zero, d_init ? d_bytes : sizeof(float), error)) {
            cleanup();
            return false;
        }
        const auto upload1 = Clock::now();

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 4;
        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkResult vr = vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool);
        if (vr != VK_SUCCESS) {
            *error = std::string("vkCreateDescriptorPool(vector) failed: ") + vkResultName(vr);
            cleanup();
            return false;
        }

        VkDescriptorSetAllocateInfo set_info{};
        set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        set_info.descriptorPool = descriptor_pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &vector_descriptor_layout;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        vr = vkAllocateDescriptorSets(device, &set_info, &descriptor_set);
        if (vr != VK_SUCCESS) {
            *error = std::string("vkAllocateDescriptorSets(vector) failed: ") + vkResultName(vr);
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            cleanup();
            return false;
        }

        VkDescriptorBufferInfo infos[4]{};
        VkWriteDescriptorSet writes[4]{};
        for (uint32_t i = 0; i < 4; ++i) {
            infos[i].buffer = bufs[i].buffer;
            infos[i].offset = 0;
            infos[i].range = bufs[i].bytes;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);

        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = command_pool;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vr = vkAllocateCommandBuffers(device, &cmd_alloc, &cmd);
        if (vr != VK_SUCCESS) {
            *error = std::string("vkAllocateCommandBuffers(vector) failed: ") + vkResultName(vr);
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            cleanup();
            return false;
        }

        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vr = vkBeginCommandBuffer(cmd, &begin);
        if (vr != VK_SUCCESS) {
            *error = std::string("vkBeginCommandBuffer(vector) failed: ") + vkResultName(vr);
            vkFreeCommandBuffers(device, command_pool, 1, &cmd);
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            cleanup();
            return false;
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vector_pipeline);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vector_pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
        vkCmdPushConstants(cmd, vector_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, std::max(1u, dispatch_x), std::max(1u, dispatch_y), 1);
        vr = vkEndCommandBuffer(cmd);
        if (vr != VK_SUCCESS) {
            *error = std::string("vkEndCommandBuffer(vector) failed: ") + vkResultName(vr);
            vkFreeCommandBuffers(device, command_pool, 1, &cmd);
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            cleanup();
            return false;
        }

        const auto dispatch0 = Clock::now();
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        vr = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
        if (vr == VK_SUCCESS) {
            vr = vkQueueWaitIdle(queue);
        }
        const auto dispatch1 = Clock::now();
        if (vr != VK_SUCCESS) {
            *error = std::string("Vulkan vector dispatch failed: ") + vkResultName(vr);
            vkFreeCommandBuffers(device, command_pool, 1, &cmd);
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            cleanup();
            return false;
        }

        const auto download0 = Clock::now();
        if (c_out && !download(bufs[2], c_out, c_bytes, error)) {
            vkFreeCommandBuffers(device, command_pool, 1, &cmd);
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            cleanup();
            return false;
        }
        if (d_out && !download(bufs[3], d_out, d_bytes, error)) {
            vkFreeCommandBuffers(device, command_pool, 1, &cmd);
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
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
        vkFreeCommandBuffers(device, command_pool, 1, &cmd);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        cleanup();
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
        if (impl_->vector_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(impl_->device, impl_->vector_pipeline, nullptr);
        }
        if (impl_->vector_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(impl_->device, impl_->vector_pipeline_layout, nullptr);
        }
        if (impl_->vector_descriptor_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(impl_->device, impl_->vector_descriptor_layout, nullptr);
        }
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

    VkDescriptorSetLayoutBinding vector_bindings[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        vector_bindings[i].binding = i;
        vector_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        vector_bindings[i].descriptorCount = 1;
        vector_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo vector_layout_info{};
    vector_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    vector_layout_info.bindingCount = 4;
    vector_layout_info.pBindings = vector_bindings;
    vr = vkCreateDescriptorSetLayout(impl_->device, &vector_layout_info, nullptr, &impl_->vector_descriptor_layout);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkCreateDescriptorSetLayout(vector) failed: ") + vkResultName(vr);
        return false;
    }

    VkPushConstantRange vector_push_range{};
    vector_push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    vector_push_range.offset = 0;
    vector_push_range.size = sizeof(VectorPushConstants);
    VkPipelineLayoutCreateInfo vector_pipeline_layout_info{};
    vector_pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    vector_pipeline_layout_info.setLayoutCount = 1;
    vector_pipeline_layout_info.pSetLayouts = &impl_->vector_descriptor_layout;
    vector_pipeline_layout_info.pushConstantRangeCount = 1;
    vector_pipeline_layout_info.pPushConstantRanges = &vector_push_range;
    vr = vkCreatePipelineLayout(impl_->device, &vector_pipeline_layout_info, nullptr, &impl_->vector_pipeline_layout);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkCreatePipelineLayout(vector) failed: ") + vkResultName(vr);
        return false;
    }

    VkShaderModuleCreateInfo vector_shader_info{};
    vector_shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vector_shader_info.codeSize = vulkan_shader::kVectorOpsSpvSize;
    vector_shader_info.pCode = vulkan_shader::kVectorOpsSpv;
    VkShaderModule vector_shader = VK_NULL_HANDLE;
    vr = vkCreateShaderModule(impl_->device, &vector_shader_info, nullptr, &vector_shader);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkCreateShaderModule(vector) failed: ") + vkResultName(vr);
        return false;
    }
    VkPipelineShaderStageCreateInfo vector_stage{};
    vector_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vector_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    vector_stage.module = vector_shader;
    vector_stage.pName = "main";
    VkComputePipelineCreateInfo vector_pipeline_info{};
    vector_pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    vector_pipeline_info.stage = vector_stage;
    vector_pipeline_info.layout = impl_->vector_pipeline_layout;
    vr = vkCreateComputePipelines(impl_->device, VK_NULL_HANDLE, 1, &vector_pipeline_info, nullptr, &impl_->vector_pipeline);
    vkDestroyShaderModule(impl_->device, vector_shader, nullptr);
    if (vr != VK_SUCCESS) {
        *error = std::string("vkCreateComputePipelines(vector) failed: ") + vkResultName(vr);
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

bool VulkanBackend::rmsNorm(const float* x,
                            const float* weight,
                            int n,
                            float eps,
                            float* y,
                            VulkanGemvTiming* timing,
                            std::string* error) {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (!initialize(error) || !x || !weight || !y || n <= 0) {
        if (error && n <= 0) {
            *error = "invalid rmsNorm arguments";
        }
        return false;
    }
    VectorPushConstants pc{};
    pc.op = 0;
    pc.n = static_cast<uint32_t>(n);
    pc.heads = 1;
    pc.head_dim = static_cast<uint32_t>(n);
    pc.eps = eps;
    return impl_->runVector(pc,
                            x,
                            static_cast<size_t>(n) * sizeof(float),
                            weight,
                            static_cast<size_t>(n) * sizeof(float),
                            nullptr,
                            0,
                            nullptr,
                            static_cast<size_t>(n) * sizeof(float),
                            y,
                            nullptr,
                            timing,
                            1,
                            1,
                            error);
#else
    (void)x;
    (void)weight;
    (void)n;
    (void)eps;
    (void)y;
    (void)timing;
    if (error) {
        *error = "custom Vulkan vector ops are only compiled for Android builds";
    }
    return false;
#endif
}

bool VulkanBackend::headRmsNorm(float* x,
                                const float* weight,
                                int heads,
                                int head_dim,
                                float eps,
                                VulkanGemvTiming* timing,
                                std::string* error) {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (!initialize(error) || !x || !weight || heads <= 0 || head_dim <= 0) {
        if (error && (heads <= 0 || head_dim <= 0)) {
            *error = "invalid headRmsNorm arguments";
        }
        return false;
    }
    const int n = heads * head_dim;
    std::vector<float> out(static_cast<size_t>(n), 0.0f);
    VectorPushConstants pc{};
    pc.op = 0;
    pc.n = static_cast<uint32_t>(n);
    pc.heads = static_cast<uint32_t>(heads);
    pc.head_dim = static_cast<uint32_t>(head_dim);
    pc.eps = eps;
    const bool ok = impl_->runVector(pc,
                                     x,
                                     static_cast<size_t>(n) * sizeof(float),
                                     weight,
                                     static_cast<size_t>(head_dim) * sizeof(float),
                                     nullptr,
                                     0,
                                     out.data(),
                                     out.size() * sizeof(float),
                                     out.data(),
                                     nullptr,
                                     timing,
                                     static_cast<uint32_t>(heads),
                                     1,
                                     error);
    if (ok) {
        std::memcpy(x, out.data(), out.size() * sizeof(float));
    }
    return ok;
#else
    (void)x;
    (void)weight;
    (void)heads;
    (void)head_dim;
    (void)eps;
    (void)timing;
    if (error) {
        *error = "custom Vulkan vector ops are only compiled for Android builds";
    }
    return false;
#endif
}

bool VulkanBackend::l2NormalizeHeads(float* x, int heads, int head_dim, VulkanGemvTiming* timing, std::string* error) {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (!initialize(error) || !x || heads <= 0 || head_dim <= 0) {
        if (error && (heads <= 0 || head_dim <= 0)) {
            *error = "invalid l2NormalizeHeads arguments";
        }
        return false;
    }
    const int n = heads * head_dim;
    std::vector<float> out(static_cast<size_t>(n), 0.0f);
    VectorPushConstants pc{};
    pc.op = 1;
    pc.n = static_cast<uint32_t>(n);
    pc.heads = static_cast<uint32_t>(heads);
    pc.head_dim = static_cast<uint32_t>(head_dim);
    pc.eps = 1.0e-6f;
    const bool ok = impl_->runVector(pc,
                                     x,
                                     static_cast<size_t>(n) * sizeof(float),
                                     nullptr,
                                     0,
                                     nullptr,
                                     0,
                                     out.data(),
                                     out.size() * sizeof(float),
                                     out.data(),
                                     nullptr,
                                     timing,
                                     static_cast<uint32_t>(heads),
                                     1,
                                     error);
    if (ok) {
        std::memcpy(x, out.data(), out.size() * sizeof(float));
    }
    return ok;
#else
    (void)x;
    (void)heads;
    (void)head_dim;
    (void)timing;
    if (error) {
        *error = "custom Vulkan vector ops are only compiled for Android builds";
    }
    return false;
#endif
}

bool VulkanBackend::gatedRmsNorm(const float* x,
                                 const float* weight,
                                 const float* gate,
                                 int heads,
                                 int head_dim,
                                 float eps,
                                 float* y,
                                 VulkanGemvTiming* timing,
                                 std::string* error) {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (!initialize(error) || !x || !weight || !gate || !y || heads <= 0 || head_dim <= 0) {
        if (error && (heads <= 0 || head_dim <= 0)) {
            *error = "invalid gatedRmsNorm arguments";
        }
        return false;
    }
    const int n = heads * head_dim;
    VectorPushConstants pc{};
    pc.op = 2;
    pc.n = static_cast<uint32_t>(n);
    pc.heads = static_cast<uint32_t>(heads);
    pc.head_dim = static_cast<uint32_t>(head_dim);
    pc.eps = eps;
    return impl_->runVector(pc,
                            x,
                            static_cast<size_t>(n) * sizeof(float),
                            weight,
                            static_cast<size_t>(head_dim) * sizeof(float),
                            gate,
                            static_cast<size_t>(n) * sizeof(float),
                            nullptr,
                            static_cast<size_t>(n) * sizeof(float),
                            y,
                            nullptr,
                            timing,
                            static_cast<uint32_t>(heads),
                            1,
                            error);
#else
    (void)x;
    (void)weight;
    (void)gate;
    (void)heads;
    (void)head_dim;
    (void)eps;
    (void)y;
    (void)timing;
    if (error) {
        *error = "custom Vulkan vector ops are only compiled for Android builds";
    }
    return false;
#endif
}

bool VulkanBackend::rope(float* q,
                         int q_heads,
                         float* k,
                         int k_heads,
                         int head_dim,
                         int rotary_dim,
                         int position,
                         float theta,
                         VulkanGemvTiming* timing,
                         std::string* error) {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (!initialize(error) || !q || !k || q_heads <= 0 || k_heads <= 0 || head_dim <= 0 || rotary_dim <= 0) {
        if (error && (q_heads <= 0 || k_heads <= 0 || head_dim <= 0 || rotary_dim <= 0)) {
            *error = "invalid rope arguments";
        }
        return false;
    }
    const int q_n = q_heads * head_dim;
    const int k_n = k_heads * head_dim;
    std::vector<float> out_q(static_cast<size_t>(q_n), 0.0f);
    std::vector<float> out_k(static_cast<size_t>(k_n), 0.0f);
    VectorPushConstants pc{};
    pc.op = 3;
    pc.n = static_cast<uint32_t>(q_n + k_n);
    pc.q_heads = static_cast<uint32_t>(q_heads);
    pc.k_heads = static_cast<uint32_t>(k_heads);
    pc.head_dim = static_cast<uint32_t>(head_dim);
    pc.rotary_dim = static_cast<uint32_t>(rotary_dim);
    pc.position = static_cast<uint32_t>(std::max(0, position));
    pc.theta = theta;
    const uint32_t groups = static_cast<uint32_t>((q_n + k_n + 255) / 256);
    const bool ok = impl_->runVector(pc,
                                     q,
                                     static_cast<size_t>(q_n) * sizeof(float),
                                     k,
                                     static_cast<size_t>(k_n) * sizeof(float),
                                     out_q.data(),
                                     out_q.size() * sizeof(float),
                                     out_k.data(),
                                     out_k.size() * sizeof(float),
                                     out_k.data(),
                                     out_q.data(),
                                     timing,
                                     groups,
                                     1,
                                     error);
    if (ok) {
        std::memcpy(q, out_q.data(), out_q.size() * sizeof(float));
        std::memcpy(k, out_k.data(), out_k.size() * sizeof(float));
    }
    return ok;
#else
    (void)q;
    (void)q_heads;
    (void)k;
    (void)k_heads;
    (void)head_dim;
    (void)rotary_dim;
    (void)position;
    (void)theta;
    (void)timing;
    if (error) {
        *error = "custom Vulkan vector ops are only compiled for Android builds";
    }
    return false;
#endif
}

bool VulkanBackend::siluGateMul(const float* gate,
                                const float* up,
                                int n,
                                float* y,
                                VulkanGemvTiming* timing,
                                std::string* error) {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (!initialize(error) || !gate || !up || !y || n <= 0) {
        if (error && n <= 0) {
            *error = "invalid siluGateMul arguments";
        }
        return false;
    }
    VectorPushConstants pc{};
    pc.op = 4;
    pc.n = static_cast<uint32_t>(n);
    return impl_->runVector(pc,
                            gate,
                            static_cast<size_t>(n) * sizeof(float),
                            up,
                            static_cast<size_t>(n) * sizeof(float),
                            nullptr,
                            0,
                            nullptr,
                            static_cast<size_t>(n) * sizeof(float),
                            y,
                            nullptr,
                            timing,
                            static_cast<uint32_t>((n + 255) / 256),
                            1,
                            error);
#else
    (void)gate;
    (void)up;
    (void)n;
    (void)y;
    (void)timing;
    if (error) {
        *error = "custom Vulkan vector ops are only compiled for Android builds";
    }
    return false;
#endif
}

bool VulkanBackend::residualAdd(float* hidden,
                                const float* delta,
                                int n,
                                VulkanGemvTiming* timing,
                                std::string* error) {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (!initialize(error) || !hidden || !delta || n <= 0) {
        if (error && n <= 0) {
            *error = "invalid residualAdd arguments";
        }
        return false;
    }
    std::vector<float> out(static_cast<size_t>(n), 0.0f);
    VectorPushConstants pc{};
    pc.op = 5;
    pc.n = static_cast<uint32_t>(n);
    const bool ok = impl_->runVector(pc,
                                     hidden,
                                     static_cast<size_t>(n) * sizeof(float),
                                     delta,
                                     static_cast<size_t>(n) * sizeof(float),
                                     nullptr,
                                     0,
                                     out.data(),
                                     out.size() * sizeof(float),
                                     out.data(),
                                     nullptr,
                                     timing,
                                     static_cast<uint32_t>((n + 255) / 256),
                                     1,
                                     error);
    if (ok) {
        std::memcpy(hidden, out.data(), out.size() * sizeof(float));
    }
    return ok;
#else
    (void)hidden;
    (void)delta;
    (void)n;
    (void)timing;
    if (error) {
        *error = "custom Vulkan vector ops are only compiled for Android builds";
    }
    return false;
#endif
}

bool VulkanBackend::attentionOutputGate(float* hidden,
                                        const float* gate,
                                        int n,
                                        VulkanGemvTiming* timing,
                                        std::string* error) {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (!initialize(error) || !hidden || !gate || n <= 0) {
        if (error && n <= 0) {
            *error = "invalid attentionOutputGate arguments";
        }
        return false;
    }
    std::vector<float> out(static_cast<size_t>(n), 0.0f);
    VectorPushConstants pc{};
    pc.op = 6;
    pc.n = static_cast<uint32_t>(n);
    const bool ok = impl_->runVector(pc,
                                     hidden,
                                     static_cast<size_t>(n) * sizeof(float),
                                     gate,
                                     static_cast<size_t>(n) * sizeof(float),
                                     nullptr,
                                     0,
                                     out.data(),
                                     out.size() * sizeof(float),
                                     out.data(),
                                     nullptr,
                                     timing,
                                     static_cast<uint32_t>((n + 255) / 256),
                                     1,
                                     error);
    if (ok) {
        std::memcpy(hidden, out.data(), out.size() * sizeof(float));
    }
    return ok;
#else
    (void)hidden;
    (void)gate;
    (void)n;
    (void)timing;
    if (error) {
        *error = "custom Vulkan vector ops are only compiled for Android builds";
    }
    return false;
#endif
}

bool VulkanBackend::argmax(const float* logits,
                           int n,
                           int* out_index,
                           float* out_value,
                           VulkanGemvTiming* timing,
                           std::string* error) {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (!initialize(error) || !logits || !out_index || !out_value || n <= 0) {
        if (error && n <= 0) {
            *error = "invalid argmax arguments";
        }
        return false;
    }
    float out[2] = {0.0f, 0.0f};
    VectorPushConstants pc{};
    pc.op = 7;
    pc.n = static_cast<uint32_t>(n);
    const bool ok = impl_->runVector(pc,
                                     logits,
                                     static_cast<size_t>(n) * sizeof(float),
                                     nullptr,
                                     0,
                                     nullptr,
                                     0,
                                     out,
                                     sizeof(out),
                                     out,
                                     nullptr,
                                     timing,
                                     1,
                                     1,
                                     error);
    if (ok) {
        *out_index = static_cast<int>(out[0]);
        *out_value = out[1];
    }
    return ok;
#else
    (void)logits;
    (void)n;
    (void)out_index;
    (void)out_value;
    (void)timing;
    if (error) {
        *error = "custom Vulkan vector ops are only compiled for Android builds";
    }
    return false;
#endif
}

bool VulkanBackend::gqaDecode(const float* q,
                              const float* k_cache,
                              const float* v_cache,
                              int context,
                              int q_heads,
                              int kv_heads,
                              int head_dim,
                              float* out,
                              VulkanGemvTiming* timing,
                              std::string* error) {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (!initialize(error) || !q || !k_cache || !v_cache || !out || context <= 0 || q_heads <= 0 || kv_heads <= 0 ||
        head_dim <= 0) {
        if (error && (context <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0)) {
            *error = "invalid gqaDecode arguments";
        }
        return false;
    }
    const size_t q_bytes = static_cast<size_t>(q_heads) * static_cast<size_t>(head_dim) * sizeof(float);
    const size_t kv_bytes =
        static_cast<size_t>(context) * static_cast<size_t>(kv_heads) * static_cast<size_t>(head_dim) * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(q_heads) * static_cast<size_t>(head_dim) * sizeof(float);
    VectorPushConstants pc{};
    pc.op = 8;
    pc.n = static_cast<uint32_t>(context);
    pc.q_heads = static_cast<uint32_t>(q_heads);
    pc.k_heads = static_cast<uint32_t>(kv_heads);
    pc.head_dim = static_cast<uint32_t>(head_dim);
    return impl_->runVector(pc,
                            q,
                            q_bytes,
                            k_cache,
                            kv_bytes,
                            v_cache,
                            kv_bytes,
                            nullptr,
                            out_bytes,
                            out,
                            nullptr,
                            timing,
                            static_cast<uint32_t>(q_heads),
                            1,
                            error);
#else
    (void)q;
    (void)k_cache;
    (void)v_cache;
    (void)context;
    (void)q_heads;
    (void)kv_heads;
    (void)head_dim;
    (void)out;
    (void)timing;
    if (error) {
        *error = "custom Vulkan vector ops are only compiled for Android builds";
    }
    return false;
#endif
}

bool VulkanBackend::linearAttentionConv1d(const float* input,
                                          const float* state,
                                          const float* weight,
                                          int conv_dim,
                                          int kernel_dim,
                                          float* conv_out,
                                          float* new_state,
                                          VulkanGemvTiming* timing,
                                          std::string* error) {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (!initialize(error) || !input || !weight || !conv_out || conv_dim <= 0 || kernel_dim <= 0) {
        if (error && (conv_dim <= 0 || kernel_dim <= 0)) {
            *error = "invalid linearAttentionConv1d arguments";
        }
        return false;
    }
    const int history = std::max(0, kernel_dim - 1);
    std::vector<float> zero_state;
    const float* state_data = state;
    if (history > 0 && !state_data) {
        zero_state.assign(static_cast<size_t>(conv_dim) * static_cast<size_t>(history), 0.0f);
        state_data = zero_state.data();
    }
    std::vector<float> combined(static_cast<size_t>(conv_dim) +
                                    static_cast<size_t>(conv_dim) * static_cast<size_t>(history),
                                0.0f);
    VectorPushConstants pc{};
    pc.op = 9;
    pc.n = static_cast<uint32_t>(conv_dim);
    pc.heads = static_cast<uint32_t>(history);
    pc.head_dim = static_cast<uint32_t>(kernel_dim);
    const bool ok = impl_->runVector(pc,
                                     input,
                                     static_cast<size_t>(conv_dim) * sizeof(float),
                                     state_data,
                                     static_cast<size_t>(conv_dim) * static_cast<size_t>(history) * sizeof(float),
                                     weight,
                                     static_cast<size_t>(conv_dim) * static_cast<size_t>(kernel_dim) * sizeof(float),
                                     combined.data(),
                                     combined.size() * sizeof(float),
                                     combined.data(),
                                     nullptr,
                                     timing,
                                     static_cast<uint32_t>((conv_dim + 255) / 256),
                                     1,
                                     error);
    if (ok) {
        std::memcpy(conv_out, combined.data(), static_cast<size_t>(conv_dim) * sizeof(float));
        if (new_state && history > 0) {
            std::memcpy(new_state,
                        combined.data() + static_cast<size_t>(conv_dim),
                        static_cast<size_t>(conv_dim) * static_cast<size_t>(history) * sizeof(float));
        }
    }
    return ok;
#else
    (void)input;
    (void)state;
    (void)weight;
    (void)conv_dim;
    (void)kernel_dim;
    (void)conv_out;
    (void)new_state;
    (void)timing;
    if (error) {
        *error = "custom Vulkan vector ops are only compiled for Android builds";
    }
    return false;
#endif
}

bool VulkanBackend::linearAttentionStateUpdate(const float* linear_conv,
                                               const float* linear_a,
                                               const float* linear_b,
                                               const float* a_log,
                                               const float* dt_bias,
                                               float* recurrent_state,
                                               int key_heads,
                                               int value_heads,
                                               int key_dim,
                                               int value_dim,
                                               float* out,
                                               VulkanGemvTiming* timing,
                                               std::string* error) {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (!initialize(error) || !linear_conv || !linear_a || !linear_b || !a_log || !dt_bias || !recurrent_state ||
        !out || key_heads <= 0 || value_heads <= 0 || key_dim <= 0 || value_dim <= 0) {
        if (error && (key_heads <= 0 || value_heads <= 0 || key_dim <= 0 || value_dim <= 0)) {
            *error = "invalid linearAttentionStateUpdate arguments";
        }
        return false;
    }
    std::vector<float> aux(static_cast<size_t>(value_heads) * 4u, 0.0f);
    std::memcpy(aux.data(), linear_a, static_cast<size_t>(value_heads) * sizeof(float));
    std::memcpy(aux.data() + static_cast<size_t>(value_heads),
                linear_b,
                static_cast<size_t>(value_heads) * sizeof(float));
    std::memcpy(aux.data() + static_cast<size_t>(value_heads) * 2u,
                a_log,
                static_cast<size_t>(value_heads) * sizeof(float));
    std::memcpy(aux.data() + static_cast<size_t>(value_heads) * 3u,
                dt_bias,
                static_cast<size_t>(value_heads) * sizeof(float));
    const size_t mixed_values =
        static_cast<size_t>(key_heads) * static_cast<size_t>(key_dim) * 2u +
        static_cast<size_t>(value_heads) * static_cast<size_t>(value_dim);
    const size_t out_values = static_cast<size_t>(value_heads) * static_cast<size_t>(value_dim);
    const size_t state_values = static_cast<size_t>(value_heads) * static_cast<size_t>(key_dim) *
                                static_cast<size_t>(value_dim);
    std::vector<float> combined(out_values + state_values, 0.0f);
    VectorPushConstants pc{};
    pc.op = 10;
    pc.n = static_cast<uint32_t>(value_heads);
    pc.heads = static_cast<uint32_t>(key_heads);
    pc.head_dim = static_cast<uint32_t>(key_dim);
    pc.q_heads = static_cast<uint32_t>(value_dim);
    const bool ok = impl_->runVector(pc,
                                     linear_conv,
                                     mixed_values * sizeof(float),
                                     recurrent_state,
                                     state_values * sizeof(float),
                                     aux.data(),
                                     aux.size() * sizeof(float),
                                     combined.data(),
                                     combined.size() * sizeof(float),
                                     combined.data(),
                                     nullptr,
                                     timing,
                                     static_cast<uint32_t>(value_heads),
                                     1,
                                     error);
    if (ok) {
        std::memcpy(out, combined.data(), out_values * sizeof(float));
        std::memcpy(recurrent_state, combined.data() + out_values, state_values * sizeof(float));
    }
    return ok;
#else
    (void)linear_conv;
    (void)linear_a;
    (void)linear_b;
    (void)a_log;
    (void)dt_bias;
    (void)recurrent_state;
    (void)key_heads;
    (void)value_heads;
    (void)key_dim;
    (void)value_dim;
    (void)out;
    (void)timing;
    if (error) {
        *error = "custom Vulkan vector ops are only compiled for Android builds";
    }
    return false;
#endif
}

bool VulkanBackend::appendFloatVector(const float* old_data,
                                      int old_values,
                                      const float* append_data,
                                      int append_values,
                                      float* out,
                                      VulkanGemvTiming* timing,
                                      std::string* error) {
#if defined(XQ_ENABLE_CUSTOM_VULKAN) && defined(__ANDROID__)
    if (!initialize(error) || !append_data || !out || old_values < 0 || append_values <= 0) {
        if (error && (old_values < 0 || append_values <= 0)) {
            *error = "invalid appendFloatVector arguments";
        }
        return false;
    }
    float zero = 0.0f;
    const float* old_ptr = old_values > 0 && old_data ? old_data : &zero;
    VectorPushConstants pc{};
    pc.op = 11;
    pc.n = static_cast<uint32_t>(old_values);
    pc.heads = static_cast<uint32_t>(append_values);
    const int total = old_values + append_values;
    return impl_->runVector(pc,
                            old_ptr,
                            static_cast<size_t>(std::max(1, old_values)) * sizeof(float),
                            append_data,
                            static_cast<size_t>(append_values) * sizeof(float),
                            nullptr,
                            0,
                            out,
                            static_cast<size_t>(total) * sizeof(float),
                            out,
                            nullptr,
                            timing,
                            static_cast<uint32_t>((total + 255) / 256),
                            1,
                            error);
#else
    (void)old_data;
    (void)old_values;
    (void)append_data;
    (void)append_values;
    (void)out;
    (void)timing;
    if (error) {
        *error = "custom Vulkan vector ops are only compiled for Android builds";
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
    json << "],\"vector_results\":[";
    bool first_vector = true;
    auto appendVectorResult = [&](const char* name,
                                  bool ok,
                                  const DiffStats& diff,
                                  double tolerance,
                                  const VulkanGemvTiming& timing,
                                  const std::string& vector_error) {
        const bool pass = ok && diff.max_abs <= tolerance && std::isfinite(diff.max_abs);
        all_pass = all_pass && pass;
        if (!first_vector) {
            json << ",";
        }
        first_vector = false;
        json << "{\"name\":\"" << name << "\",\"backend\":\"vulkan\""
             << ",\"vulkan_upload_ms\":" << timing.upload_ms
             << ",\"vulkan_dispatch_ms\":" << timing.dispatch_ms
             << ",\"vulkan_download_ms\":" << timing.download_ms
             << ",\"vulkan_total_ms\":" << timing.total_ms
             << ",\"max_abs_error\":" << diff.max_abs
             << ",\"mean_abs_error\":" << diff.mean_abs
             << ",\"tolerance\":" << tolerance
             << ",\"pass\":" << (pass ? "true" : "false");
        if (!ok) {
            json << ",\"error\":\"" << jsonEscape(vector_error) << "\"";
        }
        json << "}";
    };

    {
        const int n = 1024;
        std::vector<float> x = makeDeterministicInput(n);
        std::vector<float> w(n);
        for (int i = 0; i < n; ++i) {
            w[static_cast<size_t>(i)] = 0.8f + 0.001f * static_cast<float>(i % 17);
        }
        std::vector<float> cpu(n), vk(n);
        double sum = 0.0;
        for (float v : x) {
            sum += static_cast<double>(v) * static_cast<double>(v);
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(sum / n) + 1.0e-6f);
        for (int i = 0; i < n; ++i) {
            cpu[static_cast<size_t>(i)] = x[static_cast<size_t>(i)] * inv * w[static_cast<size_t>(i)];
        }
        VulkanGemvTiming timing;
        std::string vector_error;
        const bool ok = backend.rmsNorm(x.data(), w.data(), n, 1.0e-6f, vk.data(), &timing, &vector_error);
        appendVectorResult("rmsnorm_vector", ok, ok ? compareVectors(cpu, vk) : DiffStats{}, 1.0e-4, timing, vector_error);
    }

    {
        const int q_heads = 2;
        const int k_heads = 1;
        const int head_dim = 16;
        const int rotary_dim = 8;
        const int position = 7;
        const float theta = 10000000.0f;
        std::vector<float> q = makeDeterministicInput(q_heads * head_dim);
        std::vector<float> k = makeDeterministicInput(k_heads * head_dim);
        std::vector<float> q_ref = q;
        std::vector<float> k_ref = k;
        auto ropeRef = [&](std::vector<float>* data, int heads) {
            const int active_dim = std::min(head_dim, rotary_dim);
            const int half = active_dim / 2;
            for (int h = 0; h < heads; ++h) {
                float* base = data->data() + h * head_dim;
                for (int i = 0; i < half; ++i) {
                    const float inv_freq = std::pow(theta, -static_cast<float>(2 * i) / static_cast<float>(active_dim));
                    const float angle = static_cast<float>(position) * inv_freq;
                    const float c = std::cos(angle);
                    const float s = std::sin(angle);
                    const float x0 = base[i];
                    const float x1 = base[half + i];
                    base[i] = x0 * c - x1 * s;
                    base[half + i] = x1 * c + x0 * s;
                }
            }
        };
        ropeRef(&q_ref, q_heads);
        ropeRef(&k_ref, k_heads);
        VulkanGemvTiming timing;
        std::string vector_error;
        const bool ok = backend.rope(q.data(), q_heads, k.data(), k_heads, head_dim, rotary_dim, position, theta, &timing, &vector_error);
        std::vector<float> cpu = q_ref;
        cpu.insert(cpu.end(), k_ref.begin(), k_ref.end());
        std::vector<float> got = q;
        got.insert(got.end(), k.begin(), k.end());
        appendVectorResult("rope_qk_vector", ok, ok ? compareVectors(cpu, got) : DiffStats{}, 1.0e-4, timing, vector_error);
    }

    {
        const int n = 2048;
        std::vector<float> gate = makeDeterministicInput(n);
        std::vector<float> up = makeDeterministicInput(n);
        std::vector<float> cpu(n), vk(n);
        for (int i = 0; i < n; ++i) {
            const float g = gate[static_cast<size_t>(i)];
            cpu[static_cast<size_t>(i)] = g / (1.0f + std::exp(-g)) * up[static_cast<size_t>(i)];
        }
        VulkanGemvTiming timing;
        std::string vector_error;
        const bool ok = backend.siluGateMul(gate.data(), up.data(), n, vk.data(), &timing, &vector_error);
        appendVectorResult("silu_gate_mul_vector", ok, ok ? compareVectors(cpu, vk) : DiffStats{}, 1.0e-5, timing, vector_error);
    }

    {
        const int n = 2048;
        std::vector<float> hidden = makeDeterministicInput(n);
        std::vector<float> delta = makeDeterministicInput(n);
        std::vector<float> cpu = hidden;
        for (int i = 0; i < n; ++i) {
            cpu[static_cast<size_t>(i)] += delta[static_cast<size_t>(i)];
        }
        VulkanGemvTiming timing;
        std::string vector_error;
        const bool ok = backend.residualAdd(hidden.data(), delta.data(), n, &timing, &vector_error);
        appendVectorResult("residual_add_vector", ok, ok ? compareVectors(cpu, hidden) : DiffStats{}, 1.0e-6, timing, vector_error);
    }

    {
        const int context = 5;
        const int q_heads = 4;
        const int kv_heads = 2;
        const int head_dim = 8;
        std::vector<float> q = makeDeterministicInput(q_heads * head_dim);
        std::vector<float> k_cache = makeDeterministicInput(context * kv_heads * head_dim);
        std::vector<float> v_cache = makeDeterministicInput(context * kv_heads * head_dim);
        std::vector<float> cpu(q_heads * head_dim, 0.0f);
        std::vector<float> vk(q_heads * head_dim, 0.0f);
        const int group = std::max(1, q_heads / kv_heads);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        std::vector<float> scores(context);
        for (int qh = 0; qh < q_heads; ++qh) {
            const int kh = std::min(kv_heads - 1, qh / group);
            float max_score = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < context; ++t) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q[static_cast<size_t>(qh * head_dim + d)] *
                           k_cache[static_cast<size_t>((t * kv_heads + kh) * head_dim + d)];
                }
                scores[static_cast<size_t>(t)] = dot * scale;
                max_score = std::max(max_score, scores[static_cast<size_t>(t)]);
            }
            float denom = 0.0f;
            for (int t = 0; t < context; ++t) {
                scores[static_cast<size_t>(t)] = std::exp(scores[static_cast<size_t>(t)] - max_score);
                denom += scores[static_cast<size_t>(t)];
            }
            const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
            for (int t = 0; t < context; ++t) {
                const float weight = scores[static_cast<size_t>(t)] * inv;
                for (int d = 0; d < head_dim; ++d) {
                    cpu[static_cast<size_t>(qh * head_dim + d)] +=
                        weight * v_cache[static_cast<size_t>((t * kv_heads + kh) * head_dim + d)];
                }
            }
        }
        VulkanGemvTiming timing;
        std::string vector_error;
        const bool ok = backend.gqaDecode(
            q.data(), k_cache.data(), v_cache.data(), context, q_heads, kv_heads, head_dim, vk.data(), &timing, &vector_error);
        appendVectorResult("gqa_decode_vector", ok, ok ? compareVectors(cpu, vk) : DiffStats{}, 1.0e-4, timing, vector_error);
    }

    {
        const int conv_dim = 96;
        const int kernel_dim = 4;
        const int history = kernel_dim - 1;
        std::vector<float> input = makeDeterministicInput(conv_dim);
        std::vector<float> state = makeDeterministicInput(conv_dim * history);
        std::vector<float> weight = makeDeterministicInput(conv_dim * kernel_dim);
        std::vector<float> cpu_out(conv_dim, 0.0f);
        std::vector<float> cpu_state = state;
        for (int c = 0; c < conv_dim; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < history; ++k) {
                acc += state[static_cast<size_t>(c * history + k)] *
                       weight[static_cast<size_t>(c * kernel_dim + k)];
            }
            acc += input[static_cast<size_t>(c)] * weight[static_cast<size_t>(c * kernel_dim + history)];
            cpu_out[static_cast<size_t>(c)] = acc / (1.0f + std::exp(-acc));
            for (int k = 0; k < history; ++k) {
                cpu_state[static_cast<size_t>(c * history + k)] =
                    (k + 1 < history) ? state[static_cast<size_t>(c * history + k + 1)] : input[static_cast<size_t>(c)];
            }
        }
        std::vector<float> vk_out(conv_dim, 0.0f);
        std::vector<float> vk_state(state.size(), 0.0f);
        VulkanGemvTiming timing;
        std::string vector_error;
        const bool ok = backend.linearAttentionConv1d(input.data(),
                                                      state.data(),
                                                      weight.data(),
                                                      conv_dim,
                                                      kernel_dim,
                                                      vk_out.data(),
                                                      vk_state.data(),
                                                      &timing,
                                                      &vector_error);
        std::vector<float> cpu = cpu_out;
        cpu.insert(cpu.end(), cpu_state.begin(), cpu_state.end());
        std::vector<float> got = vk_out;
        got.insert(got.end(), vk_state.begin(), vk_state.end());
        appendVectorResult("linear_attention_conv1d_vector",
                           ok,
                           ok ? compareVectors(cpu, got) : DiffStats{},
                           1.0e-5,
                           timing,
                           vector_error);
    }

    {
        const int key_heads = 2;
        const int value_heads = 4;
        const int key_dim = 8;
        const int value_dim = 8;
        const int mixed_values = key_heads * key_dim * 2 + value_heads * value_dim;
        std::vector<float> linear_conv = makeDeterministicInput(mixed_values);
        std::vector<float> linear_a = makeDeterministicInput(value_heads);
        std::vector<float> linear_b = makeDeterministicInput(value_heads);
        std::vector<float> a_log(value_heads);
        std::vector<float> dt_bias(value_heads);
        for (int i = 0; i < value_heads; ++i) {
            a_log[static_cast<size_t>(i)] = -0.7f + 0.02f * static_cast<float>(i);
            dt_bias[static_cast<size_t>(i)] = 0.03f * static_cast<float>(i - 1);
        }
        std::vector<float> cpu_state = makeDeterministicInput(value_heads * key_dim * value_dim);
        std::vector<float> vk_state = cpu_state;
        std::vector<float> cpu(value_heads * value_dim, 0.0f);
        std::vector<float> vk(value_heads * value_dim, 0.0f);
        const float* query = linear_conv.data();
        const float* key = linear_conv.data() + key_heads * key_dim;
        const float* value = linear_conv.data() + key_heads * key_dim * 2;
        const int factor = std::max(1, value_heads / std::max(1, key_heads));
        for (int vh = 0; vh < value_heads; ++vh) {
            const int kh = std::min(key_heads - 1, vh / factor);
            const float beta = 1.0f / (1.0f + std::exp(-linear_b[static_cast<size_t>(vh)]));
            const float gate = -std::exp(a_log[static_cast<size_t>(vh)]) *
                               std::log(1.0f + std::exp(linear_a[static_cast<size_t>(vh)] +
                                                        dt_bias[static_cast<size_t>(vh)]));
            gatedDeltaDecodeReference(query + kh * key_dim,
                                      key + kh * key_dim,
                                      value + vh * value_dim,
                                      beta,
                                      gate,
                                      key_dim,
                                      value_dim,
                                      cpu_state.data() + static_cast<size_t>(vh * key_dim * value_dim),
                                      cpu.data() + static_cast<size_t>(vh * value_dim));
        }
        VulkanGemvTiming timing;
        std::string vector_error;
        const bool ok = backend.linearAttentionStateUpdate(linear_conv.data(),
                                                           linear_a.data(),
                                                           linear_b.data(),
                                                           a_log.data(),
                                                           dt_bias.data(),
                                                           vk_state.data(),
                                                           key_heads,
                                                           value_heads,
                                                           key_dim,
                                                           value_dim,
                                                           vk.data(),
                                                           &timing,
                                                           &vector_error);
        std::vector<float> cpu_combined = cpu;
        cpu_combined.insert(cpu_combined.end(), cpu_state.begin(), cpu_state.end());
        std::vector<float> got = vk;
        got.insert(got.end(), vk_state.begin(), vk_state.end());
        appendVectorResult("linear_attention_state_update_vector",
                           ok,
                           ok ? compareVectors(cpu_combined, got) : DiffStats{},
                           1.0e-4,
                           timing,
                           vector_error);
    }

    {
        const int old_values = 17;
        const int append_values = 9;
        std::vector<float> old_data = makeDeterministicInput(old_values);
        std::vector<float> append_data = makeDeterministicInput(append_values);
        std::vector<float> cpu = old_data;
        cpu.insert(cpu.end(), append_data.begin(), append_data.end());
        std::vector<float> vk(cpu.size(), 0.0f);
        VulkanGemvTiming timing;
        std::string vector_error;
        const bool ok = backend.appendFloatVector(
            old_data.data(), old_values, append_data.data(), append_values, vk.data(), &timing, &vector_error);
        appendVectorResult("append_vector", ok, ok ? compareVectors(cpu, vk) : DiffStats{}, 1.0e-6, timing, vector_error);
    }

    {
        const int n = 2048;
        std::vector<float> logits = makeDeterministicInput(n);
        logits[1337] = 42.0f;
        int ref_idx = 0;
        float ref_val = logits[0];
        for (int i = 1; i < n; ++i) {
            if (logits[static_cast<size_t>(i)] > ref_val) {
                ref_val = logits[static_cast<size_t>(i)];
                ref_idx = i;
            }
        }
        int vk_idx = -1;
        float vk_val = 0.0f;
        VulkanGemvTiming timing;
        std::string vector_error;
        const bool ok = backend.argmax(logits.data(), n, &vk_idx, &vk_val, &timing, &vector_error);
        DiffStats diff;
        diff.max_abs = (ok && vk_idx == ref_idx) ? std::abs(static_cast<double>(vk_val - ref_val)) : 1.0e9;
        diff.mean_abs = diff.max_abs;
        appendVectorResult("argmax_vector", ok, diff, 1.0e-6, timing, vector_error);
    }

    json << "],\"wall_clock_ms\":" << elapsedMs(start, Clock::now())
         << ",\"status\":\"" << (all_pass ? "ok" : "error") << "\",\"pass\":" << (all_pass ? "true" : "false")
         << "}";
    copyJson(json_out, json_capacity, json.str());
    return all_pass ? XQ_OK : XQ_ERR_RUNTIME;
}

}  // namespace kernels
}  // namespace xq
