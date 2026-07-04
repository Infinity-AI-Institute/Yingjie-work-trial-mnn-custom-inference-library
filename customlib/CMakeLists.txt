set(XQWEN35_SOURCES
  runtime/session.cpp
  runtime/tokenizer_bridge.cpp
  runtime/sampler.cpp
  runtime/cache_manager.cpp
  runtime/telemetry.cpp
  runtime/custom_model.cpp
  mnn_bridge/mnn_loader.cpp
  mnn_bridge/graph_rewrite.cpp
  mnn_bridge/custom_op_registry.cpp
  mnn_bridge/xqwen_linear_execution.cpp
  mnn_bridge/xqwen_rmsnorm_execution.cpp
  mnn_bridge/xqwen_rope_execution.cpp
  mnn_bridge/xqwen_attention_execution.cpp
  mnn_bridge/xqwen_delta_execution.cpp
  mnn_bridge/xqwen_ffn_execution.cpp
  kernels/reference/ref_gemv.cpp
  kernels/reference/ref_rmsnorm.cpp
  kernels/reference/ref_rope.cpp
  kernels/reference/ref_attention.cpp
  kernels/reference/ref_delta.cpp
  kernels/generated/generated_dispatch.cpp
  kernels/generated/arm64_neon/xq_gemv_w4a16_neon.cpp
  kernels/generated/arm64_neon/xq_gemv_w3a16_neon.cpp
  kernels/generated/arm64_neon/xq_gemv_w2a16_neon.cpp
)

add_library(xqwen35 SHARED ${XQWEN35_SOURCES})
add_library(xqwen35::xqwen35 ALIAS xqwen35)

target_include_directories(xqwen35
  PUBLIC include
  PRIVATE runtime kernels mnn_bridge
)

target_compile_definitions(xqwen35 PRIVATE XQWEN35_BUILD=1)

if(XQ_ENABLE_MNN)
  target_compile_definitions(xqwen35 PRIVATE XQ_ENABLE_MNN=1)
  target_include_directories(xqwen35 PRIVATE ../third_party/MNN/include ../third_party/MNN/source)
endif()

if(MSVC)
  target_compile_options(xqwen35 PRIVATE /O2 /DNDEBUG)
else()
  target_compile_options(xqwen35 PRIVATE -O3 -DNDEBUG -Wall -Wextra -Wno-unused-parameter)
  if(ANDROID)
    target_compile_options(xqwen35 PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(xqwen35 PRIVATE -Wl,--gc-sections)
  endif()
endif()
