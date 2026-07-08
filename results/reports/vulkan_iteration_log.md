# Vulkan Iteration Log

This log records the implementation/debug iterations for the post-v27 custom Vulkan attempt. The official v27 CPU final remains the accepted systems/kernel and quality-gated result unless a later run satisfies the full Vulkan acceptance criteria.

## Baseline Restore

- Working copy: `C:\xqvk2`
- Starting commit: `7879066 Document full Vulkan backend blocker`
- MNN commit: `0bff03cbef43c783f44e41484b9f8a0b28bd758d`
- Device Farm project: `arn:aws:devicefarm:us-west-2:884244642857:project:64d2cc31-abd6-49f8-97da-162f82410bc0`
- Device pool: `arn:aws:devicefarm:us-west-2:884244642857:devicepool:64d2cc31-abd6-49f8-97da-162f82410bc0/14d31c96-b8fc-4930-99c7-1a8948124213`

Baseline build evidence:

- `results/reports/evidence/baseline_fetch_mnn_xqvk2.txt`
- `results/reports/evidence/baseline_build_mnn_android_xqvk2.txt`
- `results/reports/evidence/baseline_build_android_xqvk2.txt`
- `results/reports/evidence/baseline_host_cmake_xqvk2.txt`
- `results/reports/evidence/baseline_host_build_xqvk2.txt`
- `results/reports/evidence/baseline_host_ctest_xqvk2.txt`

## Attempt 1: Add Real Vulkan Runtime And W4A16 Shader

Code changes:

- Added `customlib/runtime/vulkan_backend.hpp`.
- Added `customlib/runtime/vulkan_backend.cpp`.
- Added `customlib/kernels/generated/vulkan/w4a16_gemv.comp`.
- Added generated SPIR-V include `customlib/kernels/generated/vulkan/w4a16_gemv_spv.inc`.
- Added `xq_run_vulkan_w4a16_selftest` C ABI.
- Added `NativeBenchmark.runVulkanSelfTest` JNI path.
- Added Android instrumentation flag `run_vulkan_selftest=true`.

What was implemented:

- Vulkan instance/device/queue/pipeline setup.
- Host-visible buffers and dispatch path.
- W4A16 GEMV compute shader with packed int4 read and groupwise dequantization.
- CPU reference comparison for 256 x 256 and 4096 x 4096 shapes.

Result:

- Host build initially exposed a C++ issue in the `VulkanBackend::Impl` definition.
- Fix: moved the `Impl` definition out of the anonymous namespace so it matched the forward declaration.
- Host ctest then passed.
- Android build passed.

Evidence:

- `results/reports/evidence/vulkan_attempt1_host_build.log`
- `results/reports/evidence/vulkan_attempt1_host_ctest.log`
- `results/reports/evidence/vulkan_attempt1_build_android.log`

## Attempt 2: Device Farm W4A16 Selftest, First Spec

Run ARN:

```text
arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/ac30385c-0fbb-4a1d-a897-ac870a1d8efc
```

Result:

- Device Farm run result: FAILED.
- Failure reason: test spec grepped the wrong stream for `BENCH_VULKAN_SELFTEST_JSON`.
- Kernel artifact result: Vulkan W4A16 selftest JSON showed the kernel itself passed.

Evidence:

- `results/reports/evidence/vulkan_attempt1_selftest_schedule.log`
- `results/reports/evidence/vulkan_attempt1_selftest_wait.log`
- `results/reports/evidence/vulkan_attempt1_selftest_download.log`
- `results/reports/evidence/vulkan_w4a16_selftest_attempt1.json`

Next fix:

- Update the test spec to read the artifact JSON written by instrumentation rather than relying on one log stream.

## Attempt 3: Device Farm W4A16 Selftest, Fixed Spec

Run ARN:

```text
arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/da6dfbc9-27eb-4a42-9202-bffda6b6bbe6
```

Result:

- Device Farm result: PASSED.
- Vulkan runtime initialized on `Adreno (TM) 840`.
- Vulkan W4A16 GEMV matched CPU reference for 256 x 256 and 4096 x 4096 test shapes.

Selftest metrics:

| Shape | CPU reference ms | Vulkan dispatch ms | Vulkan total ms | Max abs error | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| 256 x 256 | 0.00787 | 0.46615 | 0.64829 | 4.17e-07 | PASS |
| 4096 x 4096 | 1.04010 | 8.75693 | 12.74010 | 6.85e-06 | PASS |

Evidence:

- `results/reports/evidence/vulkan_attempt2_selftest_spec_upload.log`
- `results/reports/evidence/vulkan_attempt2_selftest_schedule.log`
- `results/reports/evidence/vulkan_attempt2_selftest_wait.log`
- `results/reports/evidence/vulkan_attempt2_selftest_download.log`
- `results/reports/evidence/vulkan_w4a16_selftest_attempt2_passed.json`

Next fix:

- Integrate the Vulkan W4A16 GEMV path into the measured custom runtime for projection families.

## Attempt 4: Runtime Integration For W4A16 Projection Families

Code changes:

- Updated `customlib/runtime/custom_model.hpp`.
- Updated `customlib/runtime/custom_model.cpp`.
- Updated `customlib/runtime/session.cpp`.
- Updated `android/benchmark_app/src/main/cpp/benchmark_jni.cpp`.

What changed:

- `CustomModel::load` accepts backend request.
- `backend = vulkan` or `cpu_vulkan_hybrid` initializes `VulkanBackend`.
- `runLinear` attempts `VulkanBackend::gemvW4A16` for W4 matrices.
- Projection trace rows report `backend = vulkan` when the Vulkan GEMV succeeds.
- Benchmark JSON reports `custom_backend_actual = cpu_vulkan_hybrid` for this integration, not full Vulkan.

Build/test result:

- Host build: PASS.
- Host ctest: PASS, 2 / 2.
- Android build: PASS.

Evidence:

- `results/reports/evidence/vulkan_final_host_build.txt`
- `results/reports/evidence/vulkan_final_host_ctest.txt`
- `results/reports/evidence/vulkan_final_android_build.txt`

## Attempt 5: Full-Model Short Integration Run

Run ARN:

```text
arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/938721e9-46ab-4266-8e02-4445eb0704bd
```

Run purpose:

- Prove the full Qwen3.5 model path can enter the custom Vulkan W4A16 projection kernels.
- Keep the run short because the first integration is upload-per-call and expected to be slow.

Settings:

| Field | Value |
| --- | --- |
| Model | full Qwen3.5-9B custom package |
| Model SHA | `9de692be1c1ef1002fac25bd8f93c76e1d31975caa234fe1725f9eb294bfaa34` |
| Requested backend | `vulkan` |
| Actual backend | `cpu_vulkan_hybrid` |
| Prompt tokens | 1 |
| max_new_tokens | 1 |
| Warmup / measured | 0 / 1 |

Result:

- Device Farm result: PASSED.
- `BENCH_RESULT_JSON` emitted.
- `use_mnn_fallback = 0`.
- `calls_mnn_llm_response_for_measured_generation = false`.
- `vulkan_generation_kernels_used = true`.
- Projection-family trace rows used `backend = vulkan`.

Performance:

| Metric | Value |
| --- | ---: |
| Prefill TPS | 0.186569 |
| Decode TPS | 0.173017 |
| Decode TPOT | 5779.77 ms |

Representative Vulkan trace rows:

| Kernel | Calls | Total ms | Mean ms |
| --- | ---: | ---: | ---: |
| `linear_attn_in_proj_qkv_w4a16` | 48 | 1322.33 | 27.5486 |
| `linear_down_proj_w4a16` | 64 | 2270.69 | 35.4796 |
| `linear_gate_proj_w4a16` | 64 | 2509.98 | 39.2184 |
| `linear_q_proj_w4a16` | 16 | 438.759 | 27.4225 |

Evidence:

- `results/reports/evidence/customlib_vulkan_integration_short_benchmark.json`
- `results/reports/evidence/customlib_vulkan_integration_short_summary.json`
- `results/reports/evidence/customlib_vulkan_integration_short_model_bootstrap.json`
- `results/reports/evidence/customlib_vulkan_integration_short_model_bootstrap_discovery.txt`

Metadata caveat: this run occurred before the probe-status wording patch. The raw `vulkan_attempt` subobject has stale "not enabled" text, but the same JSON records `custom_backend_actual = cpu_vulkan_hybrid`, `custom_path.vulkan_generation_kernels_used = true`, and per-kernel rows with `backend = vulkan`. The source now reports the W4A16-linear hybrid status directly.

Why it was not accepted:

- It was a 1-token short integration, not the required 512/256 final benchmark.
- Actual backend was `cpu_vulkan_hybrid`, not full `vulkan`.
- Major non-linear and output op families stayed CPU.
- No `BENCH_QUALITY_JSON` was produced.
- Performance was far below v27 CPU custom and stock MNN CPU.

## Current Blocker Summary

The Vulkan implementation is real but incomplete:

- W4A16 Vulkan GEMV exists and passes Device Farm correctness.
- W4A16 projection families can run through Vulkan inside the full-model custom path.
- The runtime still needs persistent GPU-resident weights and Vulkan implementations for RMSNorm, RoPE, attention, linear-attention state, lm_head, sampling, and prefill KV.
- The current upload-per-call design is slower than CPU and cannot approach 10 TPS.

No final Vulkan success, quality pass, speedup, or 10 TPS claim is made.
