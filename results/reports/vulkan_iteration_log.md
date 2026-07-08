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

## Attempt 6: Add Vulkan Vector, GQA, And Argmax Kernels

Code changes:

- Added `customlib/kernels/generated/vulkan/vector_ops.comp`.
- Added generated SPIR-V include `customlib/kernels/generated/vulkan/vector_ops_spv.inc`.
- Extended `customlib/runtime/vulkan_backend.*` with vector/tensor dispatch helpers.
- Integrated Vulkan attempts in `customlib/runtime/custom_model.cpp`.

New kernel families attempted:

- RMSNorm and Q/K normalization
- Qwen3.5 active-slice RoPE
- grouped-query attention score / softmax / V reduce
- SiLU gate multiply
- residual add
- attention output gate
- greedy argmax

Device Farm selftest evidence:

- Run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/f3506fac-58d8-4b03-bdf1-5f81d24af6ae`
- Evidence JSON: `results/reports/evidence/vulkan_linear_state_selftest_passed.json`

Result:

- W4A16 4096 x 4096: PASS, max abs error `6.85453e-06`.
- `rmsnorm_vector`: PASS.
- `rope_qk_vector`: PASS.
- `silu_gate_mul_vector`: PASS.
- `residual_add_vector`: PASS.
- `gqa_decode_vector`: PASS.
- `argmax_vector`: PASS.

## Attempt 7: Linear-Attention State Vulkan Fix

Initial state-update shader result:

- Run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/4e0bfbd2-d0c7-49ba-b70c-ec8cf197202b`
- Result: failed `linear_attention_state_update_vector`.
- Failure: max abs error `0.0007919` exceeded tolerance `0.0001`.

Fix:

- The shader computed the prediction from the pre-decay recurrent state.
- The CPU reference decays state before computing the prediction.
- Updated shader math to use `decay * state` in the prediction term.

Retest:

- Run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/f3506fac-58d8-4b03-bdf1-5f81d24af6ae`
- Result: PASSED.
- `linear_attention_state_update_vector` max abs error `1.49012e-08`.

## Attempt 8: Extended Full-Model Short Integration

Run ARN:

```text
arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/e6c5ec8c-7f52-4164-bc7f-c64f5e6105d5
```

Settings:

| Field | Value |
| --- | --- |
| Model | full Qwen3.5-9B package |
| Prompt tokens | 1 |
| max_new_tokens | 1 |
| Warmup / measured | 0 / 1 |
| Requested backend | `vulkan` |
| Actual backend | `cpu_vulkan_hybrid` |

Result:

- Device Farm result: PASSED.
- `BENCH_RESULT_JSON` emitted.
- `use_mnn_fallback = 0`.
- `calls_mnn_llm_response_for_measured_generation = false`.
- `fallback_op_families = []`.
- `vulkan_generation_kernels_used = true`.
- Decode TPS `0.139909`, TPOT `7147.49 ms`.

Evidence:

- `results/reports/evidence/customlib_vulkan_linear_state_integration_short_benchmark.json`

Backend map in the evidence reports Vulkan for:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj,
linear_attention_projections,
rmsnorm, rope, attention,
linear_attention_state, activation,
lm_head, sampling, prefill_kv_build
```

Remaining issue:

- The actual backend remains `cpu_vulkan_hybrid`.
- `embedding_bf16_row_read` remains CPU.
- Performance is far below the accepted CPU final due per-operation host-visible upload/dispatch/download.

## Attempt 9: Vulkan/Hybrid Quality Validation

First quality attempt:

- Run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/01cb5b7b-7cc8-476a-8a99-f47a1f9711b0`
- Result: failed before `BENCH_QUALITY_JSON`; the test spec used expired model presigned URLs and download returned HTTP 403.

Retry:

- Run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/e7081f14-99c2-4943-b6f9-222f8501fb90`
- Evidence:
  - `results/reports/evidence/quality_validation_custom_vulkan_linear_state_english.json`
  - `results/reports/evidence/quality_validation_vulkan_linear_state_english_comparison.json`
  - `results/reports/quality_validation_vulkan_linear_state_report.md`

Result:

- `BENCH_QUALITY_JSON` emitted.
- Quality sanity gate: PASS.
- Comparison-gate prompts: 5 / 5.
- Exact full-token matches: 1 / 5.
- Invalid tokens: none.
- Empty outputs: none.
- Repeated token 220: none.
- Degenerate repetition: none.

This is a useful output sanity pass, not a production semantic-quality claim.

## Attempt 10: Required Full 512/256 Benchmark

Run ARN:

```text
arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/e6650fcf-589f-49f6-8033-ed2030873d95
```

Settings:

| Field | Value |
| --- | --- |
| Model | full Qwen3.5-9B package |
| Prompt tokens | 512 |
| max_new_tokens | 256 |
| Warmup / measured | 1 / 3 |
| Requested backend | `vulkan` |

Result:

- Device Farm status/result: `COMPLETED` / `STOPPED`.
- Job timeout: 150 minutes.
- No accepted `BENCH_RESULT_JSON`.
- Evidence: `results/reports/evidence/customlib_vulkan_full_benchmark_stopped.json`.

Logcat progress:

```text
16:50:30 BENCH_START engine=customlib requested_backend=vulkan actual_backend=cpu_vulkan_hybrid
18:04:35 BENCH_ITER engine=customlib warmup=1 prompt=512 generated=256 prefill_ms=2840779.534 decode_ms=1599572.239 status=0
```

The completed warmup alone took about 74.006 minutes, with partial warmup decode TPS `0.160043`. The job stopped before the three measured iterations, so this cannot be used as the final performance comparison.

## Current Blocker Summary

The Vulkan implementation is real and exercised on Device Farm, but it is not an accepted full Vulkan final:

- W4A16 Vulkan GEMV exists and passes Device Farm correctness.
- Vulkan vector/tensor kernels for RMSNorm, RoPE, GQA, linear-attention state, KV append, activation/residual, output gate, and argmax pass Device Farm selftests.
- A full-model short custom path reports Vulkan for the major traced tensor families and does not call MNN generation.
- Vulkan/hybrid quality validation emits `BENCH_QUALITY_JSON` and passes sanity.
- The runtime still reports `custom_backend_actual = cpu_vulkan_hybrid`, not `vulkan`.
- The full 512/256 benchmark stopped at 150 minutes before measured iterations and before final `BENCH_RESULT_JSON`.
- The current host-visible upload/download design is far slower than CPU and cannot approach 10 TPS.

No final full-Vulkan success, 10 TPS claim, or speedup claim is made.
