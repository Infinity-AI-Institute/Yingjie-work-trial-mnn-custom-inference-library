# Final Vulkan Delivery Attempt: Blocker Report

## Verdict

Outcome B: Vulkan blocked after real implementation attempts.

This pass did not stop at auditing the empty Vulkan folder. It added a custom Vulkan runtime, implemented a real W4A16 GEMV compute shader, validated that shader on AWS Device Farm Samsung Galaxy S26 Ultra, and integrated the Vulkan GEMV path into measured custom Qwen3.5 generation for projection families.

The result is not accepted as the requested final Vulkan delivery because it is not full Vulkan generation and it is not quality validated:

- `custom_backend_requested = vulkan` was exercised in a full-model short integration run.
- `custom_backend_actual = cpu_vulkan_hybrid`, not `vulkan`.
- `vulkan_generation_kernels_used = true` for W4A16 projection GEMV rows.
- Major op families still ran on CPU: RMSNorm, RoPE, attention, linear-attention state, lm_head, sampling, and prefill KV build.
- No Vulkan `BENCH_QUALITY_JSON` was produced.
- No full 512-token / 256-token final Vulkan benchmark was accepted.
- No 10 tokens/sec claim is made.

The previous official v27 CPU customlib result remains the official final systems/kernel and quality-gated benchmark result. The Vulkan work in this report is visible implementation evidence and a blocker record, not a final speedup claim.

## Starting Point

- Fresh working copy: `C:\xqvk2`
- Starting commit: `7879066 Document full Vulkan backend blocker`
- Remote: `https://github.com/Infinity-AI-Institute/Yingjie-work-trial-mnn-custom-inference-library`
- MNN pinned commit: `0bff03cbef43c783f44e41484b9f8a0b28bd758d`
- Device Farm project: `arn:aws:devicefarm:us-west-2:884244642857:project:64d2cc31-abd6-49f8-97da-162f82410bc0`
- Device pool: `arn:aws:devicefarm:us-west-2:884244642857:devicepool:64d2cc31-abd6-49f8-97da-162f82410bc0/14d31c96-b8fc-4930-99c7-1a8948124213`
- Target device observed by Vulkan selftest: `Adreno (TM) 840`

AWS credentials were usable with profile `qpnpu-devicefarm`. The model package used the existing three-part EXTERNAL_DATA upload flow and verified the final model SHA:

```text
9de692be1c1ef1002fac25bd8f93c76e1d31975caa234fe1725f9eb294bfaa34
```

## What Was Implemented

### Vulkan Runtime And Connector

New files:

- `customlib/runtime/vulkan_backend.hpp`
- `customlib/runtime/vulkan_backend.cpp`

The runtime attempts to:

- create a Vulkan instance
- select a physical device
- create a logical device and compute queue
- allocate host-visible buffers
- upload/download buffers
- create descriptor layouts and a compute pipeline
- dispatch a compute shader
- measure upload, dispatch, download, and total CPU wall-clock time

Android CMake now links Vulkan for the custom library:

- `customlib/CMakeLists.txt`

### Vulkan W4A16 GEMV Kernel

New shader files:

- `customlib/kernels/generated/vulkan/w4a16_gemv.comp`
- `customlib/kernels/generated/vulkan/w4a16_gemv_spv.inc`

The shader reads packed 4-bit weights, unpacks int4 codes, applies groupwise scale and zero-point values, and computes GEMV directly without materializing a dequantized matrix.

It supports the dequantization formulas used by the custom package:

```text
code_dot = sum(code_i * activation_i)
x_sum    = sum(activation_i)
partial  = scale * (code_dot - zero * x_sum)
partial  = scale * code_dot + zero * x_sum
```

### JNI / Instrumentation Hooks

Updated files:

- `customlib/include/xqwen35.h`
- `customlib/runtime/session.cpp`
- `android/benchmark_app/src/main/java/com/example/xqwen35bench/NativeBenchmark.java`
- `android/benchmark_app/src/main/cpp/benchmark_jni.cpp`
- `android/benchmark_app/src/androidTest/java/com/example/xqwen35bench/BenchmarkInstrumentationTest.java`

The added C ABI/JNI selftest entry emits `BENCH_VULKAN_SELFTEST_JSON` and writes a Device Farm artifact:

```text
bench_artifacts/vulkan_w4a16_selftest.json
```

### Runtime Integration Attempt

Updated files:

- `customlib/runtime/custom_model.hpp`
- `customlib/runtime/custom_model.cpp`
- `customlib/runtime/session.cpp`

The custom runtime attempts Vulkan W4A16 GEMV for packed W4 matrices when `backend = vulkan` or `backend = cpu_vulkan_hybrid`. Trace rows for projection families report `backend = vulkan` when the Vulkan GEMV path succeeds.

This integration is intentionally reported as hybrid, not full Vulkan. It uploads weights per GEMV call, so it is correctness-oriented evidence, not a performance-quality implementation.

## Build And Test Evidence

Host build and tests:

```text
results/reports/evidence/vulkan_final_host_build.txt
results/reports/evidence/vulkan_final_host_ctest.txt
```

Result:

```text
100% tests passed, 0 tests failed out of 2
```

Android build:

```text
results/reports/evidence/vulkan_final_android_build.txt
```

Result:

```text
BUILD SUCCESSFUL
```

## Device Farm Attempts

### Attempt 1: Vulkan W4A16 Selftest

- Run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/ac30385c-0fbb-4a1d-a897-ac870a1d8efc`
- Device Farm result: failed test-spec grep, because the first spec looked for the JSON in the wrong stream.
- Kernel artifact result: Vulkan selftest JSON passed.
- Evidence: `results/reports/evidence/vulkan_w4a16_selftest_attempt1.json`

The failure was in the wrapper/spec, not the shader math. The spec was fixed to read the artifact JSON.

### Attempt 2: Vulkan W4A16 Selftest, Fixed Spec

- Run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/da6dfbc9-27eb-4a42-9202-bffda6b6bbe6`
- Device Farm result: PASSED
- Evidence: `results/reports/evidence/vulkan_w4a16_selftest_attempt2_passed.json`

Key result from the passed selftest:

| Shape | CPU reference ms | Vulkan dispatch ms | Vulkan total ms | Max abs error |
| --- | ---: | ---: | ---: | ---: |
| 256 x 256 | 0.00787 | 0.46615 | 0.64829 | 4.17e-07 |
| 4096 x 4096 | 1.04010 | 8.75693 | 12.74010 | 6.85e-06 |

This proves a real Vulkan W4A16 GEMV kernel executed correctly on the target phone, but also shows the naive upload-per-call path is much slower than the CPU reference.

### Attempt 3: Full-Model Short Integration

- Run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/938721e9-46ab-4266-8e02-4445eb0704bd`
- Device Farm result: PASSED
- Evidence:
  - `results/reports/evidence/customlib_vulkan_integration_short_benchmark.json`
  - `results/reports/evidence/customlib_vulkan_integration_short_summary.json`
  - `results/reports/evidence/customlib_vulkan_integration_short_model_bootstrap.json`
  - `results/reports/evidence/customlib_vulkan_integration_short_model_bootstrap_discovery.txt`

Note: this short run was produced before the benchmark probe wording was corrected. Its raw `vulkan_attempt` subobject still contains stale text saying the Vulkan kernel was not enabled. The same raw JSON also contains the authoritative fields `custom_backend_actual = cpu_vulkan_hybrid`, `custom_path.vulkan_generation_kernels_used = true`, and per-kernel trace rows with `backend = vulkan`. The current source updates the probe wording so future runs report this as W4A16-linear Vulkan hybrid rather than "not enabled".

Run settings:

| Field | Value |
| --- | --- |
| Model | full Qwen3.5-9B package |
| Model SHA | `9de692be1c1ef1002fac25bd8f93c76e1d31975caa234fe1725f9eb294bfaa34` |
| Requested backend | `vulkan` |
| Actual backend | `cpu_vulkan_hybrid` |
| Prompt tokens | 1 |
| max_new_tokens | 1 |
| Warmup / measured | 0 / 1 |

Important measured fields:

| Field | Value |
| --- | --- |
| `use_mnn_fallback` | `0` |
| `calls_mnn_llm_response_for_measured_generation` | `false` |
| `vulkan_generation_kernels_used` | `true` |
| `fallback_op_families` | `[]` |
| Decode TPS | `0.173017` |
| Decode TPOT | `5779.77 ms` |

This short run is not a final benchmark. It exists to prove that the full-model measured path can enter Vulkan W4A16 projection kernels. It is not comparable to the official 512/256 v27 benchmark and it is far slower than stock MNN.

## What Actually Ran On Vulkan

In the short integration evidence, these op families were routed to Vulkan:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`
- `linear_attention_projections`

Representative Vulkan trace rows:

| Kernel | Backend | Calls | Total ms | Mean ms |
| --- | --- | ---: | ---: | ---: |
| `linear_attn_in_proj_qkv_w4a16` | vulkan | 48 | 1322.33 | 27.5486 |
| `linear_down_proj_w4a16` | vulkan | 64 | 2270.69 | 35.4796 |
| `linear_gate_proj_w4a16` | vulkan | 64 | 2509.98 | 39.2184 |
| `linear_q_proj_w4a16` | vulkan | 16 | 438.759 | 27.4225 |

## What Stayed On CPU

The following major measured op families stayed on CPU:

- RMSNorm
- RoPE
- grouped-query attention score/softmax/V-reduce
- KV append/read
- linear-attention recurrent state
- lm_head
- greedy sampling
- prefill KV build

Therefore this is not a full Vulkan generation path and cannot satisfy the requested `custom_backend_actual = vulkan` acceptance criteria.

## Quality Status

No accepted Vulkan quality validation exists for this attempt.

The short integration run emitted `BENCH_RESULT_JSON`, not `BENCH_QUALITY_JSON`. It used only one prompt token and one generated token, so it cannot prove output quality, exact-token match, repeated-token safety, or semantic plausibility.

The official quality-passing result remains the v27 CPU customlib result in:

- `results/reports/quality_validation_report.md`
- `results/reports/evidence/quality_validation_v27_english_comparison.json`

## Exact Blocker

The blocker is not Device Farm scheduling. Device Farm scheduling works. The blocker is implementation completeness and performance:

1. The current Vulkan implementation only covers W4A16 GEMV projection families.
2. Full generation still needs Vulkan implementations for RMSNorm, RoPE, attention, linear-attention state, lm_head, sampling, and prefill KV build.
3. The current Vulkan GEMV uploads weights/scales/zeros for every call instead of keeping the Qwen3.5 weights resident on GPU.
4. The selftest shows 4096 x 4096 W4A16 Vulkan total time `12.74010 ms` versus CPU reference `1.04010 ms`.
5. The short full-model hybrid run was much slower than CPU final numbers.
6. No Vulkan quality validation was completed.

## Final Claim Status

Allowed claims:

- A real custom Vulkan runtime was implemented.
- A real W4A16 GEMV Vulkan compute shader was implemented.
- The W4A16 Vulkan shader passed Device Farm correctness on Samsung Galaxy S26 Ultra.
- A full-model short custom generation run used Vulkan for W4A16 projection families.
- The measured custom generation path still did not call `MNN::Transformer::Llm::response`.

Non-claims:

- No full Vulkan custom generation claim.
- No custom Vulkan quality pass.
- No 10 TPS claim.
- No speedup over stock MNN.
- No production-quality Vulkan output claim.

## Next Required Work For Outcome A

To turn this into an accepted Vulkan final, the next implementation must:

- make Qwen3.5 W4A16 weights GPU-resident across decode
- add Vulkan RMSNorm, RoPE, attention, linear-attention state, lm_head, sampling, and prefill KV kernels
- remove CPU backends from major measured op families
- run Device Farm quality validation and emit `BENCH_QUALITY_JSON`
- run the full 512 prompt-token / 256 decode-token benchmark and emit `BENCH_RESULT_JSON`
- compare against stock MNN with same model/device/settings
