# Final Vulkan Delivery Attempt: Blocker Report

## Verdict

Outcome B: Vulkan blocked after real implementation attempts.

This pass did not stop at auditing the empty Vulkan folder. It added a custom Vulkan runtime, implemented real Vulkan compute shaders for W4A16 GEMV and the major vector/tensor decode kernels, validated those shaders on AWS Device Farm Samsung Galaxy S26 Ultra, integrated them into measured custom Qwen3.5 generation, ran quality validation, and attempted the required 512 prompt-token / 256 decode-token full benchmark.

The result is still not accepted as the requested final Vulkan delivery because the official full benchmark did not complete and the runtime still reports a hybrid backend rather than full Vulkan:

- `custom_backend_requested = vulkan` was exercised in full-model short integration, quality, and full-benchmark attempts.
- `custom_backend_actual = cpu_vulkan_hybrid`, not `vulkan`.
- `vulkan_generation_kernels_used = true`.
- The latest short integration trace reports Vulkan for q/k/v/o, gate/up/down, linear-attention projections, RMSNorm, RoPE, attention, linear-attention state, activation, lm_head, sampling, and prefill KV build.
- A Device Farm quality run emitted `BENCH_QUALITY_JSON` and passed the deterministic sanity gate, but exact full-output token match was only 1 / 5 and production-quality output is not claimed.
- The full 512-token / 256-token benchmark entered custom generation and completed one warmup iteration, but Device Farm stopped it at the 150-minute job limit before the measured iterations and before final `BENCH_RESULT_JSON`.
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
- `customlib/kernels/generated/vulkan/vector_ops.comp`
- `customlib/kernels/generated/vulkan/vector_ops_spv.inc`

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
results/reports/evidence/vulkan_linear_state_final_host_build.txt
results/reports/evidence/vulkan_linear_state_final_host_ctest.txt
```

Result:

```text
100% tests passed, 0 tests failed out of 2
```

Android build:

```text
results/reports/evidence/vulkan_final_android_build.txt
results/reports/evidence/vulkan_linear_state_final_android_build.txt
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

### Extended Vulkan Vector/Tensor Kernels

The later Vulkan pass added shader paths for the non-linear decode families that were missing from the first W4A16-only attempt:

- RMSNorm and Q/K normalization
- Qwen3.5 active-slice RoPE
- grouped-query attention score / stable softmax / V reduce
- linear-attention conv1d and recurrent state update
- KV append
- SiLU gate multiply and residual add
- attention output gate
- greedy argmax

Device Farm selftest run:

```text
arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/f3506fac-58d8-4b03-bdf1-5f81d24af6ae
```

Evidence:

- `results/reports/evidence/vulkan_linear_state_selftest_passed.json`

All vector/tensor selftest rows passed:

| Selftest row | Backend | Max abs error | Verdict |
| --- | --- | ---: | --- |
| `rmsnorm_vector` | vulkan | 0 | PASS |
| `rope_qk_vector` | vulkan | 7.45e-09 | PASS |
| `silu_gate_mul_vector` | vulkan | 2.79e-09 | PASS |
| `residual_add_vector` | vulkan | 0 | PASS |
| `gqa_decode_vector` | vulkan | 2.24e-08 | PASS |
| `linear_attention_conv1d_vector` | vulkan | 3.73e-09 | PASS |
| `linear_attention_state_update_vector` | vulkan | 1.49e-08 | PASS |
| `append_vector` | vulkan | 0 | PASS |
| `argmax_vector` | vulkan | 0 | PASS |

The first linear-attention state shader attempt failed tolerance because it computed the prediction from the pre-decay state. The fix moved the decay into the prediction term to match the CPU reference. That is why this report counts the final state-update shader as a real corrected implementation attempt, not a label-only backend change.

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

### Attempt 4: Extended Full-Model Short Integration

- Run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/e6c5ec8c-7f52-4164-bc7f-c64f5e6105d5`
- Device Farm result: PASSED
- Evidence: `results/reports/evidence/customlib_vulkan_linear_state_integration_short_benchmark.json`

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
| Decode TPS | `0.139909` |
| Decode TPOT | `7147.49 ms` |

The op-family backend map in this run reports Vulkan for the major tensor families:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj,
linear_attention_projections,
rmsnorm, rope, attention,
linear_attention_state, activation,
lm_head, sampling, prefill_kv_build
```

The non-Vulkan trace row was `embedding_bf16_row_read` with `backend = cpu`. Because the benchmark still reports `custom_backend_actual = cpu_vulkan_hybrid`, this evidence is not treated as a full Vulkan final.

### Attempt 5: Device Farm Quality Validation

First quality attempt:

```text
arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/01cb5b7b-7cc8-476a-8a99-f47a1f9711b0
```

Result: failed before quality JSON because the test spec used expired model presigned URLs and model download returned HTTP 403. This was an infrastructure/spec issue, not a model-output result.

Retry quality run:

```text
arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/e7081f14-99c2-4943-b6f9-222f8501fb90
```

Evidence:

- `results/reports/evidence/quality_validation_custom_vulkan_linear_state_english.json`
- `results/reports/evidence/quality_validation_vulkan_linear_state_english_comparison.json`
- `results/reports/quality_validation_vulkan_linear_state_report.md`

Quality result:

| Metric | Value |
| --- | --- |
| Native quality JSON | emitted |
| Quality gate | PASS |
| Prompt suite | 5 English fixed-token prompts |
| Comparison-gate prompts | 5 / 5 |
| Exact full-token matches | 1 / 5 |
| Invalid token IDs | none |
| Empty outputs | none |
| Repeated-token-220 failure | none |
| Degenerate repetition | none |

This is a deterministic sanity pass, not a production semantic-quality claim and not token-level equivalence.

### Attempt 6: Required Full 512/256 Benchmark

- Run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/e6650fcf-589f-49f6-8033-ed2030873d95`
- Device Farm status/result: `COMPLETED` / `STOPPED`
- Job timeout: 150 minutes
- Evidence: `results/reports/evidence/customlib_vulkan_full_benchmark_stopped.json`

Required benchmark settings were used:

| Field | Value |
| --- | --- |
| Prompt tokens | 512 |
| max_new_tokens | 256 |
| Warmup / measured | 1 / 3 |
| Requested backend | `vulkan` |
| Actual backend observed in logcat | `cpu_vulkan_hybrid` |

Logcat evidence:

```text
16:50:30 BENCH_START engine=customlib requested_backend=vulkan actual_backend=cpu_vulkan_hybrid
18:04:35 BENCH_ITER engine=customlib warmup=1 prompt=512 generated=256 prefill_ms=2840779.534 decode_ms=1599572.239 status=0
```

Derived from the completed warmup only:

| Partial warmup metric | Value |
| --- | ---: |
| Prefill time | 47.346 min |
| Decode time | 26.660 min |
| Decode TPS | 0.160043 |
| Decode TPOT | 6248.329 ms |

No final `BENCH_RESULT_JSON` was emitted because no measured iteration completed before Device Farm stopped the run. This is the hard blocker for accepting a Vulkan performance final.

## What Actually Ran On Vulkan

In the latest short integration evidence, these op families were routed to Vulkan:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`
- `linear_attention_projections`
- `rmsnorm`
- `rope`
- `attention`
- `linear_attention_state`
- `activation`
- `lm_head`
- `sampling`
- `prefill_kv_build`

Representative Vulkan trace rows:

| Kernel | Op family | Backend | Calls | Total ms | Mean ms |
| --- | --- | --- | ---: | ---: | ---: |
| `linear_q_proj_w4a16` | q_proj | vulkan | 16 | 469.258 | 29.3286 |
| `rmsnorm_input` | rmsnorm | vulkan | 64 | 32.313 | 0.504891 |
| `rope_qk_custom` | rope | vulkan | 16 | 8.56464 | 0.535290 |
| `attention_gqa_decode_custom` | attention | vulkan | 16 | 8.72990 | 0.545619 |
| `linear_attention_state_update_custom` | linear_attention_state | vulkan | 48 | 175.775 | 3.66198 |
| `kv_append_custom` | prefill_kv_build | vulkan | 16 | 15.0247 | 0.939043 |
| `lm_head_custom` | lm_head | vulkan | 1 | 1583.25 | 1583.25 |
| `sampling_greedy_custom` | sampling | vulkan | 1 | 1583.26 | 1583.26 |
| `embedding_bf16_row_read` | other | cpu | 2 | 2.81807 | 1.40904 |

## Why This Is Still Not Full Vulkan

The latest short integration no longer has the same missing-op-family problem as the first W4A16-only attempt; the major tensor families above report `backend = vulkan` in the trace. It is still not accepted as full Vulkan for three reasons:

1. The benchmark reports `custom_backend_actual = cpu_vulkan_hybrid`, not `vulkan`.
2. The runtime still uses CPU-side embedding/file I/O and CPU control around the Vulkan dispatches.
3. The implementation uses host-visible upload/download dispatches rather than a persistent GPU-resident Qwen3.5 execution graph, so the official full 512/256 Device Farm run timed out before measured iterations.

Therefore this cannot satisfy the requested `custom_backend_actual = vulkan` acceptance criteria, even though the short integration proves real Vulkan kernels executed inside the measured custom generation path.

## Quality Status

Vulkan/hybrid quality validation exists and passed the deterministic sanity gate:

- Run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/e7081f14-99c2-4943-b6f9-222f8501fb90`
- Evidence: `results/reports/evidence/quality_validation_vulkan_linear_state_english_comparison.json`
- `BENCH_QUALITY_JSON`: emitted
- Prompt sanity: 5 / 5
- Exact full-output token match: 1 / 5
- Invalid/empty/degenerate output: none
- Repeated token 220 failure: none

This is useful quality evidence, but it does not replace the final v27 CPU customlib result because the required full performance benchmark did not complete. It is also not a production-quality semantic benchmark.

## Exact Blocker

The blocker is not Device Farm scheduling. Device Farm scheduling works. The blocker is the current Vulkan runtime architecture and performance:

1. The runtime reports `custom_backend_actual = cpu_vulkan_hybrid`, not `vulkan`.
2. CPU-side embedding/file I/O and CPU control remain around the Vulkan kernels.
3. The current Vulkan GEMV and vector paths allocate/upload/dispatch/download through host-visible buffers per operation instead of keeping the full Qwen3.5 model, activations, KV cache, and recurrent state resident on GPU.
4. The W4A16 selftest shows 4096 x 4096 Vulkan total time `12.0972 ms` versus CPU reference `1.09781 ms`.
5. The latest short full-model integration decode TPS was `0.139909`, far below the v27 CPU custom result.
6. The required full benchmark completed one warmup only: prefill `2840779.534 ms`, decode `1599572.239 ms`, then Device Farm stopped the run at 150 minutes before measured iterations and before final `BENCH_RESULT_JSON`.

## Final Claim Status

Allowed claims:

- A real custom Vulkan runtime was implemented.
- A real W4A16 GEMV Vulkan compute shader was implemented.
- Real Vulkan vector/tensor kernels were implemented for RMSNorm, RoPE, GQA decode, linear-attention conv/state update, KV append, activation/residual, output gate, and argmax.
- Vulkan W4A16 and vector/tensor selftests passed Device Farm correctness on Samsung Galaxy S26 Ultra.
- A full-model short custom generation run used Vulkan for the major traced tensor op families.
- Vulkan/hybrid quality validation emitted `BENCH_QUALITY_JSON` and passed the deterministic sanity gate.
- The measured custom generation path still did not call `MNN::Transformer::Llm::response`.

Non-claims:

- No full Vulkan custom generation claim.
- No accepted full 512/256 Vulkan benchmark result.
- No 10 TPS claim.
- No speedup over stock MNN.
- No token-equivalence or production-quality Vulkan output claim.

## Next Required Work For Outcome A

To turn this into an accepted Vulkan final, the next implementation must:

- make Qwen3.5 W4A16 weights GPU-resident across decode
- make activations, KV cache, recurrent state, and lm_head argmax GPU-resident across decode
- remove CPU-side embedding/file I/O and CPU control from the measured major path, or explicitly report the result as hybrid
- run the full 512 prompt-token / 256 decode-token benchmark and emit `BENCH_RESULT_JSON`
- compare against stock MNN with same model/device/settings
