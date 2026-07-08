# Kernel Library Code Walkthrough Final

This document describes the accepted v27 custom inference library used for the Qwen3.5-9B AWS Device Farm quality and performance final. It matches the measured code path and evidence JSON, not a planned architecture.

## Final Measured Path

The final custom Android benchmark sets `xq_options.use_mnn_fallback = 0` in `android/benchmark_app/src/main/cpp/benchmark_jni.cpp` and enters the public ABI in `customlib/include/xqwen35.h`. The final run requested `cpu_vulkan_hybrid`, but the benchmark JSON reports `custom_backend_actual = cpu`; no custom Vulkan kernel is claimed.

Post-v27 note: a later Vulkan attempt added `customlib/runtime/vulkan_backend.*` and `customlib/kernels/generated/vulkan/w4a16_gemv.comp`, then validated a real W4A16 GEMV shader on Device Farm. A short full-model integration routed projection-family W4A16 rows to Vulkan, but the actual measured backend was `cpu_vulkan_hybrid` and major op families remained CPU. That attempt is documented in `results/reports/final_vulkan_blocker_report.md`; it does not replace the accepted v27 CPU final.

Measured generation path:

```text
NativeBenchmark.runBenchmark
  -> xq_create
  -> xq_generate
     -> xq::runtime::Session::prefill
        -> xq::runtime::CustomModel::prefill
           -> load embedding for every prompt token
           -> run full layer stack for each prompt position
           -> append full-attention K/V cache
           -> update linear-attention recurrent state
     -> xq::runtime::Session::decodeOne
        -> xq::runtime::CustomModel::decodeOne
           -> run custom layer stack
           -> run real lm_head logits
           -> run greedy sampling
```

Final evidence:

- `custom_path.use_mnn_fallback = 0`
- `custom_path.calls_mnn_llm_response_for_measured_generation = false`
- `runtime.selected_kernels.full_custom_decode = true`
- `runtime.selected_kernels.hotpath_replaced = true`
- `runtime.selected_kernels.fallback_op_families = []`

The optional MNN fallback connector still exists for debug compatibility, but it is not used by the measured custom benchmark.

## Repository Architecture

| Area | Files | Role |
| --- | --- | --- |
| Public ABI | `customlib/include/xqwen35.h` | Stable C ABI for Android/JNI and host tests. |
| Session connector | `customlib/runtime/session.cpp` | Chooses MNN fallback only when requested; final custom run loads `CustomModel`. |
| Custom runtime | `customlib/runtime/custom_model.cpp` | Implements prefill, decode, attention, linear attention, lm_head, sampling, and trace recording. |
| Generated kernels | `customlib/kernels/generated/arm64_neon/xq_gemv_w4a16_neon.cpp` | Fused dequant + GEMV W4A16 kernel used by decode linear families and streaming lm_head argmax. |
| Reference kernels | `customlib/kernels/reference/*.cpp` | Host correctness references and non-final low-bit experiments. |
| Packer | `customlib/packer/pack_qwen35_xq4.py`, `customlib/packer/pack_qwen35_xq4_numpy.py` | Converts Qwen3.5 safetensors into the custom W4A16 package and manifest. |
| Stock Android app | `android/app` | Stock MNN baseline and stock quality instrumentation. |
| Custom Android app | `android/benchmark_app` | Customlib instrumentation, JSON evidence, MNN hot-path trace evidence, custom quality instrumentation. |
| Vulkan probe | `android/benchmark_app/src/main/cpp/benchmark_jni.cpp` | Probes Vulkan availability and records whether Vulkan kernels were actually used. |
| Device Farm connector | `scripts/aws` plus Android bootstrap tests | Uploads, schedules, downloads, and verifies Device Farm artifacts. |
| Quality connector | `runQualityValidation` JNI methods and `scripts/quality/compare_quality_validation.py` | Dumps generated token IDs and compares stock/custom output sanity separately from TPOT/TPS measurement. |

## Generated W4A16 GEMV

The final generated W4 path is `customlib/kernels/generated/arm64_neon/xq_gemv_w4a16_neon.cpp`. It is a generated/tiled fused dequant + GEMV implementation and does not call `gemvLowBitReference`.

For each output row and quantization group, it computes:

```text
code_dot = sum(code_i * activation_i)
x_sum    = sum(activation_i)
partial  = scale * (code_dot - zero * x_sum)       # symmetric v1 package
partial  = scale * code_dot + zero * x_sum         # affine-asymmetric v2 package
```

The arm64 path unpacks int4 nibbles, widens to float, accumulates with NEON FMA, and uses persistent worker threads for large row-parallel GEMV. It precomputes activation group sums once per input and reuses those sums across rows. The v27 optimization branches once per matrix on symmetric versus affine-asymmetric dequant, so the final symmetric Qwen package avoids a per-row/per-group branch. `gemvW4A16ArgmaxNeon` streams the `lm_head` argmax and avoids materializing full logits for greedy sampling.

Qwen3.5 shapes exercised on Device Farm include 4096 x 4096 attention/output projections, 12288 x 4096 MLP gate/up projections, 4096 x 12288 MLP down projection, grouped key/value projection shapes, and the 248320 x 4096 `lm_head`.

## Custom Prefill

`CustomModel::prefill` runs over every prompt token. It is not a last-token-only prefill.

For each prompt position it:

- reads the BF16 embedding row for that token through a persistent embedding stream
- runs every layer in order
- appends K/V into full-attention layer caches
- updates linear-attention recurrent state in linear-attention layers
- records `prefill_token_custom`
- records `prefill_kv_build_custom` during full-attention prefill K/V append

v27 reserves KV cache capacity up front and reuses embedding/attention buffers to reduce allocation overhead.

## Custom Full-Attention Decode

`CustomModel::runFullAttention` implements grouped-query decode attention:

1. Apply input RMSNorm.
2. Run custom W4A16 `q_proj`, `k_proj`, and `v_proj`.
3. Apply Q/K norm where Qwen3.5 supplies q_norm/k_norm weights.
4. Apply Qwen3.5 active-slice RoPE at the current absolute position.
5. Append K/V to the per-layer cache.
6. Map each query head to the correct KV head.
7. Compute `score = dot(q, k) / sqrt(head_dim)`.
8. Run max-subtracted softmax.
9. Reduce with V.
10. Apply attention output gate.
11. Run custom W4A16 `o_proj`.

Trace rows include `linear_q_proj_w4a16`, `linear_k_proj_w4a16`, `linear_v_proj_w4a16`, `rope_qk_custom`, `kv_append_custom`, `attention_score_custom`, `attention_softmax_custom`, `attention_v_reduce_custom`, `attention_output_gate_custom`, and `linear_o_proj_w4a16`.

## Custom Linear-Attention State

`CustomModel::runLinearAttention` maintains recurrent state per linear-attention layer across prefill and decode. It uses exported Qwen3.5 tensors:

- `linear_attn_in_proj_a_weight.xq4`
- `linear_attn_in_proj_b_weight.xq4`
- `linear_attn_in_proj_qkv_weight.xq4`
- `linear_attn_in_proj_z_weight.xq4`
- `linear_attn_out_proj_weight.xq4`
- `linear_attn_conv1d_weight.f32`
- `linear_attn_A_log.f32`
- `linear_attn_dt_bias.f32`

The v27 path uses the exported `A_log` and `dt_bias` gate update instead of the rejected approximation path. Trace rows include `linear_attention_conv1d_custom`, `linear_attention_qk_l2norm_custom`, `linear_attention_state_update_custom`, `linear_attention_gated_rmsnorm_custom`, and `linear_attn_out_proj_w4a16`.

## FFN, RMSNorm, And RoPE

The runtime applies Qwen-style norm weight handling by adding one to exported norm vectors before use. It implements RMSNorm, active-slice RoPE, and FFN gate math directly in `custom_model.cpp`:

```text
post_attention_rmsnorm
gate_proj W4A16
up_proj W4A16
silu(gate) * up
down_proj W4A16
residual add
```

Trace rows include `rmsnorm_input`, `rmsnorm_post_attention`, `rmsnorm_final`, `linear_gate_proj_w4a16`, `linear_up_proj_w4a16`, `silu_gate_mul_custom`, and `linear_down_proj_w4a16`.

## Real lm_head And Greedy Sampling

`CustomModel::sampleGreedy` removes hash-based token generation:

1. Apply final RMSNorm.
2. Run `lm_head_custom` with packed W4A16 `lm_head.weight`.
3. Compute greedy argmax over real logits using streaming W4A16 argmax.
4. Return the selected token.

The final benchmark settings use `temperature = 0`, `top_k = 1`, and `top_p = 1`, so greedy argmax is the required behavior. Trace rows include `lm_head_custom` and `sampling_greedy_custom`.

## Connectors

### MNN Baseline Connector

`android/app/src/main/cpp/stock_benchmark_jni.cpp` creates `MNN::Transformer::Llm`, configures stock MNN CPU, runs generation, and emits stock `BENCH_RESULT_JSON`. It also implements `runQualityValidation` for stock `BENCH_QUALITY_JSON` using the same English fixed token-ID prompts.

### Custom JNI Connector

`android/benchmark_app/src/main/cpp/benchmark_jni.cpp` exposes `NativeBenchmark.runBenchmark(...)` and `NativeBenchmark.runQualityValidation(...)`. The benchmark path emits TPOT/TPS and per-kernel wall clock. The quality path emits generated token IDs for fixed prompts and does not pollute TPOT/TPS timing.

### Fallback Connector

`customlib/runtime/session.cpp` still supports `xq_options.use_mnn_fallback != 0` for debugging. The final custom Device Farm run does not use it.

### Vulkan Probe Connector

The custom JNI accepts `cpu`, `vulkan`, and `cpu_vulkan_hybrid`. It probes `libvulkan.so` and `vkGetInstanceProcAddr`. The final v27 run reports probe success but `custom_backend_actual = cpu` and `vulkan_generation_kernels_used = false`.

### Benchmark And Quality Connectors

Android instrumentation writes JSON artifacts into Device Farm customer artifacts:

- `stock_mnn_cpu_benchmark.json`
- `customlib_cpu_vulkan_hybrid_benchmark.json`
- `quality_validation_stock.json`
- `quality_validation_custom.json`

`scripts/quality/compare_quality_validation.py` compares stock/custom quality JSON and produces `quality_validation_v27_english_comparison.json` plus `quality_validation_report.md`.

## Evidence Summary

- Final custom benchmark ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/c444a573-d0e1-4924-abe2-5ca587478c2c`
- Final stock benchmark ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/5165cf70-4ad2-4f21-bd40-89c6dd97c246`
- Final custom quality ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/4ac8ce83-abf1-42fe-8cd2-3ffde19821f2`
- Final stock quality ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/9908fb50-103c-4698-81c9-f2f12b98b335`
- Custom decode TPS / TPOT: `2.11989` / `471.723 ms`
- Stock decode TPS / TPOT: `2.26870` / `440.780 ms`
- Custom / stock decode ratio: `0.9344x`
- Quality gate: PASS, exact full-output token match `1 / 5`, comparison sanity `5 / 5`.

The final result is a systems/kernel benchmark plus deterministic output sanity guard. It is not a production-quality language-model certification.
