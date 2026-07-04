# Final Device Farm Report: Qwen3.5-9B MNN vs Custom Hotpath v14

## Honest Verdict

- Faithful benchmark: PARTIAL. Both runs use the same Device Farm device, same Qwen3.5-9B package, same prompt length, same generation settings, and the custom measured decode path executes custom W4A16/RMSNorm/RoPE kernels. It is not a fully faithful end-to-end Qwen implementation yet because attention, linear-attention recurrent state, lm_head, sampling, and prefill KV build remain explicit fallbacks/placeholders.
- Requirement 1 met: YES. The code walkthrough is in docs/kernel_library_code_walkthrough_v14.md, with kernel-library architecture, generated kernels, MNN connector, fallback connector, JNI connector, benchmark connector, and Device Farm connector.
- Requirement 2 met: YES for measured hotpath comparison. Stock MNN and customlib both ran on Device Farm on Qwen3.5-9B with identical prompt and generation settings; overall TPOT/TPS, custom per-kernel wall clock, and MNN hot-path trace tables are below.
- Custom speedup claim allowed: NO. Custom decode TPS is 0.2477x stock, and custom TPOT is 4.04x stock TPOT, so this result must be described as a functional custom-hotpath bring-up, not a speedup.

## Device Farm Evidence

- Device: Samsung Galaxy S26 Ultra.
- Device ARN: arn:aws:devicefarm:us-west-2::device:536B9FDAEAA14A11B504A3ECC86DA717.
- Device pool ARN: arn:aws:devicefarm:us-west-2:884244642857:devicepool:64d2cc31-abd6-49f8-97da-162f82410bc0/14d31c96-b8fc-4930-99c7-1a8948124213.
- Stock MNN run ARN: arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/a2f55f2f-b13f-4abf-8362-79810afd58ef; result PASSED; instrumentation output contains OK (1 test): True.
- Custom hotpath run ARN: arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/c3353a1b-56f6-4a2e-87ab-f4bd20b3416a; result PASSED; instrumentation output contains OK (1 test): True.
- Model package: Qwen/Qwen3.5-9B, revision c202236235762e1c871ad0ccb60c8ee5ba337b9a, quantization w4a16_groupwise (4-bit), group size 64.
- Model artifact: split Device Farm uploads, 3 parts: 4,500,000,000 + 4,500,000,000 + 1,623,583,117 bytes = 10,623,583,117 bytes.
- Full model zip sha256 verified on-device: ad51ea82e015341c76a48d1ad14b2740787c240624079c48067b2dc42bd85879.
- Bootstrap evidence: both model_bootstrap_discovery.txt files show all 3 parts downloaded, reassembled, sha256 checked, unzipped, and source marked downloaded.

## Same-Settings Comparison

| Field | Stock MNN | Custom hotpath |
|---|---|---|
| Device | Samsung Galaxy S26 Ultra | Samsung Galaxy S26 Ultra |
| Prompt tokens requested | 512 | 512 |
| max_new_tokens | 256 | 256 |
| Prompt token id | 16 | 16 |
| Temperature / top_k / top_p | 0 / 1 / 1 | 0 / 1 / 1 |
| Warmup / measured iterations | 1 / 5 | 1 / 5 |
| Backend | cpu | cpu |

| Metric | Stock MNN median | Custom hotpath median | Custom / Stock |
|---|---:|---:|---:|
| Prefill TPS | 45.6910 | 2,222,570.0000 | not comparable |
| Decode TPS | 2.266760 | 0.561446 | 0.2477x |
| Decode TPOT ms | 441.158 | 1,781.120 | 4.04x slower |

Note: custom prefill TPS is emitted for schema completeness but is not an apples-to-apples prefill result in v14 because prefill/KV build is still listed as fallback.

## Custom Path Coverage

Selected kernels from Device Farm custom JSON:

~~~text
custom_decode_loop;hotpath_replaced=true;linear=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,linear_attn_qkv_z_out;rmsnorm=custom;rope=custom;fallback_ops=attention,linear_attention_state,lm_head,sampling,prefill_kv
~~~

- hotpath_replaced: True
- Replaced op families: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, rmsnorm, rope, linear_attn_qkv_z_out
- Fallback op families: attention, linear_attention_state, lm_head, sampling, prefill_kv_build
- calls_mnn_llm_response_for_measured_generation: False

## Custom Per-Kernel Wall Clock

| Kernel / op family | Calls | Total ms | Mean ms |
|---|---:|---:|---:|
| linear_up_proj_w4a16 | 8192 | 107,642.000 | 13.139900 |
| linear_gate_proj_w4a16 | 8192 | 107,624.000 | 13.137700 |
| linear_down_proj_w4a16 | 8192 | 100,680.000 | 12.290000 |
| linear_attn_in_proj_qkv_w4a16 | 6144 | 53,843.700 | 8.763620 |
| linear_attn_out_proj_w4a16 | 6144 | 26,928.700 | 4.382920 |
| linear_attn_in_proj_z_w4a16 | 6144 | 26,916.300 | 4.380900 |
| linear_q_proj_w4a16 | 2048 | 17,948.000 | 8.763660 |
| linear_o_proj_w4a16 | 2048 | 8,966.360 | 4.378110 |
| linear_v_proj_w4a16 | 2048 | 2,245.570 | 1.096470 |
| linear_k_proj_w4a16 | 2048 | 2,244.850 | 1.096120 |
| silu_gate_mul_custom | 8192 | 348.247 | 0.042511 |
| fallback_linear_attention_state | 6144 | 289.081 | 0.047051 |
| rope_qk_custom | 2048 | 91.781 | 0.044815 |
| rmsnorm_input | 8192 | 37.991 | 0.004638 |
| rmsnorm_post_attention | 8192 | 36.420 | 0.004446 |
| fallback_attention_passthrough | 2048 | 7.456 | 0.003641 |
| rmsnorm_final | 256 | 1.251 | 0.004887 |
| embedding_bf16_row_read | 1 | 0.211 | 0.211302 |
| fallback_lm_head_sampling_hash | 256 | 0.195 | 0.000763 |

## MNN Hot-Path Trace: Top Op Types

MNN trace was collected in the custom app as a separate debug-callback run with prompt 512 and max_new_tokens 16. It is evidence for stock MNN hot-path op timing, not part of measured custom generation.

| Stage | Type | Calls | Total ms | Mean ms |
|---|---|---:|---:|---:|
| prefill_or_first_token_inferred | Convolution | 249 | 9,615.360 | 38.615900 |
| decode_inferred | Convolution | 3984 | 4,384.870 | 1.100620 |
| decode_inferred | UnaryOp | 2464 | 1,349.510 | 0.547689 |
| prefill_or_first_token_inferred | Raster | 521 | 483.003 | 0.927070 |
| decode_inferred | LinearAttention | 384 | 447.735 | 1.165980 |
| decode_inferred | LayerNorm | 1680 | 354.890 | 0.211244 |
| prefill_or_first_token_inferred | LinearAttention | 24 | 305.893 | 12.745600 |
| decode_inferred | While | 528 | 274.678 | 0.520224 |
| prefill_or_first_token_inferred | UnaryOp | 154 | 240.915 | 1.564390 |
| prefill_or_first_token_inferred | BinaryOp | 169 | 118.169 | 0.699226 |
| prefill_or_first_token_inferred | Attention | 8 | 112.038 | 14.004800 |
| decode_inferred | Attention | 128 | 96.649 | 0.755067 |

## MNN Hot-Path Trace: Top Op Names

| Stage | Type | Name | Calls | Total ms | Mean ms |
|---|---|---|---:|---:|---:|
| decode_inferred | Convolution | /lm/lm_head/Linear | 16 | 243.469 | 15.216800 |
| prefill_or_first_token_inferred | Convolution | /layers.21/mlp/down_proj/Linear | 1 | 88.302 | 88.302200 |
| prefill_or_first_token_inferred | Convolution | /layers.30/mlp/down_proj/Linear | 1 | 85.303 | 85.303200 |
| prefill_or_first_token_inferred | Convolution | /layers.22/mlp/down_proj/Linear | 1 | 84.706 | 84.706200 |
| prefill_or_first_token_inferred | Convolution | /layers.10/mlp/down_proj/Linear | 1 | 84.645 | 84.645400 |
| prefill_or_first_token_inferred | Convolution | /layers.25/mlp/down_proj/Linear | 1 | 83.670 | 83.669500 |
| prefill_or_first_token_inferred | Convolution | /layers.27/mlp/gate_proj/Linear | 1 | 82.774 | 82.774300 |
| prefill_or_first_token_inferred | Convolution | /layers.4/mlp/gate_proj/Linear | 1 | 82.138 | 82.138300 |
| prefill_or_first_token_inferred | Convolution | /layers.26/mlp/gate_proj/Linear | 1 | 80.576 | 80.575800 |
| prefill_or_first_token_inferred | Convolution | /layers.13/mlp/up_proj/Linear | 1 | 80.549 | 80.548700 |
| prefill_or_first_token_inferred | Convolution | /layers.6/mlp/up_proj/Linear | 1 | 80.548 | 80.547500 |
| prefill_or_first_token_inferred | Convolution | /layers.10/mlp/up_proj/Linear | 1 | 80.526 | 80.525900 |

## Kernel Library Code Walkthrough Pointers

- Architecture: customlib/runtime/session.cpp selects CustomModel when use_mnn_fallback is 0; CustomModel owns the decode loop, packed weights, kernel trace, and explicit fallback accounting.
- Generated W4A16 kernels: customlib/kernels/generated/arm64_neon/xq_gemv_w4a16_neon.cpp implements fused int4 unpack + groupwise dequant + tiled GEMV for Qwen3.5 shapes including 4096x4096, 12288x4096, 4096x12288, and k/v projection shapes. A repository scan of this W4 file finds no gemvLowBitReference call.
- RMSNorm and RoPE: customlib/runtime/custom_model.cpp runs custom RMSNorm and custom RoPE in the measured decode loop before/around q/k/v/o and MLP linears.
- MNN connector: android/app/src/main/cpp/stock_benchmark_jni.cpp is the stock MNN baseline and still uses MNN LLM for stock comparison.
- Fallback connector: attention, linear-attention recurrent state, lm_head, sampling, and prefill_kv_build remain explicit fallback families and are present in selected_kernels and per-kernel trace.
- JNI/benchmark connector: android/benchmark_app/src/main/cpp/benchmark_jni.cpp emits BENCH_RESULT_JSON with selected_kernels, per-kernel wall clock, overall TPS/TPOT, ratio inputs, and MNN trace evidence.
- Device Farm connector: Android instrumentation ModelBootstrap downloads the split model parts, verifies the full sha256, unzips the package, runs the same settings, and pulls benchmark artifacts.

## Minimum Remaining Work

- Replace the fallback attention/linear-attention recurrent state with a real KV-cache attention implementation.
- Replace lm_head and sampling fallback with real logits projection and sampling/top-k path.
- Replace placeholder/fallback prefill KV build with a real prefill path so prefill TPS becomes comparable.
- Optimize W4A16 GEMV further; current generated kernel is real and tiled, but Device Farm decode speed is 0.2477x stock MNN, so no speedup claim is allowed.


