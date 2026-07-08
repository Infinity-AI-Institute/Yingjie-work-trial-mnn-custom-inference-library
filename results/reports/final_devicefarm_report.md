# Final Device Farm Report: Qwen3.5-9B Stock MNN vs Customlib v27

## Honest Verdict

- Faithful benchmark: YES. The final accepted runs are full Qwen3.5-9B Device Farm runs on the same Samsung Galaxy S26 Ultra pool, same verified model package, same 512-token prompt length, same 256-token decode length, greedy settings, and 1 warmup / 3 measured iterations.
- Requirement 1 met: YES. The custom kernel library exists and the code walkthrough is `docs/kernel_library_code_walkthrough_final.md`.
- Requirement 2 met: YES. This report compares generated custom kernels against stock MNN with per-kernel wall clock, MNN hot-path trace, overall TPOT/TPS, Device Farm ARNs, and evidence JSON.
- Requirement 3, output quality guard: YES for deterministic sanity validation. Stock MNN and customlib both emitted `BENCH_QUALITY_JSON`; 5 / 5 prompts passed the comparison sanity gate, with 1 / 5 exact full-output token matches. No production-level semantic quality claim is made.
- Custom speedup claim allowed: NO. Final custom decode TPS is `2.11989` and fresh stock MNN CPU decode TPS is `2.26870`, so custom is `0.9344x` stock CPU. The accepted v27 custom path is faster than v17 by `9.60%` but still slower than stock MNN CPU.
- Vulkan attempted: YES. The custom run requested `cpu_vulkan_hybrid` and probed Vulkan successfully, but `custom_backend_actual = cpu` and no custom Vulkan generation kernels were used. Stock Vulkan remains a prior attempted backend that crashed before benchmark JSON in v17 and is not used for the final speed comparison.

## Post-v27 Vulkan Implementation Attempt

After v27, a separate Vulkan pass added a real custom Vulkan runtime, W4A16 GEMV shader, vector/tensor shaders, grouped-query attention shader, linear-attention state shader, KV append shader, and argmax shader. Those kernels passed Device Farm selftests on Samsung Galaxy S26 Ultra. A short full-model custom integration run routed the major traced tensor op families through Vulkan, and a separate quality-validation run emitted `BENCH_QUALITY_JSON` and passed the deterministic sanity gate.

That later attempt does not replace this final report:

- Device Farm extended Vulkan selftest PASS run: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/f3506fac-58d8-4b03-bdf1-5f81d24af6ae`
- Short full-model hybrid run: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/e6c5ec8c-7f52-4164-bc7f-c64f5e6105d5`
- Short run backend: `custom_backend_requested = vulkan`, `custom_backend_actual = cpu_vulkan_hybrid`
- Vulkan families in short integration trace: q/k/v/o, gate/up/down, linear-attention projections, RMSNorm, RoPE, attention, linear-attention state, activation, lm_head, sampling, and prefill KV build
- CPU rows remaining in short trace: embedding BF16 row read / file I/O and CPU control; actual backend remains hybrid
- Quality status: Vulkan/hybrid quality `BENCH_QUALITY_JSON` emitted; sanity gate passed, exact full-output match 1 / 5, no production-quality semantic claim
- Required full benchmark run: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/e6650fcf-589f-49f6-8033-ed2030873d95`, result `STOPPED` at 150 minutes before measured iterations and before final `BENCH_RESULT_JSON`
- Speed status: no 10 TPS claim and no Vulkan speedup claim

See `results/reports/final_vulkan_blocker_report.md` and `results/reports/vulkan_iteration_log.md` for the implementation attempt and blocker evidence.

## Final Acceptance Summary

| Gate | Result | Evidence |
| --- | --- | --- |
| Full custom path | PASS | `use_mnn_fallback = 0`, `calls_mnn_llm_response_for_measured_generation = false`, `full_custom_decode = true`, `fallback_op_families = []` in `customlib_cpu_vulkan_hybrid_benchmark_v27.json` |
| Output quality | PASS | `quality_validation_v27_english_comparison.json`, 5 / 5 comparison-gate prompts, no invalid/empty/degenerate outputs |
| Full Device Farm benchmark | PASS | custom v27 run `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/c444a573-d0e1-4924-abe2-5ca587478c2c` emitted `BENCH_RESULT_JSON` |
| Performance versus v17 | PASS | custom v27 decode TPS `2.11989` versus v17 `1.93417`, ratio `1.0960x` |
| Performance versus fresh stock | NO SPEEDUP | custom v27 / stock v27 decode ratio `0.9344x` |

## Device Farm Evidence

- Project ARN: `arn:aws:devicefarm:us-west-2:884244642857:project:64d2cc31-abd6-49f8-97da-162f82410bc0`
- Device pool ARN: `arn:aws:devicefarm:us-west-2:884244642857:devicepool:64d2cc31-abd6-49f8-97da-162f82410bc0/14d31c96-b8fc-4930-99c7-1a8948124213`
- Device: Samsung Galaxy S26 Ultra, model ID SM-S948U1, Android 16.
- Model: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, W4A16 groupwise quantization, group size 64.
- Full model zip SHA-256 verified on-device: `9de692be1c1ef1002fac25bd8f93c76e1d31975caa234fe1725f9eb294bfaa34`.
- Split package bytes: 4,500,000,000 + 4,500,000,000 + 2,266,405,198 = 11,266,405,198.
- Final stock benchmark run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/5165cf70-4ad2-4f21-bd40-89c6dd97c246`.
- Final custom benchmark run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/c444a573-d0e1-4924-abe2-5ca587478c2c`.
- Final stock quality run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/9908fb50-103c-4698-81c9-f2f12b98b335`.
- Final custom quality run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/4ac8ce83-abf1-42fe-8cd2-3ffde19821f2`.

| Run | Result | Evidence |
| --- | --- | --- |
| stock_mnn_v27_benchmark_cpu_qwen35_9b | PASSED | `results/reports/evidence/stock_mnn_cpu_benchmark_v27.json` |
| customlib_v27_benchmark_cpu_vulkan_hybrid_qwen35_9b | PASSED | `results/reports/evidence/customlib_cpu_vulkan_hybrid_benchmark_v27.json` |
| stock_mnn_v27_quality_english_qwen35_9b | PASSED | `results/reports/evidence/quality_validation_v27_stock_english.json` |
| customlib_v27_quality_english_qwen35_9b | PASSED | `results/reports/evidence/quality_validation_v27_custom_english.json` |
| stock/custom quality comparison | PASS | `results/reports/evidence/quality_validation_v27_english_comparison.json` |

## Same Settings

| Field | Value |
| --- | --- |
| Prompt tokens requested | 512 |
| max_new_tokens | 256 |
| Prompt token id for performance run | 16 |
| Temperature / top_k / top_p | 0 / 1 / 1 |
| Batch size | 1 |
| Warmup / measured iterations | 1 / 3 |
| Quality prompts | five deterministic English fixed token-ID prompts |
| Model package | same verified Qwen3.5-9B W4A16 package |
| Device pool | same Samsung Galaxy S26 Ultra pool |

## Overall Performance

| Variant | Engine | Requested backend | Actual backend | Status | Prefill TPS | Decode TPS | Decode TPOT | Device Farm run ARN | Evidence |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Fresh stock CPU v27 | stock_mnn | cpu | cpu | ok | 45.25510 | 2.26870 | 440.780 ms | `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/5165cf70-4ad2-4f21-bd40-89c6dd97c246` | `results/reports/evidence/stock_mnn_cpu_benchmark_v27.json` |
| Final custom v27 | customlib | cpu_vulkan_hybrid | cpu | ok, full custom decode | 2.47191 | 2.11989 | 471.723 ms | `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/c444a573-d0e1-4924-abe2-5ca587478c2c` | `results/reports/evidence/customlib_cpu_vulkan_hybrid_benchmark_v27.json` |
| Accepted v17 baseline | customlib | cpu_vulkan_hybrid | cpu | historical baseline | 2.21477 | 1.93417 | 517.018 ms | `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/1de54984-c9d9-41d1-81c1-7eed941585ed` | `results/reports/evidence/customlib_cpu_vulkan_hybrid_benchmark_v17.json` |
| v26 buffer-reuse candidate | customlib | cpu_vulkan_hybrid | cpu | superseded by v27 | 2.02717 | 1.75582 | 569.535 ms | `see v26 scheduled evidence` | `results/reports/evidence/customlib_cpu_vulkan_hybrid_benchmark_v26.json` |

## Comparisons

| Comparison | Decode TPS ratio | Decode TPOT delta | Interpretation |
| --- | ---: | ---: | --- |
| Final custom v27 / fresh stock MNN CPU v27 | 0.9344x | +30.943 ms | Custom is slower than stock; no speedup is claimed. |
| Final custom v27 / accepted v17 custom baseline | 1.0960x | -45.295 ms | v27 improves custom decode TPS by 9.60% and lowers TPOT. |
| Final custom v27 / v26 buffer-reuse candidate | 1.2074x | -97.812 ms | v27 recovers performance after quality fixes and buffer reuse. |

## Output Correctness / Quality Guard

TPS/TPOT measure speed, not semantic quality. The final custom result is accepted only because it also passes a deterministic output sanity guard against stock MNN on Device Farm. The validation uses fixed English token-ID prompts so both engines receive identical prompt IDs. It reports exact token match, prefix match length, token match rate, edit distance, invalid-token checks, empty-output checks, and repeated-token checks.

- Quality gate passed: YES.
- Stock native quality gate: YES.
- Custom native quality gate: YES.
- Exact full-output token match: `1 / 5` prompts.
- Comparison-gate pass: `5 / 5` prompts.
- Invalid token IDs: none in stock or custom outputs.
- Empty outputs: none.
- Degenerate repeated-token outputs: none.
- Repeated token 220 failure: none.
- Custom decoded text is not claimed because tokenizer decode is unavailable in the custom artifact; token IDs are dumped and compared directly.

| Prompt | Exact match | Prefix tokens | Token match rate | Edit distance | First mismatch | Custom sanity | Repeated token 220 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code_like | no | 14 | 0.438 | 18 | 14 | yes | no |
| english_explain | no | 10 | 0.312 | 2 | 10 | yes | no |
| english_factual | no | 16 | 0.500 | 16 | 16 | yes | no |
| long_512_token_style | yes | 32 | 1.000 | 0 | none | yes | no |
| math_reasoning | no | 4 | 0.125 | 27 | 4 | yes | no |

Token examples:

- `code_like`: stock first tokens `[271, 727, 884, 2784, 11, 292, 1590, 198, 262, 460, 264, 478, 292, 271, 727, 1228]`, custom first tokens `[271, 727, 884, 2784, 11, 292, 1590, 198, 262, 460, 264, 478, 292, 271, 7734, 264]`.
- `english_explain`: stock first tokens `[271, 248068, 271, 248069, 271, 760, 12515, 7701, 6105, 1521, 279, 8964, 579, 16078, 1100, 9872]`, custom first tokens `[271, 248068, 271, 248069, 271, 760, 12515, 7701, 6105, 1521, 8964, 579, 16078, 1100, 9872, 22602]`.
- `english_factual`: stock first tokens `[271, 248068, 198, 90700, 8340, 25, 198, 16, 13, 220, 2972, 2014, 53983, 279, 5952, 64700]`, custom first tokens `[271, 248068, 198, 90700, 8340, 25, 198, 16, 13, 220, 2972, 2014, 53983, 279, 5952, 64700]`.
- `long_512_token_style`: stock first tokens `[1330, 61446, 31969, 13, 7860, 5437, 537, 279, 2614, 27502, 8129, 303, 1330, 61446, 31969, 13]`, custom first tokens `[1330, 61446, 31969, 13, 7860, 5437, 537, 279, 2614, 27502, 8129, 303, 1330, 61446, 31969, 13]`.
- `math_reasoning`: stock first tokens `[271, 20, 248044, 248045, 8944, 248046]`, custom first tokens `[271, 20, 248044, 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045]`.

Limitations: The custom library is validated here for operator correctness, deterministic full custom decode-path execution, and basic generated-token sanity against stock MNN. It is not claimed as a production-quality model replacement; full semantic quality validation would require perplexity and downstream task benchmarks.

## Custom Path Coverage

Measured custom generation evidence from the v27 JSON:

```text
custom_decode_loop;hotpath_replaced=true;full_custom_decode=true;linear=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,linear_attn_qkv_a_b_z_out,lm_head;rmsnorm=custom;rope=custom;attention=custom_gqa_decode;linear_attention_state=custom_gated_delta;sampling=custom_greedy;prefill_kv_build=custom;fallback_ops=none
```

- `runtime.selected_kernels.hotpath_replaced = true`
- `runtime.selected_kernels.full_custom_decode = true`
- `runtime.selected_kernels.fallback_op_families = []`
- `custom_path.use_mnn_fallback = 0`
- `custom_path.calls_mnn_llm_response_for_measured_generation = false`
- `custom_backend_requested = cpu_vulkan_hybrid`
- `custom_backend_actual = cpu`

| Op family | Backend | Replaced? | Fallback? |
| --- | --- | ---: | ---: |
| q_proj | cpu | yes | no |
| k_proj | cpu | yes | no |
| v_proj | cpu | yes | no |
| o_proj | cpu | yes | no |
| gate_proj | cpu | yes | no |
| up_proj | cpu | yes | no |
| down_proj | cpu | yes | no |
| rmsnorm | cpu | yes | no |
| rope | cpu | yes | no |
| attention | cpu | yes | no |
| linear_attention_state | cpu | yes | no |
| lm_head | cpu | yes | no |
| sampling | cpu | yes | no |
| prefill_kv_build | cpu | yes | no |

## What Changed Since v17

The accepted v27 result starts from the v17 full-custom systems baseline, rejects the faster-but-broken repeated-token quality path, and adds correctness plus performance fixes before final acceptance:

- RMSNorm and Q/K norm loader now applies Qwen-style weight offset handling through `addOneToVector`.
- RoPE uses the Qwen3.5 active-slice layout and correct absolute position handling.
- Linear-attention recurrent update uses exported `A_log` and `dt_bias` gate semantics instead of the earlier approximation.
- XQ4 loading supports both symmetric v1 and affine-asymmetric v2 matrices without confusing dequant formulas.
- Prefill and decode reduce allocations by reusing buffers, reserving KV cache, keeping an embedding stream open, and reusing attention score/max buffers.
- The W4A16 generated GEMV branches once per matrix on affine-vs-symmetric dequant, so the final symmetric Qwen package avoids a per-row/per-group runtime branch.

## Custom Per-Kernel Wall Clock

This table comes from `results/reports/evidence/customlib_cpu_vulkan_hybrid_benchmark_v27.json`. The requested custom backend was `cpu_vulkan_hybrid`; the actual measured backend for all custom ops was CPU.

| Kernel / row | Op family | Backend | Calls | Total ms | Mean ms | Min ms | Max ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| prefill_token_custom | other | cpu | 512 | 208930.000 | 408.067000 | 372.307000 | 454.881000 |
| linear_down_proj_w4a16 | down_proj | cpu | 24576 | 62198.300 | 2.530850 | 2.247400 | 4.261930 |
| linear_gate_proj_w4a16 | gate_proj | cpu | 24576 | 61881.400 | 2.517960 | 2.271460 | 3.536090 |
| linear_up_proj_w4a16 | up_proj | cpu | 24576 | 61864.200 | 2.517260 | 2.253650 | 3.979010 |
| linear_attn_in_proj_qkv_w4a16 | linear_attention_projection | cpu | 18432 | 31118.100 | 1.688270 | 1.567190 | 3.366560 |
| linear_attention_state_update_custom | linear_attention_state | cpu | 18432 | 30394.700 | 1.649020 | 1.506720 | 2.999900 |
| linear_attn_out_proj_w4a16 | linear_attention_projection | cpu | 18432 | 15940.500 | 0.864829 | 0.794479 | 2.335420 |
| linear_attn_in_proj_z_w4a16 | linear_attention_projection | cpu | 18432 | 15837.200 | 0.859226 | 0.792604 | 1.517710 |
| sampling_greedy_custom | sampling | cpu | 256 | 12108.300 | 47.298200 | 43.414400 | 58.854500 |
| lm_head_custom | lm_head | cpu | 256 | 12108.000 | 47.297000 | 43.413200 | 58.852400 |
| attention_score_custom | attention_score | cpu | 6144 | 10389.600 | 1.691020 | 0.004323 | 4.418380 |
| linear_q_proj_w4a16 | q_proj | cpu | 6144 | 10374.300 | 1.688520 | 1.568440 | 2.452760 |
| linear_o_proj_w4a16 | o_proj | cpu | 6144 | 5296.200 | 0.862011 | 0.791823 | 1.584430 |
| attention_v_reduce_custom | attention_v_reduce | cpu | 6144 | 2421.630 | 0.394145 | 0.001094 | 1.097190 |
| linear_k_proj_w4a16 | k_proj | cpu | 6144 | 1847.090 | 0.300634 | 0.275781 | 0.544375 |
| linear_v_proj_w4a16 | v_proj | cpu | 6144 | 1843.890 | 0.300113 | 0.275156 | 0.723489 |
| linear_attention_conv1d_custom | linear_attention_state | cpu | 18432 | 1816.120 | 0.098531 | 0.090729 | 0.348906 |
| silu_gate_mul_custom | activation | cpu | 24576 | 1380.530 | 0.056174 | 0.050625 | 0.161667 |
| linear_attn_in_proj_a_w4a16 | linear_attention_projection | cpu | 18432 | 663.120 | 0.035977 | 0.033125 | 0.082552 |
| linear_attn_in_proj_b_w4a16 | linear_attention_projection | cpu | 18432 | 655.961 | 0.035588 | 0.032917 | 0.157240 |
| linear_attention_gated_rmsnorm_custom | linear_attention_state | cpu | 18432 | 434.937 | 0.023597 | 0.021354 | 0.205729 |
| attention_softmax_custom | attention_softmax | cpu | 6144 | 168.329 | 0.027397 | 0.000416 | 0.133750 |
| rmsnorm_input | rmsnorm | cpu | 24576 | 147.669 | 0.006009 | 0.005260 | 0.040364 |
| rmsnorm_post_attention | rmsnorm | cpu | 24576 | 146.737 | 0.005971 | 0.005260 | 0.083282 |
| attention_output_gate_custom | attention | cpu | 6144 | 139.876 | 0.022766 | 0.017968 | 0.131667 |
| rope_qk_custom | rope | cpu | 6144 | 134.287 | 0.021857 | 0.016510 | 0.055156 |
| linear_attention_qk_l2norm_custom | linear_attention_state | cpu | 18432 | 100.407 | 0.005447 | 0.004895 | 0.123854 |
| embedding_bf16_row_read | other | cpu | 768 | 20.528 | 0.026730 | 0.019167 | 0.061562 |
| prefill_kv_build_custom | other | cpu | 6144 | 12.782 | 0.002080 | 0.001250 | 0.037291 |
| kv_append_custom | other | cpu | 6144 | 10.262 | 0.001670 | 0.000886 | 0.036927 |
| rmsnorm_final | rmsnorm | cpu | 768 | 4.803 | 0.006253 | 0.005521 | 0.013594 |

Note: `sampling_greedy_custom` wraps the greedy section and overlaps with `lm_head_custom`; do not sum those two as exclusive time.

## Generated Kernel Microbench

The microbench ran on the same custom Device Farm APK after measured custom generation. It is not used for the overall TPS calculation.

| Kernel | Backend | Implementation | Qwen3.5 shape | Mean wall ms | Median wall ms |
| --- | --- | --- | --- | ---: | ---: |
| gemv_w4a16_attn_4096x4096 | cpu | generated_tiled_neon_fused_dequant_gemv_parallel_group_sum_reuse | attention_or_output_projection | 1.10762 | 1.07260 |
| gemv_w4a16_mlp_up_12288x4096 | cpu | generated_tiled_neon_fused_dequant_gemv_parallel_group_sum_reuse | mlp_gate_or_up_projection | 2.91837 | 3.15880 |
| gemv_w4a16_mlp_down_4096x12288 | cpu | generated_tiled_neon_fused_dequant_gemv_parallel_group_sum_reuse | mlp_down_projection | 3.07146 | 3.20208 |
| rmsnorm_hidden4096 | cpu | customlib_reference | hidden_size | 0.00550 | 0.00552 |
| rope_q_heads16_dim256_pos512 | cpu | customlib_reference | q_heads_x_head_dim | 0.03797 | 0.03771 |
| gqa_decode_ctx512_q16_kv4_dim256 | cpu | customlib_reference | full_attention_layer_decode | 2.43495 | 2.43479 |
| gated_delta_decode_k128_v128 | cpu | customlib_reference | linear_attention_layer_decode | 0.04771 | 0.04776 |

## MNN Hot-Path Trace: Top Op Types

The MNN hot-path trace is a separate debug-callback trace collected inside the custom APK after measured custom generation. It is evidence for MNN op wall-clock hotspots; it is not part of the measured custom generation path.

| Stage | Type | Calls | Total ms | Mean ms |
| --- | --- | ---: | ---: | ---: |
| prefill_or_first_token_inferred | Convolution | 249 | 10529.300 | 42.286400 |
| decode_inferred | Convolution | 3984 | 4440.390 | 1.114550 |
| decode_inferred | UnaryOp | 2464 | 1326.500 | 0.538351 |
| prefill_or_first_token_inferred | Raster | 521 | 490.795 | 0.942025 |
| decode_inferred | LinearAttention | 384 | 448.954 | 1.169150 |
| decode_inferred | LayerNorm | 1680 | 348.949 | 0.207708 |
| prefill_or_first_token_inferred | LinearAttention | 24 | 331.501 | 13.812500 |
| decode_inferred | While | 528 | 270.926 | 0.513118 |
| prefill_or_first_token_inferred | UnaryOp | 154 | 249.381 | 1.619350 |
| prefill_or_first_token_inferred | BinaryOp | 169 | 117.313 | 0.694161 |
| prefill_or_first_token_inferred | Attention | 8 | 117.168 | 14.646000 |
| decode_inferred | Attention | 128 | 97.709 | 0.763351 |

## MNN Hot-Path Trace: Top Op Names

| Stage | Type | Name | Calls | Total ms | Mean ms |
| --- | --- | --- | ---: | ---: | ---: |
| decode_inferred | Convolution | `/lm/lm_head/Linear` | 16 | 243.503 | 15.218900 |
| prefill_or_first_token_inferred | Convolution | `/layers.31/mlp/down_proj/Linear` | 1 | 94.766 | 94.765900 |
| prefill_or_first_token_inferred | Convolution | `/layers.22/mlp/down_proj/Linear` | 1 | 94.241 | 94.240900 |
| prefill_or_first_token_inferred | Convolution | `/layers.23/mlp/down_proj/Linear` | 1 | 92.362 | 92.362000 |
| prefill_or_first_token_inferred | Convolution | `/layers.18/mlp/down_proj/Linear` | 1 | 92.242 | 92.241500 |
| prefill_or_first_token_inferred | Convolution | `/layers.15/mlp/down_proj/Linear` | 1 | 91.799 | 91.798600 |
| prefill_or_first_token_inferred | Convolution | `/layers.6/mlp/up_proj/Linear` | 1 | 90.082 | 90.081600 |
| prefill_or_first_token_inferred | Convolution | `/layers.21/mlp/gate_proj/Linear` | 1 | 89.344 | 89.343600 |
| prefill_or_first_token_inferred | Convolution | `/layers.22/mlp/up_proj/Linear` | 1 | 88.979 | 88.979000 |
| prefill_or_first_token_inferred | Convolution | `/layers.19/mlp/up_proj/Linear` | 1 | 88.921 | 88.921200 |
| prefill_or_first_token_inferred | Convolution | `/layers.8/mlp/gate_proj/Linear` | 1 | 88.461 | 88.461300 |
| prefill_or_first_token_inferred | Convolution | `/layers.23/mlp/up_proj/Linear` | 1 | 88.426 | 88.425600 |

## Code Walkthrough Pointers

- Public ABI: `customlib/include/xqwen35.h`.
- Custom runtime selection: `customlib/runtime/session.cpp` creates `CustomModel` when `use_mnn_fallback = 0`; the Android custom benchmark sets that field to zero.
- Full custom prefill/decode: `customlib/runtime/custom_model.cpp` implements prompt-token prefill, KV append, grouped-query attention, gated-delta linear attention state, final `lm_head_custom`, and greedy sampling.
- Generated W4A16 GEMV: `customlib/kernels/generated/arm64_neon/xq_gemv_w4a16_neon.cpp` uses tiled fused dequant GEMV, persistent worker threads, activation group-sum reuse, branch-specialized symmetric/affine dequant, and streaming lm_head argmax. The W4 file contains no `gemvLowBitReference` call.
- Packer/exporter: `customlib/packer/pack_qwen35_xq4.py` and `customlib/packer/pack_qwen35_xq4_numpy.py` export `lm_head.weight`, full-attention projections, MLP projections, linear-attention `in_proj_a`, `in_proj_b`, `conv1d.weight`, `A_log`, and `dt_bias`.
- Stock MNN connector: `android/app/src/main/cpp/stock_benchmark_jni.cpp`.
- Custom JNI/benchmark/quality connector and Vulkan probe: `android/benchmark_app/src/main/cpp/benchmark_jni.cpp`.
- Device Farm bootstrap connector: `android/benchmark_app/src/androidTest/java/com/example/xqwen35bench/ModelBootstrap.java` and the matching stock app bootstrap.

## Test Evidence

- Host build: `cmake --build build-host-quality` passed; log: `results/reports/evidence/host_build_v27.txt`.
- Host tests: `ctest --test-dir build-host-quality --output-on-failure` -> 2 / 2 passed; log: `results/reports/evidence/host_ctest_v27.txt`.
- Android build: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_android.ps1` passed; log: `results/reports/evidence/android_build_v27.txt`.
- Device Farm stock quality v27: PASSED and emitted `BENCH_QUALITY_JSON`.
- Device Farm custom quality v27: PASSED and emitted `BENCH_QUALITY_JSON`.
- Device Farm stock benchmark v27: PASSED and emitted `BENCH_RESULT_JSON`.
- Device Farm custom benchmark v27: PASSED and emitted `BENCH_RESULT_JSON`.

## Final Artifacts

- Final report: `results/reports/final_devicefarm_report.md`
- Machine-readable summary: `results/reports/final_devicefarm_report.json`
- Code walkthrough: `docs/kernel_library_code_walkthrough_final.md`
- Quality validation report: `results/reports/quality_validation_report.md`
- Quality comparison: `results/reports/evidence/quality_validation_v27_english_comparison.json`
- Final custom benchmark JSON: `results/reports/evidence/customlib_cpu_vulkan_hybrid_benchmark_v27.json`
- Final stock benchmark JSON: `results/reports/evidence/stock_mnn_cpu_benchmark_v27.json`
- Presentation deck: `results/reports/qwen35_9b_v27_quality_performance_presentation.pptx`

## Cleanup Evidence

Generated build trees, raw Device Farm artifacts, APKs, ZIPs, model shards, virtualenvs, and private upload specs are excluded from the final committed evidence. Use `scripts/cleanup_local_artifacts.ps1` to remove local-only artifacts after preserving the small report and evidence files.
