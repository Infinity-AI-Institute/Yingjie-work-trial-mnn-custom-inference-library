# Qwen3.5-9B Custom Mobile Inference Library

This repository builds and benchmarks a custom Android inference library for `Qwen/Qwen3.5-9B` against stock MNN on AWS Device Farm.

Final pinned inputs:

- MNN: `0bff03cbef43c783f44e41484b9f8a0b28bd758d`
- Hugging Face model: `Qwen/Qwen3.5-9B`
- HF revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Custom quantization: W4A16 groupwise, group size 64
- Final model package SHA-256: `9de692be1c1ef1002fac25bd8f93c76e1d31975caa234fe1725f9eb294bfaa34`

## Final Status

The accepted v27 delivery is complete. Stock MNN CPU and customlib were scheduled on the same AWS Device Farm Samsung Galaxy S26 Ultra pool with the same full Qwen3.5-9B package, same 512-token prompt, same 256-token decode length, greedy settings, and 1 warmup / 3 measured iterations. Separate Device Farm quality-validation runs also compared stock and custom generated token IDs on five deterministic English prompts.

The measured custom generation path is full custom CPU generation:

- `use_mnn_fallback = 0`
- `custom_path.calls_mnn_llm_response_for_measured_generation = false`
- `runtime.selected_kernels.full_custom_decode = true`
- `runtime.selected_kernels.hotpath_replaced = true`
- `runtime.selected_kernels.fallback_op_families = []`

Final performance result:

| Variant | Requested backend | Actual backend | Status | Prefill TPS | Decode TPS | Decode TPOT |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Stock MNN v27 | cpu | cpu | ok | 45.25510 | 2.26870 | 440.780 ms |
| Customlib v27 | cpu_vulkan_hybrid | cpu | ok, full custom decode | 2.47191 | 2.11989 | 471.723 ms |
| Accepted v17 custom baseline | cpu_vulkan_hybrid | cpu | historical baseline | 2.21477 | 1.93417 | 517.018 ms |

Speedup verdict: no stock-MNN speedup is claimed. Final custom is `0.9344x` fresh stock MNN CPU decode TPS. It is, however, `1.0960x` the accepted v17 custom baseline and passes the new output-quality guard.

## Output Quality Guard

TPS/TPOT measure speed, not semantic quality. The final v27 custom result passed a deterministic Device Farm quality guard:

- Stock quality run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/9908fb50-103c-4698-81c9-f2f12b98b335`
- Custom quality run ARN: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/4ac8ce83-abf1-42fe-8cd2-3ffde19821f2`
- Quality gate: PASS
- Exact full-output token match: `1 / 5` prompts
- Comparison-gate prompts: `5 / 5`
- Invalid tokens / empty outputs / repeated-token-220 degeneration: none

The custom library is validated for operator correctness, full custom decode-path execution, and basic generated-token sanity against stock MNN. It is not claimed as a production-quality model replacement; full semantic quality validation with perplexity and downstream task benchmarks remains future work.

## Final Evidence

- Final report: `results/reports/final_devicefarm_report.md`
- Machine-readable summary: `results/reports/final_devicefarm_report.json`
- Code walkthrough: `docs/kernel_library_code_walkthrough_final.md`
- Quality validation report: `results/reports/quality_validation_report.md`
- Stock benchmark evidence: `results/reports/evidence/stock_mnn_cpu_benchmark_v27.json`
- Custom benchmark evidence: `results/reports/evidence/customlib_cpu_vulkan_hybrid_benchmark_v27.json`
- Quality comparison evidence: `results/reports/evidence/quality_validation_v27_english_comparison.json`
- Presentation deck: `results/reports/qwen35_9b_v27_quality_performance_presentation.pptx`

Final AWS Device Farm runs:

- Stock MNN CPU benchmark: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/5165cf70-4ad2-4f21-bd40-89c6dd97c246`
- Customlib benchmark: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/c444a573-d0e1-4924-abe2-5ca587478c2c`
- Stock quality validation: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/9908fb50-103c-4698-81c9-f2f12b98b335`
- Custom quality validation: `arn:aws:devicefarm:us-west-2:884244642857:run:64d2cc31-abd6-49f8-97da-162f82410bc0/4ac8ce83-abf1-42fe-8cd2-3ffde19821f2`
- Device pool: `arn:aws:devicefarm:us-west-2:884244642857:devicepool:64d2cc31-abd6-49f8-97da-162f82410bc0/14d31c96-b8fc-4930-99c7-1a8948124213`

## What The Custom Library Replaces

The final measured custom path replaces the requested decode and prefill families:

- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `gate_proj`, `up_proj`, `down_proj`
- `rmsnorm`
- `rope`
- `attention`
- `linear_attention_state`
- `lm_head`
- `sampling`
- `prefill_kv_build`

The public ABI is in `customlib/include/xqwen35.h`. The measured Android custom benchmark enters the library through `xq_generate`, then runs `Session::prefill`, `Session::decodeOne`, and `CustomModel` without calling `MNN::Transformer::Llm::response`.

## Build Host Correctness Tests

```powershell
$env:PATH = "C:\Users\Yingjie Huang\qwen-phone-npu-trial\third_party\toolchains\w64devkit\bin;$env:PATH"
cmake --build build-host-quality
ctest --test-dir build-host-quality --output-on-failure
```

Final host result: 2 / 2 tests passed. The tests cover W4A16 GEMV/reference behavior, lm_head argmax, greedy sampling, stable softmax, grouped-query attention, KV cache, RoPE, RMSNorm, linear-attention state, prefill KV length, and tiny end-to-end custom decode.

## Build Android APKs

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_android.ps1
```

The v27 Device Farm runs used rebuilt stock/custom release APKs plus androidTest APKs from this flow.

## Model Packing

The packers are `customlib/packer/pack_qwen35_xq4.py` and `customlib/packer/pack_qwen35_xq4_numpy.py`. They export W4A16 matrices and runtime metadata for full-attention projections, MLP projections, Qwen3.5 linear-attention tensors, `lm_head.weight`, embeddings, norm vectors, RoPE metadata, and quantization metadata. Device Farm reassembled and verified the three-part model package on-device before every final run.

## Repository Cleanup

Generated build trees, raw Device Farm artifacts, APKs, ZIPs, model shards, virtualenvs, and local upload specs are ignored. Use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/cleanup_local_artifacts.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/cleanup_local_artifacts.ps1 -Execute
```

The cleanup keeps source, documentation, final reports, and small evidence JSON/TXT/YML files. It keeps the final local model zip unless `-RemoveFinalModelZip` is explicitly passed.

## Honesty Rules

This repository only claims results backed by artifacts. The final v27 result is a faithful same-device full-model stock-vs-custom benchmark with a separate output-quality guard. It is not a stock-MNN speedup result and does not claim custom Vulkan kernel execution.

## Post-v27 Vulkan Attempt

A later Vulkan implementation attempt is documented separately in `results/reports/final_vulkan_blocker_report.md` and `results/reports/vulkan_iteration_log.md`. That pass added a real custom Vulkan runtime plus a W4A16 GEMV compute shader, validated the shader on AWS Device Farm Samsung Galaxy S26 Ultra, and ran a short full-model custom generation integration where projection-family W4A16 rows reported `backend = vulkan`.

That attempt is not the official final result:

- accepted official result remains v27 CPU customlib;
- `custom_backend_actual = cpu_vulkan_hybrid`, not full `vulkan`;
- RMSNorm, RoPE, attention, linear-attention state, lm_head, sampling, and prefill KV still ran on CPU;
- no Vulkan quality-validation `BENCH_QUALITY_JSON` was produced;
- no 10 TPS or Vulkan speedup is claimed.
