# Kernel Library Code Walkthrough v14

This document describes the current custom decode path used for the Qwen3.5-9B Device Farm comparison.

## Architecture

The benchmark has two engines:

- Stock MNN: `android/app/src/main/cpp/stock_benchmark_jni.cpp` creates an MNN LLM session and calls MNN generation for the baseline.
- Customlib: `android/benchmark_app/src/main/cpp/benchmark_jni.cpp` calls the public C ABI in `customlib/include/xqwen35.h` with `use_mnn_fallback = 0`.

The custom path enters:

```text
xq_create
  -> xq::runtime::Session::load
  -> xq::runtime::CustomModel::load
xq_generate
  -> xq::runtime::Session::prefill
  -> xq::runtime::Session::decodeOne
  -> xq::runtime::CustomModel::decodeOne
```

For measured custom generation, the custom benchmark does not call `MNN::Llm::response`. MNN is still linked into the custom APK for the separate MNN hot-path trace evidence path, but that trace is emitted outside the measured custom generation loop.

## Weight Format

`customlib/packer/pack_qwen35_xq4.py` converts Qwen3.5-9B safetensors into the custom `.xq4` format. Each file stores:

- magic: `XQW4A16`
- output rows and input columns
- group size, currently 64
- row-major int4 payload, two codes per byte
- fp32 scale and zero arrays per row/group

The runtime loader in `customlib/runtime/custom_model.cpp` maps every packed projection needed by decode:

- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `gate_proj`, `up_proj`, `down_proj`
- Qwen3.5 linear-attention projections: `in_proj_qkv`, `in_proj_z`, `out_proj`

It also loads `.f32` RMSNorm vectors and reads token embeddings from `embeddings_bf16.bin`.

## Generated W4A16 GEMV

The primary generated kernel is `customlib/kernels/generated/arm64_neon/xq_gemv_w4a16_neon.cpp`.

The implementation is no longer a wrapper around `gemvLowBitReference`. The W4 path performs fused dequantization and GEMV directly:

```text
for row tile:
  initialize row accumulators
  for quant group:
    unpack int4 nibbles
    compute sum(code * activation)
    compute sum(activation)
    apply scale * (code_dot - zero * x_sum)
  write output row
```

On arm64 NEON, groups are processed in 16-code chunks using `vld1_u8`, nibble unpacking, `vzip_u8`, widening, float conversion, and vector FMA accumulation. The scalar helper is only the non-NEON body for the same generated W4 algorithm, not the old reference GEMV dispatcher. Invalid/non-W4 inputs return zero output rather than calling the reference kernel.

The Qwen3.5 shapes covered by the generated W4 kernel are:

- `4096 x 4096`: attention `q/k/v/o` and linear-attention z/out style matrices
- `12288 x 4096`: MLP `gate/up`
- `4096 x 12288`: MLP `down`
- Qwen3.5 grouped key/value projection files used by the full-attention layers

## Decode Operators

`customlib/runtime/custom_model.cpp` drives the decode graph one token at a time.

Custom operators:

- RMSNorm: `rmsNorm` in `CustomModel`, traced as `rmsnorm_input`, `rmsnorm_post_attention`, and `rmsnorm_final`.
- RoPE: `applyRoPE`, traced as `rope_qk_custom`.
- Linear hotpath: `runLinear`, traced by projection family, calls `xq::kernels::gemvW4A16Neon`.
- FFN gate: `siluGateMul`, traced as `silu_gate_mul_custom`.

Replaced op families reported in `selected_kernels`:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj,
rmsnorm, rope, linear_attn_qkv_z_out
```

Current fallbacks are explicit and measured as fallback trace entries:

```text
attention, linear_attention_state, lm_head, sampling, prefill_kv_build
```

These fallbacks mean the custom path is a true custom decode-hotpath benchmark, but it is not allowed to claim an end-to-end model-quality or full-kernel speedup over MNN for attention/lm_head until those families are also replaced.

## Connectors

### MNN Connector

The stock baseline uses unmodified MNN through `MNN::Transformer::Llm` in `android/app/src/main/cpp/stock_benchmark_jni.cpp`.

The custom app also contains `runMnnHotpathTraceJson` in `android/benchmark_app/src/main/cpp/benchmark_jni.cpp`. That code creates an MNN session and records MNN hot-path op type/name wall clock for evidence only. It is not on the measured custom generation path.

### Fallback Connector

`customlib/runtime/session.cpp` still supports an MNN fallback mode for API compatibility when `xq_options.use_mnn_fallback != 0`. The v14 custom benchmark sets `use_mnn_fallback = 0`, so `Session::load` creates `CustomModel` instead of MNN LLM for measured generation.

### JNI Connector

The Android test apps call:

- `NativeStockBenchmark.runBenchmark(...)` for stock MNN.
- `NativeBenchmark.runBenchmark(...)` for customlib.

Both return `BENCH_RESULT_JSON` and write the same JSON into app external files for Device Farm artifact collection.

### Benchmark Connector

The custom benchmark JSON includes:

- `runtime.selected_kernels.hotpath_replaced = true`
- `replaced_op_families`
- `fallback_op_families`
- `custom_path.calls_mnn_llm_response_for_measured_generation = false`
- `per_kernel_wall_clock`
- prefill TPS, decode TPS, and decode TPOT
- MNN hot-path trace evidence

Stock MNN JSON reports the same prompt/generation settings and stock prefill/decode TPS/TPOT fields.

### Device Farm Connector

The instrumentation bootstrap supports large model artifacts through split Device Farm uploads:

- `model_zip_part_urls_file` contains one URL per model zip part.
- `ModelBootstrap.downloadParts` streams each part into one final zip file in order.
- The full zip sha256 is verified before unzip.
- The zip is deleted after unzip to reduce device storage pressure.

This is required because the full hotpath package is 10.6GB and Device Farm single upload objects are limited to less than the observed 5GB S3 upload cap.

## Evidence Fields To Check

For a valid v14 custom run, the benchmark JSON must show:

```json
{
  "runtime": {
    "selected_kernels": {
      "hotpath_replaced": true
    }
  },
  "custom_path": {
    "calls_mnn_llm_response_for_measured_generation": false
  }
}
```

The trace must include nonzero wall time for W4 linear families, RMSNorm, and RoPE, and it must also list the fallback families honestly.
