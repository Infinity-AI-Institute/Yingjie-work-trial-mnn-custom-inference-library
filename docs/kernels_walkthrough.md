# Kernel Walkthrough

The custom library is organized around generated low-bit decode kernels plus reference implementations used only for tests and non-W4 experimental paths. The current v14 code walkthrough is in `docs/kernel_library_code_walkthrough_v14.md`.

## Current Kernel Set

- W4A16 GEMV: `customlib/kernels/generated/arm64_neon/xq_gemv_w4a16_neon.cpp`
- W3A16 GEMV: `customlib/kernels/generated/arm64_neon/xq_gemv_w3a16_neon.cpp`
- W2A16 GEMV: `customlib/kernels/generated/arm64_neon/xq_gemv_w2a16_neon.cpp`
- RMSNorm decode path: `customlib/runtime/custom_model.cpp`
- RoPE decode path: `customlib/runtime/custom_model.cpp`
- GQA attention decode fallback: `customlib/runtime/custom_model.cpp`
- Linear-attention state fallback: `customlib/runtime/custom_model.cpp`

The W4A16 generated kernel is a real fused dequant+GEMV implementation. It does not call `gemvLowBitReference` from `gemvW4A16Neon`.

## Packing Layout

Weights are row-major, bitpacked per output row:

- W4: two weights per byte.
- W3: eight weights per three bytes, with tail handling.
- W2: four weights per byte.

Each row is split into groupwise quantization groups. Each group stores `scale` and `zero`; dequantization is:

```text
weight = (code - zero) * scale
```

The headline comparison uses W4 group size 64. W2/W3 are experimental unless stock MNN can run the same bitwidth fairly.

## Correctness

`xq_kernel_correctness` verifies:

- Low-bit pack/dequant GEMV versus dense dequantized reference.
- RMSNorm numeric path.
- RoPE finite output.
- GQA attention decode softmax path.
- Gated Delta recurrent update.

The tests fail on mismatches and are intended to run on host and Android.

## Dispatcher

`customlib/kernels/generated/generated_dispatch.cpp` records selected kernels and CPU features. The custom runtime also reports decode coverage from `CustomModel::selectedKernelSummary`, including replaced and fallback op families.
