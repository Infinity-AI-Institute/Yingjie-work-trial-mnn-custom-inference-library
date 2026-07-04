# Connector Walkthrough

The custom library keeps stock MNN unmodified for the baseline. The measured v14 custom path uses `customlib/runtime/CustomModel` for decode hotpath execution with `use_mnn_fallback = 0`; MNN is not the main custom generation path.

## MNN Version

Pinned MNN commit:

```text
0bff03cbef43c783f44e41484b9f8a0b28bd758d
```

The pinned tree includes Qwen3.5 export support:

- `transformers/llm/export/utils/model_mapper.py`
- `transformers/llm/export/utils/custom_op.py`
- `transformers/llm/export/utils/transformers.py`

MNN's exporter emits LLM custom op semantics such as:

- `LlmExporter::FakeLinear`
- `LlmExporter::FusedAttention`
- `LlmExporter::FusedLinearAttention`

## Custom Runtime Connector

The public ABI in `customlib/include/xqwen35.h` is implemented by `customlib/runtime/session.cpp`.

- `xq_create` loads either MNN fallback mode or `CustomModel`.
- The v14 Android custom benchmark sets `xq_options.use_mnn_fallback = 0`.
- `xq_generate` dispatches to the custom prefill/decode loop when `CustomModel` is loaded.
- `xq_get_kernel_trace_json` returns per-kernel wall-clock trace entries from the measured custom path.

The measured v14 custom decode loop replaces linear decode projections, RMSNorm, RoPE, and FFN gate math. Attention, linear-attention recurrent state, lm_head, sampling, and prefill KV build remain explicit fallbacks.

## Fallback Rules

- The stock baseline may use MNN end to end.
- The custom v14 measured path must not call `MNN::Llm::response`.
- Any fallback must be logged in selected kernel metadata and final reports.
- The public ABI never exposes raw MNN objects.
