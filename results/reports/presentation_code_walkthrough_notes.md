# Word-For-Word Speaker Notes

Use this as the exact English script for the final v27 presentation deck.
The PowerPoint speaker notes contain the same spoken text. The cue after each slide is an instruction, not spoken text.

## Evidence framing

- Main final benchmark: v27 Device Farm benchmark.
- Fresh stock MNN CPU decode TPS: 2.26870.
- Final custom v27 decode TPS: 2.11989.
- Custom / stock decode ratio: 0.9344x.
- Historical v17 custom decode TPS: 1.93417, or 0.8493x of stock MNN CPU.
- Do not claim speedup, 20 TPS, production-quality output, token-level equivalence, or custom Vulkan kernel execution.

## Slide 1: Title

Say exactly:

Good morning. Today I am presenting my Qwen3.5-9B custom mobile inference library. The point of this presentation is not to claim that I beat MNN. The point is to show exactly what I built, how the custom runtime differs from stock MNN, how the kernels and connectors work, what was measured on AWS Device Farm, and where the remaining quality limits are. The short version is this: the custom path is real and benchmarked, v27 is the best final result, it is still slower than MNN, and I do not claim production-quality model output.

Advance after finishing this paragraph.

## Slide 2: The work trial had two concrete requirements

Say exactly:

I want to start with the actual requirements, because this keeps the presentation honest. I was asked to create a custom kernel library and provide a code walkthrough of the kernels and connectors. I was also asked to compare those generated kernels to MNN using wall-clock per kernel, overall time per output token, and tokens per second. The comparison had to use the same full Qwen3.5-9B package, the same Device Farm phone, and the same generation settings. So the rest of the deck is organized as evidence against those two requirements.

Advance after finishing this paragraph.

## Slide 3: The honest result is useful, but not flattering

Say exactly:

Here is the honest verdict. Requirement 1 is met for a real systems and kernel library with a code walkthrough. Requirement 2 is met for a same-device Device Farm benchmark comparison. The custom path does not beat stock MNN. In the final v27 run, custom decode is 2.11989 tokens per second, and fresh stock MNN CPU decode is 2.26870 tokens per second, so custom is 0.9344x stock. I also do not claim production-quality output. Earlier quality runs exposed repeated-token failures, and v27 passes a deterministic sanity guard, but only one out of five validation prompts exactly matches stock MNN.

Advance after finishing this paragraph.

## Slide 4: Customlib is not the same execution path as stock MNN

Say exactly:

This slide is the cleanest way to explain what is different from MNN. The stock path sends the prompt and model package into MNN::Transformer::Llm, then MNN owns the internal graph, runtime, operators, output, and timing. The custom path enters my C ABI, creates a Session, loads CustomModel, runs a custom prefill and decode loop, calls custom W4A16 kernels and custom attention, lm_head, and sampling, and then emits custom timing JSON. The key evidence fields are use_mnn_fallback equals zero, calls_mnn_llm_response_for_measured_generation equals false, full_custom_decode equals true, and fallback_op_families is empty.

Advance after finishing this paragraph.

## Slide 5: The repository has a distinct custom inference stack

Say exactly:

This is the repository map I use to defend that the work is not just a wrapper. The public C ABI is in customlib/include/xqwen35.h. The session connector is in customlib/runtime/session.cpp. The custom runtime is in customlib/runtime/custom_model.cpp. The generated W4A16 kernel is in customlib/kernels/generated/arm64_neon/xq_gemv_w4a16_neon.cpp. The packer is in customlib/packer/pack_qwen35_xq4.py. The custom Android benchmark connector is benchmark_jni.cpp, and the stock MNN connector is stock_benchmark_jni.cpp. So the custom path has its own package format, runtime, kernels, connectors, and evidence outputs.

Advance after finishing this paragraph.

## Slide 6: The public C ABI is the external boundary

Say exactly:

The C ABI is the boundary between Android or host code and the custom inference library. The important entry points are xq_create, xq_generate, xq_prefill, xq_decode_one, xq_get_last_metrics, and xq_get_kernel_trace_json. This matters because the Android benchmark is not directly calling MNN APIs for measured custom generation. It is calling this custom ABI. The fallback option is explicit in xq_options, and the benchmark sets that option to zero for the custom run.

Advance after finishing this paragraph.

## Slide 7: Session chooses CustomModel when fallback is disabled

Say exactly:

This is the runtime selection logic. Session supports an MNN fallback mode, and that is useful for debugging. If use_mnn_fallback is nonzero, Session can create an MNN LLM. But in the measured custom benchmark, use_mnn_fallback is zero. In that branch, Session creates CustomModel and loads the custom model package. Then xq_generate performs prefill once and calls decodeOne repeatedly. So the repo contains MNN response calls, but the measured custom generation path does not use them.

Advance after finishing this paragraph.

## Slide 8: Measured custom generation follows the custom call chain

Say exactly:

This is the call chain I want you to keep in mind for the rest of the walkthrough. The Android benchmark calls NativeBenchmark.runBenchmark. That calls xq_create, then xq_generate. Inside xq_generate, Session calls prefill, CustomModel runs prefill across the prompt, then Session calls decodeOne for each generated token, and CustomModel runs decodeOne and sampleGreedy. The benchmark records BENCH_RESULT_JSON after this path, including TPOT, TPS, selected kernels, and per-kernel wall-clock rows.

Advance after finishing this paragraph.

## Slide 9: The custom package carries weights for the custom runtime

Say exactly:

The custom runtime needs a custom package, not just the MNN graph. The packer exports W4A16 matrices for q, k, v, o, gate, up, and down projections. It also exports linear-attention tensors, including in_proj_qkv, in_proj_a, in_proj_b, in_proj_z, out_proj, conv1d.weight, A_log, and dt_bias. It exports lm_head.weight for real logits, BF16 embeddings, norm vectors, RoPE metadata, and xqwen35_manifest.json. The MNN package remains useful for the stock baseline and optional fallback, but the measured custom linears use the custom W4A16 package.

Advance after finishing this paragraph.

## Slide 10: The W4A16 kernel fuses unpack, dequant, and GEMV

Say exactly:

This is the main generated kernel story. W4A16 means the weights are stored as four-bit values, the activation vector is treated as the runtime activation input, and accumulation is performed in floating point. The kernel avoids materializing dequantized weights. For each row and quantization group, it computes a code dot product, computes the activation group sum, and combines them as scale times code_dot minus zero times x_sum. That fused path targets batch-one decode GEMV, which is the mobile LLM hot path. The final generated file does not call gemvLowBitReference.

Advance after finishing this paragraph.

## Slide 11: Custom prefill walks the whole prompt, not only the last token

Say exactly:

A common shortcut would be to only load the last prompt token, but that would not be faithful. The custom prefill path walks every prompt token. For each token, it reads the BF16 embedding row, runs the full layer stack, appends key and value cache entries for full-attention layers, and updates recurrent state for linear-attention layers. The trace includes prefill_token_custom and prefill_kv_build_custom. This proves the systems path exists, but it does not by itself prove production-quality language output.

Advance after finishing this paragraph.

## Slide 12: The measured custom path replaces the requested decode families

Say exactly:

This table summarizes operator coverage. The q, k, v, and o projections use custom W4A16 GEMV. The gate, up, and down MLP projections also use custom W4A16 GEMV. RMSNorm and RoPE are implemented in the custom runtime. Attention is custom grouped-query decode. The linear-attention recurrent state is custom. lm_head uses packed W4A16 logits, and sampling uses greedy argmax. For the measured custom path, fallback_op_families is empty, which is the key coverage field.

Advance after finishing this paragraph.

## Slide 13: Full-attention decode is implemented explicitly

Say exactly:

The full-attention decode order is explicit in CustomModel. First the runtime applies input RMSNorm. Then it runs q, k, and v W4A16 projections. It applies q_norm and k_norm where required, applies active-slice RoPE at the absolute position, appends K and V to the cache, maps query heads to grouped key-value heads, computes dot product scores divided by square root of head_dim, performs stable softmax, reduces over V, applies the output gate, and finally runs o_proj as W4A16 GEMV. The trace rows show attention_score_custom, attention_softmax_custom, attention_v_reduce_custom, kv_append_custom, and linear_o_proj_w4a16.

Advance after finishing this paragraph.

## Slide 14: Linear-attention state is custom and accuracy-sensitive

Say exactly:

The linear-attention layers maintain custom recurrent state across prefill and decode. The runtime uses exported tensors such as in_proj_qkv, in_proj_a, in_proj_b, in_proj_z, out_proj, conv1d.weight, A_log, and dt_bias. The trace rows include the convolution update, qk L2 normalization, state update, gated RMSNorm, and output projection. This area is also accuracy-sensitive. Later debugging found graph-level mismatches around gate and norm semantics, so this is real custom work, but it is also a place I would harden further before making production-quality claims.

Advance after finishing this paragraph.

## Slide 15: lm_head and sampling use real logits, not hash tokens

Say exactly:

This slide addresses a very important shortcut that is not allowed. The measured path does not use hash-based token generation. The runtime loads packed lm_head.weight from the custom package. During greedy generation, it computes real logits from the final hidden state and chooses the argmax token. The optimized path can stream the argmax to avoid materializing the full logits vector. The trace rows are lm_head_custom and sampling_greedy_custom. But real logits are necessary, not sufficient, for model-quality output, which is why quality validation is separate.

Advance after finishing this paragraph.

## Slide 16: Connectors are the glue around the custom kernels

Say exactly:

When I say connector, I mean the glue that makes kernels usable from the outside world. The C ABI connector exposes xq_create, xq_generate, and metric APIs. The JNI connector lets the Android benchmark call the native library. The MNN connector is used for the stock baseline and optional fallback. The benchmark connector emits BENCH_RESULT_JSON, TPOT, TPS, and wall-clock rows. The Device Farm connector handles model bootstrap, SHA verification, and artifacts. The Vulkan connector probes capability, but no custom Vulkan kernels are claimed. The quality connector emits BENCH_QUALITY_JSON and compares token dumps.

Advance after finishing this paragraph.

## Slide 17: The benchmark holds model, device, and settings constant

Say exactly:

The benchmark design is intentionally controlled. Both stock and custom runs use the AWS Device Farm Samsung Galaxy S26 Ultra pool. They use the same full Qwen3.5-9B W4A16 package with on-device SHA verification. The performance prompt requests 512 prompt tokens using fixed token id 16. Generation uses max_new_tokens 256, temperature zero, top_k one, top_p one, and batch size one. There is one warmup and three measured iterations. Decode TPS is generated tokens divided by decode time, and TPOT is decode time divided by generated tokens.

Advance after finishing this paragraph.

## Slide 18: v27 benchmark result: custom reached 93.4% of stock CPU

Say exactly:

This is the final performance result I would present first. Fresh stock MNN CPU reaches 2.26870 decode tokens per second and 440.780 milliseconds TPOT. The accepted v17 custom baseline was 1.93417 decode tokens per second and 517.018 milliseconds TPOT. The final v27 custom result is 2.11989 decode tokens per second and 471.723 milliseconds TPOT. So v27 improves over v17, but it is still slower than stock MNN. The final custom-to-stock ratio is 0.9344x, and no speedup is claimed.

Advance after finishing this paragraph.

## Slide 19: v27 per-kernel wall clock shows where custom time goes

Say exactly:

This table explains where the custom runtime spends time. Prefill_token_custom is the largest row, because the custom prefill path walks the full prompt through the layer stack. The W4A16 MLP projections are also large: down_proj, gate_proj, and up_proj each consume significant time. Linear-attention projection and recurrent state update are also major rows. lm_head_custom and sampling_greedy_custom matter because the vocabulary is large. This table is useful because it tells me where future optimization should focus, rather than only showing overall TPS.

Advance after finishing this paragraph.

## Slide 20: Microbench isolates kernel cost, but it is not overall TPS

Say exactly:

The microbench rows are useful but limited. They isolate individual kernels such as 4096 by 4096 W4A16 GEMV, 12288 by 4096 MLP projection, 4096 by 12288 down projection, RMSNorm, RoPE, grouped-query attention decode, and gated-delta linear attention. These numbers help explain kernel-level behavior, but they are not the overall model speed. The full Device Farm run is the source for TPOT and TPS. I use microbench results for kernel development, not as a substitute for the full-model benchmark.

Advance after finishing this paragraph.

## Slide 21: MNN hot-path trace gives a separate baseline hotspot view

Say exactly:

The MNN hot-path trace is collected separately using a debug callback. It is not part of measured custom generation. It is included because it helps compare hotspot structure. In the trace, MNN spends a lot of time in Convolution-like linear operators, UnaryOp, LinearAttention, LayerNorm, Attention, and the lm_head Linear. This is directionally consistent with the custom bottlenecks: linears, attention or linear-attention state, and lm_head dominate. The important caveat is that this trace is evidence about MNN hotspots, not custom timing.

Advance after finishing this paragraph.

## Slide 22: Vulkan was attempted, but the final valid comparison is CPU

Say exactly:

I also want to be precise about Vulkan. Vulkan was attempted and probed. Stock MNN Vulkan initialized Adreno Vulkan but crashed before producing benchmark JSON in the backend sweep evidence. The custom cpu_vulkan_hybrid path successfully probed Vulkan symbols, but the final custom_backend_actual is CPU, and vulkan_generation_kernels_used is false. Therefore I do not claim custom Vulkan kernel execution, and I do not claim a Vulkan speedup. The defensible final comparison is CPU versus CPU.

Advance after finishing this paragraph.

## Slide 23: Quality validation changed what I can honestly claim

Say exactly:

Speed is not correctness. After performance benchmarking, I added deterministic quality validation. The validation uses fixed prompts, generated token dumps, exact match, prefix match length, token match rate, edit distance, invalid-token checks, empty-output checks, and repeated-token checks. Rejected quality runs found serious repeated-token behavior, especially token 220, with zero out of five exact matches. The latest v27 English sanity guard passes five out of five comparison-gate prompts and removes the repeated-token-220 failure, but only one out of five exactly matches stock MNN. So I claim basic token sanity, not production-quality output.

Advance after finishing this paragraph.

## Slide 24: Accuracy debugging found real graph-level issues

Say exactly:

The quality failures were not just formatting or reporting issues. Debugging found real graph-level correctness issues: linear-attention gate semantics mismatch, L2 norm epsilon mismatch, Qwen3.5 partial RoPE active-slice layout issues, and a full-attention q and gate split issue. Those are the kinds of details that can make generated tokens diverge even when every operator has an implementation. This is why the final position is careful: systems path and benchmark requirements are met, quality sanity is improved, but full production-quality validation remains future work.

Advance after finishing this paragraph.

## Slide 25: The claims are intentionally narrow

Say exactly:

This is the slide I use to keep the defense precise. I can claim that the custom kernel library exists. I can claim the measured custom path is separate from stock MNN generation. I can claim there is a full-model Device Farm benchmark with per-kernel wall clock, TPOT, and TPS. I can claim v27 reaches 93.4 percent of fresh stock MNN CPU decode throughput. I can also claim the quality blocker and later sanity guard are documented. I should not claim speedup over MNN, 20 tokens per second, production-quality output, token-level equivalence, or custom Vulkan kernel execution.

Advance after finishing this paragraph.

## Slide 26: Final takeaway: the system exists, but quality is the next frontier

Say exactly:

Here is my final takeaway. I built a custom systems and kernel inference path for full Qwen3.5-9B mobile inference. I connected it to Android, Device Farm, per-kernel tracing, and a stock MNN comparison. The final v27 result is slower than MNN but much closer than the v17 baseline, reaching 93.4 percent of stock CPU decode TPS. Quality validation prevented me from overclaiming, and it exposed the remaining correctness work. So the project satisfies the systems and benchmark requirements, while production-level model quality remains the next frontier.

Advance after finishing this paragraph.

## Slide 27: Appendix: code-level Q&A

Say exactly:

The main presentation is now complete. If there are implementation questions, I will use the appendix. The appendix maps each claim back to a file path and a small code snippet. The purpose is not to show every line of code. The purpose is to show that the custom ABI, runtime selection, prefill and decode loop, W4A16 kernel, benchmark connector, stock connector, packer, and quality connector all exist as concrete code, and that each one supports a specific claim I made in the main presentation.

Advance after finishing this paragraph.

## Slide 28: Appendix A: ABI entry points

Say exactly:

This appendix slide shows where the custom library starts. The file is customlib/include/xqwen35.h. The important entry points are xq_create, xq_generate, xq_prefill, xq_decode_one, xq_get_last_metrics, and xq_get_kernel_trace_json. This proves there is a stable native API boundary. It matters because Android and host tests can drive the same boundary. It does not by itself prove model correctness, but it proves the custom runtime is not just an informal script or report artifact.

Advance after finishing this paragraph.

## Slide 29: Appendix B: custom runtime selection

Say exactly:

This slide is the nuance around MNN fallback. The code reads use_mnn_fallback from the options. If fallback is enabled, Session can create an MNN LLM. If fallback is disabled, Session creates CustomModel and loads the custom package. The measured custom benchmark uses fallback disabled. So the correct statement is not that the repo has no MNN code. The correct statement is that the measured custom generation path does not call MNN LLM response.

Advance after finishing this paragraph.

## Slide 30: Appendix C: custom prefill and decode loop

Say exactly:

This slide shows the core custom generation loop. In prefill, CustomModel iterates over prompt tokens, loads each embedding row, and runs the layer stack at the correct position. In decodeOne, it runs the layer stack for the current position, samples a greedy token, and loads that token's embedding for the next step. This proves that xq_generate ultimately drives CustomModel and the custom kernels, rather than delegating the main generation path to MNN.

Advance after finishing this paragraph.

## Slide 31: Appendix D: W4A16 GEMV generated path

Say exactly:

This slide is the kernel-level evidence. gemvW4A16Neon receives a QuantizedMatrix and an activation vector. It precomputes activation group sums, then computes rows using the generated row implementation. gemvW4A16ArgmaxNeon is used for the streaming lm_head argmax path. The important point is that the W4 path computes directly from packed int4 data and scales, and the generated file does not call gemvLowBitReference. This is the main generated-kernel distinction from a reference fallback.

Advance after finishing this paragraph.

## Slide 32: Appendix E: custom benchmark connector

Say exactly:

This slide shows how Device Farm actually drives the custom path. The Android benchmark creates xq_options, sets use_mnn_fallback to zero, calls xq_create, then calls xq_generate. After generation, it emits BENCH_RESULT_JSON. That JSON contains the custom path flags, backend information, TPOT, TPS, selected kernels, and per-kernel wall clock. This is why the benchmark result can be audited from artifacts instead of relying on verbal claims.

Advance after finishing this paragraph.

## Slide 33: Appendix F: stock MNN connector

Say exactly:

This slide shows the stock baseline. The stock Android app creates MNN::Transformer::Llm and calls llm->response with the same benchmark settings. It emits stock BENCH_RESULT_JSON and stock BENCH_QUALITY_JSON. This is intentionally separate from the custom runtime. It gives a fair stock baseline, and it also gives a reference for generated-token comparison in the quality validation pipeline.

Advance after finishing this paragraph.

## Slide 34: Appendix G: packer exports custom model assets

Say exactly:

This slide shows how the custom runtime gets its weights. The packer starts with lm_head.weight and then exports attention projection weights, MLP projection weights, and linear-attention tensors. It writes quantization metadata such as bits and group size into the manifest. The runtime loader consumes these xq4 matrices and metadata. This is important because customlib is not only an alternate launcher for MNN. It has its own packed model representation for the custom kernels.

Advance after finishing this paragraph.

## Slide 35: Appendix H: quality validation connector

Say exactly:

This final appendix slide shows the quality connector. BENCH_QUALITY_JSON includes prompt token IDs, generated token IDs, exact match, prefix match length, token match rate, edit distance, invalid-token checks, empty-output checks, and repeated-token checks. This proves that the project treats speed and correctness separately. The quality history includes rejected runs, and the v27 sanity guard passes, but I still do not claim production-quality language-model behavior.

Advance after finishing this paragraph.

## Backup answers

- If asked whether custom beats MNN: No. v27 custom is 2.11989 TPS and fresh stock MNN CPU is 2.26870 TPS, about 93.4% of stock. v17 was the historical 84.9% baseline.
- If asked whether this is only an MNN wrapper: No. Stock baseline and fallback can call MNN::Llm::response, but measured custom generation enters xq_create/xq_generate and CustomModel with use_mnn_fallback=0 and calls_mnn_llm_response=false.
- If asked whether output quality is production-ready: No. Earlier validation found repeated token 220; latest sanity guard improves basic token sanity but does not prove exact token equivalence or production language quality.
- If asked whether Vulkan ran: No custom Vulkan generation kernels ran in the measured custom result. Actual backend is CPU.
