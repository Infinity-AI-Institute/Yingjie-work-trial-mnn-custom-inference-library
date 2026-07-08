# Output Correctness / Quality Guard

TPS/TPOT measure speed, not semantic quality. This validation compares deterministic generated-token dumps from stock MNN and customlib on the same fixed token-ID prompt suite.

## Verdict

- Quality gate passed: YES
- Exact token match for all prompts: NO
- Exact-match prompts: 1 / 5
- Comparison-gate prompts: 5 / 5
- Stock evidence JSON: `results/reports/evidence/quality_validation_v27_stock_english.json`
- Custom evidence JSON: `results/reports/evidence/quality_validation_custom_vulkan_linear_state_english.json`

## Prompt Comparison

| Prompt | Exact | Prefix | Token match | Edit distance | First mismatch | Custom sanity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code_like | no | 14 | 0.438 | 18 | 14 | yes |
| english_explain | no | 10 | 0.312 | 2 | 10 | yes |
| english_factual | no | 16 | 0.500 | 16 | 16 | yes |
| long_512_token_style | yes | 32 | 1.000 | 0 | none | yes |
| math_reasoning | no | 4 | 0.125 | 27 | 4 | yes |

## Token Examples

### code_like

- Stock tokens: `[271, 727, 884, 2784, 11, 292, 1590, 198, 262, 460, 264, 478, 292, 271, 727, 1228, 2797, 4406, 198, 262, 1992, 884, 7, 17, 11, 220, 18, 8, 606, 220, 20, 198]`
- Custom tokens: `[271, 727, 884, 2784, 11, 292, 1590, 198, 262, 460, 264, 478, 292, 271, 7734, 264, 12654, 709, 1146, 2784, 11, 292, 8, 421, 4523, 836, 6463, 13, 271, 727, 1146, 2784]`
- Stock decoded text: `\n\ndef add(a, b):\n    return a + b\n\ndef test_add():\n    assert add(2, 3) == 5\n`

### english_explain

- Stock tokens: `[271, 248068, 271, 248069, 271, 760, 12515, 7701, 6105, 1521, 279, 8964, 579, 16078, 1100, 9872, 22602, 89661, 314, 37728, 318, 11855, 321, 76990, 8, 777, 13057, 1056, 4860, 89661, 4016, 310]`
- Custom tokens: `[271, 248068, 271, 248069, 271, 760, 12515, 7701, 6105, 1521, 8964, 579, 16078, 1100, 9872, 22602, 89661, 314, 37728, 318, 11855, 321, 76990, 8, 777, 13057, 1056, 4860, 89661, 4016, 310, 264]`
- Stock decoded text: `\n\n\n\n\n\nThe sky appears blue because the Earth's atmosphere scatters shorter wavelengths of sunlight (blue and violet) more effectively than longer wavelengths due to`

### english_factual

- Stock tokens: `[271, 248068, 198, 90700, 8340, 25, 198, 16, 13, 220, 2972, 2014, 53983, 279, 5952, 64700, 561, 1156, 369, 9859, 364, 279, 6511, 314, 9338, 321, 6587, 279, 4087, 303, 799, 2716]`
- Custom tokens: `[271, 248068, 198, 90700, 8340, 25, 198, 16, 13, 220, 2972, 2014, 53983, 279, 5952, 64700, 198, 262, 348, 256, 15380, 25, 328, 3710, 369, 279, 6511, 314, 9338, 7285, 198, 262]`
- Stock decoded text: `\n\n\nThinking Process:\n1.  **Analyze the Request:** The user is asking for the capital of France and wants the answer in one short`

### long_512_token_style

- Stock tokens: `[1330, 61446, 31969, 13, 7860, 5437, 537, 279, 2614, 27502, 8129, 303, 1330, 61446, 31969, 13, 7860, 5437, 537, 279, 2614, 27502, 8129, 303, 1330, 61446, 31969, 13, 7860, 5437, 537, 279]`
- Custom tokens: `[1330, 61446, 31969, 13, 7860, 5437, 537, 279, 2614, 27502, 8129, 303, 1330, 61446, 31969, 13, 7860, 5437, 537, 279, 2614, 27502, 8129, 303, 1330, 61446, 31969, 13, 7860, 5437, 537, 279]`
- Stock decoded text: ` two concise bullets. Summarize the following benchmark notes in two concise bullets. Summarize the following benchmark notes in two concise bullets. Summarize the`

### math_reasoning

- Stock tokens: `[271, 20, 248044, 248045, 8944, 248046]`
- Custom tokens: `[271, 20, 248044, 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198, 248068, 198, 90700, 8340, 25, 271, 16, 13, 220, 2972, 2014, 53983, 279, 5952]`
- Stock decoded text: `\n\n5answer`

## Limitations

The custom library is validated here for operator correctness, deterministic decode-path execution, and basic generated-token sanity against stock MNN. It is not claimed as a production-quality model replacement; full semantic quality validation with perplexity and downstream task benchmarks remains future work.
