# Final Vulkan Delivery Attempt: Blocker Report

## Verdict

Outcome B: Vulkan blocked. The repository can be restored from the remote, MNN can be rebuilt, Android APKs can be rebuilt, AWS Device Farm credentials work, and the exact Samsung Galaxy S26 Ultra pool is available. However, the checked-in custom inference library does not contain a real custom Vulkan backend for measured generation. The current custom benchmark still runs the full custom CPU path, and existing v27 evidence explicitly reports `custom_backend_actual = cpu` and `vulkan_generation_kernels_used = false`.

I did not schedule a new "custom_vulkan" final run because the current code would knowingly emit an invalid result for the requested acceptance criteria. A Device Farm run is technically schedulable once model artifacts are uploaded, but it would not be an accepted full-custom Vulkan result unless the missing Vulkan runtime and kernels are implemented first.

The previous official delivery remains the v27 quality-gated CPU customlib systems/kernel benchmark in `results/reports/final_devicefarm_report.md`.

## What Was Reconstructed From Fresh Repo

- Fresh clone source: `https://github.com/Infinity-AI-Institute/Yingjie-work-trial-mnn-custom-inference-library`
- Fresh short working copy used for build: `C:\xqvk`
- HEAD commit: `b22698cca0a5d7c1ce2120917b8af229630392df`
- MNN pinned commit restored: `0bff03cbef43c783f44e41484b9f8a0b28bd758d`
- MNN Android LLM/OpenCL/Vulkan build completed from source.
- Android stock/custom APKs and androidTest APKs rebuilt successfully.
- Host CMake correctness tests passed: 2 / 2.

## Build Environment Fixes

Two checked-in recovery issues were fixed:

- `scripts/fetch_mnn.ps1` now handles an existing empty `third_party/MNN` directory instead of accidentally running `git -C` against the parent repository.
- `scripts/build_mnn_android.sh` now builds into `third_party/MNN/build_android_arm64_llm_sep`, matching the Gradle files and top-level CMake import path.

## Command Evidence

```powershell
git clone https://github.com/Infinity-AI-Institute/Yingjie-work-trial-mnn-custom-inference-library.git C:\xqvk
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fetch_mnn.ps1
bash scripts/build_mnn_android.sh
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_android.ps1
cmake -S . -B build-host-vulkan-attempt -G Ninja -DXQ_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-host-vulkan-attempt
ctest --test-dir build-host-vulkan-attempt --output-on-failure
```

Host test result:

```text
100% tests passed, 0 tests failed out of 2
```

Built APKs:

| Artifact | Size |
| --- | ---: |
| `android/app/build/outputs/apk/debug/stock_mnn_benchmark-debug.apk` | 9,098,102 |
| `android/app/build/outputs/apk/androidTest/debug/stock_mnn_benchmark-debug-androidTest.apk` | 4,479,729 |
| `android/benchmark_app/build/outputs/apk/debug/customlib_benchmark-debug.apk` | 9,229,166 |
| `android/benchmark_app/build/outputs/apk/androidTest/debug/customlib_benchmark-debug-androidTest.apk` | 4,481,349 |

## AWS / Device Farm Status

AWS access is available when using:

- AWS shim: `C:\Users\Yingjie Huang\bin\aws.cmd`
- Profile: `qpnpu-devicefarm`

Verified identity:

```json
{
  "Account": "884244642857",
  "Arn": "arn:aws:iam::884244642857:user/qpnpu-devicefarm"
}
```

Device Farm project:

```text
arn:aws:devicefarm:us-west-2:884244642857:project:64d2cc31-abd6-49f8-97da-162f82410bc0
```

Exact Samsung Galaxy S26 Ultra pool:

```text
arn:aws:devicefarm:us-west-2:884244642857:devicepool:64d2cc31-abd6-49f8-97da-162f82410bc0/14d31c96-b8fc-4930-99c7-1a8948124213
```

Device availability was reported as `HIGHLY_AVAILABLE`.

## Why The Vulkan Final Is Blocked

### 1. No custom Vulkan backend is checked in

The only customlib path under `customlib/kernels/generated/vulkan` is:

```text
customlib/kernels/generated/vulkan/.gitkeep
```

There are no checked-in custom Vulkan shader sources, SPIR-V blobs, Vulkan memory manager, custom Vulkan dispatch path, W4A16 Vulkan GEMV implementation, Vulkan attention kernels, Vulkan linear-attention kernels, or Vulkan lm_head/argmax kernels.

MNN's `libMNN_Vulkan.so` is built and packaged, but that is the stock MNN backend library. It is not a customlib Vulkan execution path.

### 2. Custom benchmark hard-codes CPU as the actual backend

Evidence:

- `android/benchmark_app/src/main/cpp/benchmark_jni.cpp:432` reports `full_or_hybrid_kernel_status = not_enabled_in_this_build`.
- `android/benchmark_app/src/main/cpp/benchmark_jni.cpp:433` reports that custom Vulkan W4A16/lm_head kernels were probed but not selected.
- `android/benchmark_app/src/main/cpp/benchmark_jni.cpp:1224` sets benchmark `actual_backend = "cpu"`.
- `android/benchmark_app/src/main/cpp/benchmark_jni.cpp:1347` sets quality validation `actual_backend = "cpu"`.

### 3. Existing final evidence explicitly says CPU

Evidence:

- `results/reports/evidence/customlib_cpu_vulkan_hybrid_benchmark_v27.json` reports:
  - `backend_actual = cpu`
  - `custom_backend_actual = cpu`
  - `vulkan_generation_kernels_used = false`
  - `op_family_backends` are CPU for all major measured op families.
- `results/reports/final_devicefarm_report.md:10` says no custom Vulkan generation kernels were used.
- `docs/kernel_library_code_walkthrough_final.md:161` says the v27 run probed Vulkan but used CPU.

## Missing Work Before A Valid Outcome A

To honestly reach the requested final Vulkan acceptance criteria, the repository needs a real custom Vulkan backend implementing and validating at least:

- W4A16 GEMV Vulkan for q/k/v/o, gate/up/down, linear-attention projections, and lm_head.
- Vulkan-resident weights and activation/state buffers.
- RMSNorm Vulkan.
- Qwen3.5 partial RoPE Vulkan.
- GQA attention score, stable softmax, V reduce, and KV cache append/read on Vulkan.
- Linear-attention recurrent state and gate semantics on Vulkan.
- lm_head + greedy argmax from real logits on Vulkan.
- Per-op trace rows with `backend = vulkan`.
- Android/Device Farm correctness tests comparing Vulkan kernels against CPU/reference.
- Device Farm full-model benchmark and quality validation producing `BENCH_RESULT_JSON` and `BENCH_QUALITY_JSON`.

## Requirement Status After This Attempt

| Requirement | Status | Evidence |
| --- | --- | --- |
| Requirement 1: custom kernel library + walkthrough | Still satisfied by v27 CPU final | `docs/kernel_library_code_walkthrough_final.md` |
| Requirement 2: MNN comparison with per-kernel wall clock, TPOT, TPS | Still satisfied by v27 CPU final | `results/reports/final_devicefarm_report.md` |
| Requirement 3: output quality guard | Still satisfied for v27 CPU final | `results/reports/quality_validation_report.md` |
| Requirement 4: full custom Vulkan, target ~10 TPS | Not met | No custom Vulkan backend exists; current measured custom backend is CPU |

## Final Decision

No accepted custom Vulkan final result was produced in this pass. I did not overwrite the v27 final report because doing so would either degrade the official accepted result or falsely claim Vulkan execution.

Previous official result preserved:

- Final report: `results/reports/final_devicefarm_report.md`
- Code walkthrough: `docs/kernel_library_code_walkthrough_final.md`
- Quality report: `results/reports/quality_validation_report.md`

