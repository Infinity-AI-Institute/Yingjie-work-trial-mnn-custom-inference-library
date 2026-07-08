package com.example.xqwen35bench;

public final class NativeBenchmark {
    static {
        System.loadLibrary("xqbench_jni");
    }

    private NativeBenchmark() {}

    public static native String runBenchmark(
            String modelDir,
            String customBackend,
            int promptTokens,
            int maxNewTokens,
            int warmupIterations,
            int measuredIterations);

    public static native String runQualityValidation(
            String modelDir,
            String customBackend,
            int maxNewTokens);

    public static native String runVulkanSelfTest();
}
