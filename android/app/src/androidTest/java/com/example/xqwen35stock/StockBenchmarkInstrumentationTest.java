package com.example.xqwen35stock;

import android.content.Context;
import android.util.Log;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public final class StockBenchmarkInstrumentationTest {
    @Test
    public void runStockBenchmark() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        File root = context.getExternalFilesDir(null);
        File dir = ModelBootstrap.resolveModelDir(context);
        String json = NativeStockBenchmark.runBenchmark(dir.getAbsolutePath(), 512, 256);
        Log.i("XQBENCH", "BENCH_RESULT_JSON " + json);
        if (root != null) {
            File artifactDir = new File(root, "bench_artifacts");
            if (!artifactDir.mkdirs() && !artifactDir.isDirectory()) {
                throw new IllegalStateException("failed to create " + artifactDir.getAbsolutePath());
            }
            try (FileOutputStream out = new FileOutputStream(new File(artifactDir, "stock_mnn_benchmark.json"))) {
                out.write(json.getBytes(StandardCharsets.UTF_8));
            }
        }
        if (!json.contains("\"status\":\"ok\"")) {
            throw new AssertionError("Benchmark returned non-ok JSON: " + json);
        }
    }
}
