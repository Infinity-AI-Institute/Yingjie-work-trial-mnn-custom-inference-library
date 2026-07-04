#!/usr/bin/env python3
import argparse
import json
import pathlib
import statistics
from typing import Any, Dict, List


def collect(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for f in path.rglob("*"):
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        if f.suffix == ".json":
            try:
                data = json.loads(text)
                if isinstance(data, dict) and "engine" in data:
                    key = json.dumps(data, sort_keys=True)
                    if key not in seen:
                        seen.add(key)
                        rows.append(data)
            except json.JSONDecodeError:
                pass
        marker = "BENCH_RESULT_JSON "
        for line in text.splitlines():
            idx = line.find(marker)
            if idx >= 0:
                try:
                    data = json.loads(line[idx + len(marker):])
                    key = json.dumps(data, sort_keys=True)
                    if key not in seen:
                        seen.add(key)
                        rows.append(data)
                except json.JSONDecodeError:
                    pass
    return rows


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"median": 0.0, "min": 0.0, "max": 0.0, "p90": 0.0}
    s = sorted(values)
    return {
        "median": statistics.median(s),
        "min": min(s),
        "max": max(s),
        "p90": s[min(len(s) - 1, int(0.9 * (len(s) - 1)))],
    }


def decode_median(row: Dict[str, Any]) -> float:
    decode = row.get("tokens_per_second", {}).get("decode")
    if isinstance(decode, (int, float)):
        return float(decode)
    if isinstance(decode, dict) and isinstance(decode.get("median"), (int, float)):
        return float(decode["median"])
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-md", default="results/reports/final_devicefarm_report.md")
    parser.add_argument("--out-json", default="results/reports/final_devicefarm_report.json")
    args = parser.parse_args()
    rows = collect(pathlib.Path(args.raw_dir))
    by_engine: Dict[str, List[float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        decode = decode_median(row)
        if decode > 0.0:
            by_engine.setdefault(row.get("engine", "unknown"), []).append(decode)
    stock = stats(by_engine.get("mnn_stock", []))
    custom = stats(by_engine.get("customlib", []))
    speedup = custom["median"] / stock["median"] if stock["median"] > 0 else 0.0
    report = {
        "schema_version": 1,
        "status": "complete" if rows and stock["median"] > 0 and custom["median"] > 0 else "blocked_or_incomplete",
        "stock_mnn": stock,
        "customlib": custom,
        "speedup_ratio": speedup,
        "target_20_decode_tps_passed": custom["median"] >= 20.0,
        "raw_dir": args.raw_dir,
        "runs": rows,
    }
    pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        "# Final Device Farm Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "| Engine | Median decode tok/s | Min | Max | P90 |",
        "|---|---:|---:|---:|---:|",
        f"| stock MNN | {stock['median']:.4f} | {stock['min']:.4f} | {stock['max']:.4f} | {stock['p90']:.4f} |",
        f"| customlib | {custom['median']:.4f} | {custom['min']:.4f} | {custom['max']:.4f} | {custom['p90']:.4f} |",
        "",
        f"Speedup ratio: `{speedup:.4f}x`",
        f">=20 decode tok/s: `{'PASS' if custom['median'] >= 20.0 else 'FAIL'}`",
        "",
        f"Raw artifacts: `{args.raw_dir}`",
        "",
    ]
    pathlib.Path(args.out_md).write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"out_md": args.out_md, "out_json": args.out_json, "status": report["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
