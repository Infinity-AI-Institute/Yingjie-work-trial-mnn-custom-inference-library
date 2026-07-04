#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import time
import zipfile


FILES = [
    "config.json",
    "export_args.json",
    "llm_config.json",
    "llm.mnn",
    "llm.mnn.weight",
    "embeddings_bf16.bin",
    "tokenizer.mtok",
]

EMBEDDINGS_BF16_BYTES = 2034237440


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--zip-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--omit-embeddings", action="store_true")
    args = parser.parse_args()

    model_dir = pathlib.Path(args.model_dir)
    zip_out = pathlib.Path(args.zip_out)
    manifest_out = pathlib.Path(args.manifest_out)
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    files = [name for name in FILES if not (args.omit_embeddings and name == "embeddings_bf16.bin")]
    missing = [name for name in files if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing model files: {missing}")

    started = time.time()
    entries = []
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name in files:
            path = model_dir / name
            print(f"adding {name} bytes={path.stat().st_size}", flush=True)
            zf.write(path, arcname=name)
            entries.append(
                {
                    "name": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    zip_sha = sha256_file(zip_out)
    manifest = {
        "schema_version": 1,
        "format": "zip64_stored",
        "model_dir": str(model_dir),
        "zip_path": str(zip_out),
        "zip_bytes": zip_out.stat().st_size,
        "zip_sha256": zip_sha,
        "elapsed_s": time.time() - started,
        "omit_embeddings": args.omit_embeddings,
        "device_sparse_embeddings_bf16_bytes": EMBEDDINGS_BF16_BYTES if args.omit_embeddings else None,
        "entries": entries,
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"zip_path": str(zip_out), "zip_bytes": zip_out.stat().st_size, "zip_sha256": zip_sha}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
