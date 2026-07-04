#!/usr/bin/env python3
import argparse
import json
import pathlib
import shutil
import struct
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from safetensors import safe_open


PINNED_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
HF_REPO = "Qwen/Qwen3.5-9B"
MAGIC = b"XQW4A16\0"


def read_config(model_dir: pathlib.Path) -> Dict:
    data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return data.get("text_config", data)


def safe_name(name: str) -> str:
    return name.replace("model.language_model.", "").replace(".", "_")


def load_weight_map(model_dir: pathlib.Path) -> Dict[str, str]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return index["weight_map"]


def tensors_for_pack(config: Dict, limit_layers: int = 0) -> Tuple[List[str], List[str]]:
    n_layers = int(config["num_hidden_layers"])
    if limit_layers > 0:
        n_layers = min(n_layers, limit_layers)
    layer_types = config["layer_types"]
    matrix_names: List[str] = []
    vector_names: List[str] = ["model.language_model.norm.weight"]
    for layer in range(n_layers):
        prefix = f"model.language_model.layers.{layer}"
        vector_names.append(f"{prefix}.input_layernorm.weight")
        vector_names.append(f"{prefix}.post_attention_layernorm.weight")
        matrix_names.extend(
            [
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        )
        if layer_types[layer] == "full_attention":
            matrix_names.extend(
                [
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight",
                    f"{prefix}.self_attn.o_proj.weight",
                ]
            )
            vector_names.append(f"{prefix}.self_attn.q_norm.weight")
            vector_names.append(f"{prefix}.self_attn.k_norm.weight")
        else:
            matrix_names.extend(
                [
                    f"{prefix}.linear_attn.in_proj_qkv.weight",
                    f"{prefix}.linear_attn.in_proj_z.weight",
                    f"{prefix}.linear_attn.out_proj.weight",
                ]
            )
            vector_names.append(f"{prefix}.linear_attn.norm.weight")
    return matrix_names, vector_names


def copy_small_files(model_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def load_tensor(model_dir: pathlib.Path, weight_map: Dict[str, str], name: str) -> torch.Tensor:
    shard = weight_map[name]
    with safe_open(model_dir / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def quantize_matrix_w4(tensor: torch.Tensor, group_size: int, dst: pathlib.Path) -> Dict:
    rows, cols = [int(v) for v in tensor.shape]
    if cols % group_size != 0:
        raise ValueError(f"{dst.name}: cols={cols} must be divisible by group_size={group_size}")
    arr = tensor.to(torch.float32).contiguous().numpy()
    groups = cols // group_size
    packed_rows: List[np.ndarray] = []
    scales = np.empty((rows, groups), dtype=np.float32)
    zeros = np.empty((rows, groups), dtype=np.float32)
    for r0 in range(0, rows, 32):
        chunk = arr[r0 : r0 + 32].reshape(-1, groups, group_size)
        mn = chunk.min(axis=2)
        mx = chunk.max(axis=2)
        scale = (mx - mn) / 15.0
        scale = np.where(scale > 0.0, scale, 1.0).astype(np.float32)
        zero = np.rint(-mn / scale).clip(0.0, 15.0).astype(np.float32)
        codes = np.rint(chunk / scale[..., None] + zero[..., None]).clip(0, 15).astype(np.uint8)
        pair = codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)
        packed_rows.append(pair.reshape(pair.shape[0], -1).copy())
        scales[r0 : r0 + chunk.shape[0]] = scale
        zeros[r0 : r0 + chunk.shape[0]] = zero
    packed = np.concatenate(packed_rows, axis=0)
    header = struct.pack(
        "<8sIIIIIQQQ",
        MAGIC,
        1,
        rows,
        cols,
        group_size,
        4,
        int(packed.nbytes),
        int(scales.size),
        int(zeros.size),
    )
    with dst.open("wb") as f:
        f.write(header)
        f.write(packed.tobytes(order="C"))
        f.write(scales.tobytes(order="C"))
        f.write(zeros.tobytes(order="C"))
    return {
        "file": dst.name,
        "rows": rows,
        "cols": cols,
        "bits": 4,
        "group_size": group_size,
        "packed_bytes": int(packed.nbytes),
        "scale_count": int(scales.size),
        "zero_count": int(zeros.size),
    }


def write_vector_f32(tensor: torch.Tensor, dst: pathlib.Path) -> Dict:
    arr = tensor.to(torch.float32).contiguous().numpy().astype(np.float32, copy=False)
    with dst.open("wb") as f:
        f.write(arr.tobytes(order="C"))
    return {"file": dst.name, "elements": int(arr.size), "dtype": "float32"}


def pack_tensors(model_dir: pathlib.Path, out_dir: pathlib.Path, names: Iterable[str], group_size: int) -> Dict[str, Dict]:
    weight_map = load_weight_map(model_dir)
    out: Dict[str, Dict] = {}
    for i, name in enumerate(names, 1):
        tensor = load_tensor(model_dir, weight_map, name)
        dst = out_dir / f"{safe_name(name)}.xq4"
        print(f"[{i}] pack {name} shape={tuple(tensor.shape)} -> {dst}")
        out[name] = quantize_matrix_w4(tensor, group_size, dst)
        del tensor
    return out


def pack_vectors(model_dir: pathlib.Path, out_dir: pathlib.Path, names: Iterable[str]) -> Dict[str, Dict]:
    weight_map = load_weight_map(model_dir)
    out: Dict[str, Dict] = {}
    for name in names:
        if name not in weight_map:
            continue
        tensor = load_tensor(model_dir, weight_map, name)
        dst = out_dir / f"{safe_name(name)}.f32"
        print(f"pack {name} shape={tuple(tensor.shape)} -> {dst}")
        out[name] = write_vector_f32(tensor, dst)
        del tensor
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--limit-layers", type=int, default=0)
    parser.add_argument("--copy-mnn-from", default="")
    args = parser.parse_args()

    model_dir = pathlib.Path(args.model_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_small_files(model_dir, out_dir)
    if args.copy_mnn_from:
        src = pathlib.Path(args.copy_mnn_from)
        for name in ("llm.mnn", "llm.mnn.weight", "llm.mnn.json", "llm_config.json", "tokenizer.mtok", "embeddings_bf16.bin"):
            if (src / name).exists() and not (out_dir / name).exists():
                shutil.copy2(src / name, out_dir / name)

    config = read_config(model_dir)
    matrix_names, vector_names = tensors_for_pack(config, args.limit_layers)
    matrices = pack_tensors(model_dir, out_dir, matrix_names, args.group_size)
    vectors = pack_vectors(model_dir, out_dir, vector_names)
    manifest = {
        "hf_repo": HF_REPO,
        "hf_revision": PINNED_REVISION,
        "format": "xqwen35_custom_w4a16",
        "quantization": {"bits": 4, "group_size": args.group_size, "scheme": "w4a16_groupwise_asymmetric"},
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
        "num_hidden_layers": int(args.limit_layers or config["num_hidden_layers"]),
        "num_attention_heads": int(config["num_attention_heads"]),
        "num_key_value_heads": int(config["num_key_value_heads"]),
        "head_dim": int(config["head_dim"]),
        "vocab_size": int(config["vocab_size"]),
        "layer_types": config["layer_types"][: int(args.limit_layers or config["num_hidden_layers"])],
        "rms_norm_eps": float(config["rms_norm_eps"]),
        "rope_theta": float(config["rope_parameters"]["rope_theta"]),
        "matrices": matrices,
        "vectors": vectors,
    }
    (out_dir / "xqwen35_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "matrices": len(matrices), "vectors": len(vectors)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
