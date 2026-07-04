#!/usr/bin/env python3
import argparse
import json
import pathlib
import urllib.request

from common import add_region_arg, aws


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:180]


def safe_extension(value: str) -> str:
    if not value:
        return ""
    cleaned = safe_name(value).lower()
    if not cleaned:
        return ""
    if cleaned.startswith("."):
        return cleaned
    if "." in cleaned:
        return pathlib.Path(cleaned).suffix or ("." + cleaned.rsplit(".", 1)[-1])
    return "." + cleaned


def redact_artifact(artifact):
    safe = dict(artifact)
    url = safe.get("url")
    if isinstance(url, str) and "?" in url:
        safe["url"] = url.split("?", 1)[0] + "?REDACTED"
    return safe


def main() -> int:
    parser = argparse.ArgumentParser()
    add_region_arg(parser)
    parser.add_argument("--run-arn", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir or ("results/raw/" + safe_name(args.run_arn)))
    out_dir.mkdir(parents=True, exist_ok=True)

    all_artifacts = []
    index = 0
    for artifact_type in ("FILE", "LOG", "SCREENSHOT"):
        try:
            artifacts = aws(["devicefarm", "list-artifacts", "--arn", args.run_arn, "--type", artifact_type], region=args.region).get("artifacts", [])
        except Exception:
            artifacts = []
        for artifact in artifacts:
            index += 1
            all_artifacts.append(redact_artifact(artifact))
            url = artifact.get("url")
            if not url:
                continue
            name = safe_name(artifact.get("name") or artifact.get("arn") or artifact_type)
            kind = safe_name(artifact.get("type") or artifact_type)
            suffix = safe_extension(artifact.get("extension") or "")
            target = out_dir / (f"{index:02d}_{name}_{kind}" + suffix)
            with urllib.request.urlopen(url, timeout=120) as resp:
                target.write_bytes(resp.read())
    (out_dir / "artifacts.json").write_text(json.dumps(all_artifacts, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "artifact_count": len(all_artifacts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
