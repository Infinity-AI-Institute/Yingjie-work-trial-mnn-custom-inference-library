#!/usr/bin/env python3
import argparse
import http.client
import json
import mimetypes
import pathlib
import shutil
import subprocess
import time
import urllib.parse

from common import add_region_arg, aws


def redact_upload(upload):
    safe = dict(upload)
    url = safe.get("url")
    if isinstance(url, str) and "?" in url:
        safe["url"] = url.split("?", 1)[0] + "?REDACTED"
    return safe


def put_file(url: str, path: pathlib.Path, content_type: str) -> None:
    parsed = urllib.parse.urlparse(url)
    target = urllib.parse.urlunparse(("", "", parsed.path, parsed.params, parsed.query, parsed.fragment))
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parsed.netloc, timeout=300)
    size = path.stat().st_size
    sent = 0
    next_log = 256 * 1024 * 1024
    try:
        conn.putrequest("PUT", target)
        conn.putheader("Content-Type", content_type)
        conn.putheader("Content-Length", str(size))
        conn.endheaders()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
                conn.send(chunk)
                sent += len(chunk)
                if sent >= next_log:
                    print(json.dumps({"upload_progress_bytes": sent, "upload_total_bytes": size}), flush=True)
                    next_log += 256 * 1024 * 1024
        resp = conn.getresponse()
        body = resp.read(4096)
        if resp.status < 200 or resp.status >= 300:
            raise RuntimeError(f"PUT upload failed status={resp.status} reason={resp.reason} body={body[:512]!r}")
    finally:
        conn.close()


def put_file_curl(url: str, path: pathlib.Path, content_type: str) -> None:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl executable not found")
    cmd = [
        curl,
        "--fail-with-body",
        "--retry",
        "3",
        "--retry-all-errors",
        "--connect-timeout",
        "60",
        "--max-time",
        "0",
        "--upload-file",
        str(path),
        "-H",
        f"Content-Type: {content_type}",
        url,
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"curl upload failed with exit code {proc.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    add_region_arg(parser)
    parser.add_argument("--project-arn", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--type", required=True, help="ANDROID_APP, INSTRUMENTATION_TEST_PACKAGE, APPIUM_JAVA_JUNIT_TEST_PACKAGE, EXTERNAL_DATA, ...")
    parser.add_argument("--name")
    parser.add_argument("--wait-seconds", type=int, default=1800)
    parser.add_argument("--put-method", choices=("curl", "python"), default="curl")
    args = parser.parse_args()

    path = pathlib.Path(args.path)
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    upload = aws(
        [
            "devicefarm",
            "create-upload",
            "--project-arn",
            args.project_arn,
            "--name",
            args.name or path.name,
            "--type",
            args.type,
            "--content-type",
            content_type,
        ],
        region=args.region,
    ).get("upload", {})
    url = upload["url"]
    print(json.dumps(redact_upload(upload), indent=2), flush=True)
    if args.put_method == "curl":
        put_file_curl(url, path, content_type)
    else:
        put_file(url, path, content_type)

    arn = upload["arn"]
    deadline = time.time() + args.wait_seconds
    while time.time() < deadline:
        cur = aws(["devicefarm", "get-upload", "--arn", arn], region=args.region).get("upload", {})
        if cur.get("status") in ("SUCCEEDED", "FAILED"):
            print(json.dumps(redact_upload(cur), indent=2))
            return 0 if cur.get("status") == "SUCCEEDED" else 4
        time.sleep(5)
    raise SystemExit("Timed out waiting for Device Farm upload processing")


if __name__ == "__main__":
    raise SystemExit(main())
