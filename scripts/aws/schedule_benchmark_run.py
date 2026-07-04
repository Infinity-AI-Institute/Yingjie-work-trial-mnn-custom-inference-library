#!/usr/bin/env python3
import argparse
import json
from typing import Dict, List

from common import add_region_arg, aws


def parse_key_value(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"empty key in {item!r}")
        out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    add_region_arg(parser)
    parser.add_argument("--project-arn", required=True)
    parser.add_argument("--app-arn", required=True)
    parser.add_argument("--test-package-arn", required=True)
    parser.add_argument("--test-spec-arn")
    parser.add_argument("--device-pool-arn", required=True)
    parser.add_argument("--name", default="qwen35-9b-stock-and-customlib")
    parser.add_argument("--extra-data-arn")
    parser.add_argument("--job-timeout-minutes", type=int, default=150)
    parser.add_argument("--artifact-path", action="append", default=[])
    parser.add_argument("--test-parameter", action="append", default=[])
    parser.add_argument("--env-var", action="append", default=[])
    args = parser.parse_args()

    test_spec = {
        "type": "INSTRUMENTATION",
        "testPackageArn": args.test_package_arn,
    }
    if args.test_spec_arn:
        test_spec["testSpecArn"] = args.test_spec_arn
    parameters = parse_key_value(args.test_parameter)
    if parameters:
        test_spec["parameters"] = parameters

    configuration = {}
    if args.extra_data_arn:
        configuration["extraDataPackageArn"] = args.extra_data_arn
    if args.artifact_path:
        configuration["customerArtifactPaths"] = {"androidPaths": args.artifact_path}
    env_vars = parse_key_value(args.env_var)
    if env_vars:
        configuration["environmentVariables"] = [
            {"name": key, "value": value} for key, value in env_vars.items()
        ]
    execution_configuration = {"jobTimeoutMinutes": args.job_timeout_minutes}

    command = [
        "devicefarm",
        "schedule-run",
        "--project-arn",
        args.project_arn,
        "--app-arn",
        args.app_arn,
        "--device-pool-arn",
        args.device_pool_arn,
        "--name",
        args.name,
        "--test",
        json.dumps(test_spec, separators=(",", ":")),
        "--configuration",
        json.dumps(configuration, separators=(",", ":")),
        "--execution-configuration",
        json.dumps(execution_configuration, separators=(",", ":")),
    ]
    run = aws(command, region=args.region).get("run", {})
    print(json.dumps(run, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
