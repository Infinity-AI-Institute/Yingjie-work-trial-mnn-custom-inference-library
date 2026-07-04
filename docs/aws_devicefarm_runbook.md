# AWS Device Farm Runbook

## Credentials

Use the AWS CLI default credential chain, `AWS_PROFILE`, environment credentials, or standard `~/.aws/config` and `~/.aws/credentials` profiles. Do not print, commit, upload, or bake credentials into APKs.

Validate identity:

```bash
python scripts/aws/check_credentials.py --region us-west-2
```

## Exact Device Selection

Final validation requires exact Samsung Galaxy S26 Ultra availability through AWS Device Farm APIs:

```bash
python scripts/aws/list_devicefarm_devices.py --region us-west-2
```

If no exact match is returned, final validation is blocked. Do not substitute another phone for final numbers.

## Project and Device Pool

```bash
python scripts/aws/create_or_get_project.py --region us-west-2
python scripts/aws/create_exact_device_pool.py --project-arn PROJECT_ARN --region us-west-2
```

The exact device pool must contain only matching Galaxy S26 Ultra device ARNs.

## Artifact Upload

Upload app and instrumentation APKs:

```bash
python scripts/aws/upload_artifact.py --project-arn PROJECT_ARN --path app.apk --type ANDROID_APP
python scripts/aws/upload_artifact.py --project-arn PROJECT_ARN --path app-androidTest.apk --type INSTRUMENTATION_TEST_PACKAGE
```

If Qwen3.5-9B artifacts exceed the single Device Farm/S3 upload limit, split the zip into parts smaller than the observed 5,368,709,120 byte cap and upload each part as `EXTERNAL_DATA`. Pass the resulting upload URLs through a pushed `model_zip_part_urls_file`; Android `ModelBootstrap` streams the parts back into one zip and verifies the full zip sha256 before unzip. Redact URL query strings from logs.

## Run and Artifacts

```bash
python scripts/aws/schedule_benchmark_run.py --project-arn PROJECT_ARN --app-arn APP_ARN --test-package-arn TEST_ARN --device-pool-arn POOL_ARN
python scripts/aws/wait_for_run.py --run-arn RUN_ARN
python scripts/aws/download_artifacts.py --run-arn RUN_ARN
python scripts/aws/make_report.py --raw-dir results/raw/RUN_DIR
```

The instrumentation test prints `BENCH_RESULT_JSON` to logcat and writes JSON to app external files where possible.
