"""S3 data sync helpers for all data-consuming scripts and pipelines."""

import os
from pathlib import Path

DEFAULT_BUCKET = "dongimon-data-storage"

_KEY_MAPPING = {
    "ACCESS_KEY": "AWS_ACCESS_KEY_ID",
    "Secret_Access_Key": "AWS_SECRET_ACCESS_KEY",
}


def load_env_credentials() -> None:
    """Populate boto3 AWS env vars from a repo-root .env file if present.

    Maps ACCESS_KEY to AWS_ACCESS_KEY_ID and Secret_Access_Key to
    AWS_SECRET_ACCESS_KEY only when the standard vars are not already set.
    """
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        mapped = _KEY_MAPPING.get(key.strip())
        if mapped and not os.environ.get(mapped):
            os.environ[mapped] = value.strip()


def sync_from_s3(local_dir: Path, prefix: str, bucket: str = DEFAULT_BUCKET) -> int:
    """Download S3 objects under prefix into local_dir, skipping existing files.

    Relative paths under the prefix are preserved, so nested prefixes
    (e.g. experiments/team_scorer/) land in matching local subdirectories.
    Files that already exist locally are not re-downloaded.

    Args:
        local_dir: Local directory to download into.
        prefix: S3 key prefix to sync (trailing slash recommended).
        bucket: S3 bucket name.

    Returns:
        Number of files downloaded.

    Raises:
        ImportError: If boto3 is not installed.
    """
    import boto3
    from boto3.s3.transfer import TransferConfig

    load_env_credentials()
    s3 = boto3.client("s3")
    transfer = TransferConfig(max_concurrency=8)
    base = prefix.rstrip("/")
    downloaded = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(base) :].lstrip("/")
            if not rel:
                continue
            dest = local_dir / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest), Config=transfer)
            downloaded += 1
    return downloaded
