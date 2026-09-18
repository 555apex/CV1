# -*- coding: utf-8 -*-
"""Download a reproducible, small SmartDoc release archive in byte ranges.

The SmartDoc sample release is a public archive, but it is too large to keep in
the project repository.  This helper downloads it to a user-selected location
and can be rerun safely after an interrupted download.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request


ASSET_API = (
    "https://api.github.com/repos/smartdoc2017-competition/"
    "sample_dataset/releases/assets/3501413"
)
DEFAULT_SIZE = 469_290_432


def _asset_size() -> int:
    req = urllib.request.Request(
        ASSET_API,
        headers={"User-Agent": "ComputerVision-course-project", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        obj = json.load(response)
    return int(obj.get("size", DEFAULT_SIZE))


def download_ranges(path: str, chunk_size: int = 8 * 1024 * 1024, retries: int = 4) -> str:
    total = _asset_size()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    current = os.path.getsize(path) if os.path.isfile(path) else 0
    if current > total:
        raise ValueError(f"existing file is larger than asset: {current} > {total}")

    with open(path, "ab") as output:
        while current < total:
            end = min(current + chunk_size - 1, total - 1)
            last_error = None
            for attempt in range(retries):
                try:
                    req = urllib.request.Request(
                        ASSET_API,
                        headers={
                            "User-Agent": "ComputerVision-course-project",
                            "Accept": "application/octet-stream",
                            "Range": f"bytes={current}-{end}",
                        },
                    )
                    with urllib.request.urlopen(req, timeout=120) as response:
                        body = response.read()
                    expected = end - current + 1
                    if len(body) != expected:
                        raise IOError(f"short range: expected {expected}, got {len(body)}")
                    output.write(body)
                    output.flush()
                    current = end + 1
                    print(f"downloaded {current / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MiB", flush=True)
                    break
                except Exception as exc:  # network interruptions are retryable
                    last_error = exc
                    time.sleep(1.5 * (attempt + 1))
            else:
                raise RuntimeError(f"failed range {current}-{end}: {last_error}") from last_error
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="local .tar.gz path")
    args = parser.parse_args()
    result = download_ranges(args.output)
    print(f"saved: {result} ({os.path.getsize(result)} bytes)")


if __name__ == "__main__":
    main()
