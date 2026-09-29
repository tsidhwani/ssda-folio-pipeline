"""Batch-download the images listed in a QA sample manifest (see
tools/select_qa_sample.py) through the SSDA retrieval API.

Rate-limited (one request every --delay seconds, default 1/sec), resumable
(skips keys already downloaded), and fault-tolerant (a failed key -- 404,
timeout, expired link -- is logged and the run continues rather than
crashing).

Requires SSDA_API_URL and SSDA_API_KEY, either already exported or sitting
in a .env file in the current directory or the repo root.

Usage:
  python tools/fetch_qa_sample.py qa_samples/sample_manifest.csv --outdir qa_samples/images
  python tools/fetch_qa_sample.py qa_samples/sample_manifest.csv --outdir qa_samples/images --limit 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def load_dotenv_into_environ() -> None:
    for candidate in (Path(".env"), Path(__file__).resolve().parents[1] / ".env"):
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
        return


def fetch_one(api_url: str, api_key: str, key: str, dest: Path, timeout: float = 30.0) -> None:
    query = urllib.parse.urlencode({"key": key})
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/download?{query}",
        headers={"x-api-key": api_key},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    with urllib.request.urlopen(payload["url"], timeout=timeout) as src, open(dest, "wb") as out:
        while chunk := src.read(1 << 20):
            out.write(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="CSV from select_qa_sample.py (needs an s3_key column)")
    ap.add_argument("--outdir", default="qa_samples/images")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--limit", type=int, default=None, help="only fetch the first N rows (for a test run)")
    args = ap.parse_args()

    load_dotenv_into_environ()
    api_url = os.environ.get("SSDA_API_URL")
    api_key = os.environ.get("SSDA_API_KEY")
    if not api_url or not api_key:
        sys.exit("SSDA_API_URL / SSDA_API_KEY not set (checked environment and .env)")

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_path = out_dir.parent / "fetch_failures.csv"
    failures = []

    n_ok = n_skip = n_fail = 0
    for i, row in enumerate(rows, 1):
        key = row["s3_key"]
        dest = out_dir / key
        if dest.exists():
            n_skip += 1
            continue

        try:
            fetch_one(api_url, api_key, key, dest)
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            failures.append({"s3_key": key, "error": str(exc)})
            if dest.exists():
                dest.unlink()  # don't leave a partial/empty file behind
        finally:
            time.sleep(args.delay)

        if i % 50 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] ok={n_ok} skipped={n_skip} failed={n_fail}")

    if failures:
        with failures_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["s3_key", "error"])
            writer.writeheader()
            writer.writerows(failures)
        print(f"failures logged -> {failures_path}")

    print(f"done: {n_ok} fetched, {n_skip} already present, {n_fail} failed (of {len(rows)})")


if __name__ == "__main__":
    main()
