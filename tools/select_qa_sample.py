"""Pick a random QA sample of sacramental-record page keys from volumes.json.

volumes.json (from the ssda-archivault repo) lists every volume with an
`identifier` and an `images` count. Raw scan keys are not listable via the
retrieval API, but every valid key is `{identifier}-{page:04d}.jpg` for
page in 1..images, so the full key space can be generated locally from this
one file.

This script:
  1. Filters volumes.json to sacramental records only (subject contains
     "marriage", "baptism", or "burial", case-insensitive).
  2. Randomly samples pages across those volumes, capping how many pages any
     single volume can contribute (so one large register can't dominate the
     QA set).
  3. Writes a CSV manifest of S3 keys + volume metadata (no images fetched
     here -- that's a separate, rate-limited step against fetch.py).

Usage:
  python tools/select_qa_sample.py --volumes-json ../ssda-archivault/volumes.json
  python tools/select_qa_sample.py --sample-size 5000 --cap 30 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

SACRAMENTAL_PATTERN = re.compile(r"marriage|baptism|burial", re.IGNORECASE)


def load_sacramental_volumes(volumes_json: Path) -> list[dict]:
    records = json.loads(volumes_json.read_text(encoding="utf-8"))
    matches = []
    for rec in records:
        fields = rec.get("fields", {})
        subjects = fields.get("subject", [])
        if any(SACRAMENTAL_PATTERN.search(s) for s in subjects):
            matches.append(fields)
    return matches


def select_sample(volumes: list[dict], sample_size: int, cap: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = list(volumes)
    rng.shuffle(pool)

    rows = []
    for vol in pool:
        if len(rows) >= sample_size:
            break
        identifier = vol["identifier"]
        n_images = vol.get("images", 0)
        if not n_images:
            continue
        take = min(cap, n_images, sample_size - len(rows))
        pages = rng.sample(range(1, n_images + 1), take)
        for page in pages:
            rows.append({
                "s3_key": f"{identifier}-{page:04d}.jpg",
                "volume_id": identifier,
                "page": page,
                "images_in_volume": n_images,
                "title": vol.get("title", ""),
                "subjects": ";".join(vol.get("subject", [])),
                "language": ";".join(vol.get("language", [])),
                "country": vol.get("country", ""),
                "institution": vol.get("institution", ""),
                "start_date": vol.get("start_date", ""),
                "end_date": vol.get("end_date", ""),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes-json",
                     default=str(Path(__file__).resolve().parents[2] / "ssda-archivault" / "volumes.json"),
                     help="path to volumes.json (default: sibling ssda-archivault repo)")
    ap.add_argument("--sample-size", type=int, default=5000)
    ap.add_argument("--cap", type=int, default=30,
                     help="max pages drawn from any single volume")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed, for a reproducible sample")
    ap.add_argument("--out", default="qa_samples/sample_manifest.csv")
    args = ap.parse_args()

    volumes_json = Path(args.volumes_json)
    if not volumes_json.exists():
        raise SystemExit(f"volumes.json not found at {volumes_json} -- pass --volumes-json explicitly")

    volumes = load_sacramental_volumes(volumes_json)
    total_images = sum(v.get("images", 0) for v in volumes)
    print(f"sacramental volumes: {len(volumes)}  total pages: {total_images}")

    rows = select_sample(volumes, args.sample_size, args.cap, args.seed)
    used_volumes = len({r["volume_id"] for r in rows})
    print(f"sampled {len(rows)} keys from {used_volumes} volumes "
          f"(cap={args.cap}, seed={args.seed})")
    if len(rows) < args.sample_size:
        print(f"WARNING: exhausted the sacramental pool at {len(rows)} keys "
              f"(< requested {args.sample_size})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"manifest -> {out_path}")


if __name__ == "__main__":
    main()
