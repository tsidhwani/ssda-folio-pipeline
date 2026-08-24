#!/usr/bin/env python3
"""run_prefix.py — one command, one changing parameter: the source key prefix.

    python scripts/run_prefix.py 176899
    python scripts/run_prefix.py 176899 201991 ...        # several volumes, in order

For each prefix this does, end to end:

  1. CROP    every source image under s3://<source.bucket>/<prefix> with the folio
             pipeline, pushing crops to the crops bucket. Image bytes are streamed
             in memory and never written to local disk.
  2. SUBMIT  the crop keys to the Archivault API using its *S3 import* flow, so the
             API copies the crops server-side out of the crops bucket. Nothing is
             re-uploaded from this machine.
  3. ARCHIVE the returned artifacts straight into the transcripts bucket (streamed
             from the presigned URLs), plus a manifest describing the run.

Everything that does not change between runs lives in `pipeline.toml` (buckets,
AWS profile names, models, steps, transcription instructions). No secrets there:
AWS auth is by profile name, and the Archivault password comes from
$ARCHIVAULT_PASSWORD or an interactive prompt.

Key naming
----------
The source bucket is flat (`{volume}-{0000}.jpg`) and the crops bucket mirrors it
exactly. A single-folio image keeps its key verbatim; a two-folio spread splits
into `<stem>A.jpg` (left / recto) and `<stem>B.jpg` (right / verso).

Resume
------
A prefix that is re-run skips any source image whose crops are already in the
crops bucket, so an interrupted run costs only the images it had not reached.
Pass --no-resume to force a full re-crop, or --skip-crop / --skip-submit to run a
single stage.

Examples
--------
  python scripts/run_prefix.py 176899 --dry-run      # list work, spend nothing
  python scripts/run_prefix.py 176899 --skip-submit  # crop only
  python scripts/run_prefix.py --config other.toml 176899
"""
import argparse
import getpass
import io
import json
import os
import posixpath
import sys
import time
import uuid
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import boto3
import cv2
import numpy as np
import requests

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from folio.process import make_config, build_pipeline, find_legacy_weights  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
DEFAULT_CONFIG = _REPO / "pipeline.toml"

# The only artifacts we keep, and the extension each one gets in the transcripts
# bucket. Keyed by the artifact name the API uses in its `artifacts` block. Anything
# absent from this map (notably `tables_zip`) is discarded.
ARTIFACT_EXT = {"json": ".json", "markdown": ".md"}
ARTIFACT_TYPE = {"json": "application/json", "markdown": "text/markdown"}

# Run-record key prefixes inside the transcripts bucket. The transcripts themselves
# stay at <prefix><volume>.json/.md; these two sit under their own folders so a
# listing of the bucket root is just transcripts.
MANIFEST_DIR = "manifest/"
REVIEW_DIR = "review/"


def _log(msg):
    print(f"[*] {msg}", flush=True)


def _warn(msg):
    print(f"[!] {msg}", flush=True)


# --------------------------------------------------------------------- config
def load_config(path=None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.is_file():
        sys.exit(f"config not found: {p}\nCopy pipeline.toml and fill in your buckets.")
    with open(p, "rb") as f:
        cfg = tomllib.load(f)
    for section in ("source", "crops", "transcripts", "crop", "api"):
        if section not in cfg:
            sys.exit(f"{p}: missing [{section}] section.")
    instr = cfg["api"].get("job", {}).get("transcription_instructions", "")
    if len(instr) > 500:
        sys.exit(f"{p}: transcription_instructions is {len(instr)} chars; the API caps at 500.")
    # Coords moved out of [crops] into their own bucket; fail loudly rather than
    # silently ignoring a stale key and writing no provenance at all.
    stale = {"write_coords", "coords_prefix"} & set(cfg["crops"])
    if stale:
        sys.exit(f"{p}: [crops] {', '.join(sorted(stale))} is obsolete — coords now go to "
                 f"their own bucket. Use:\n\n  [coords]\n  enabled = true\n  bucket  = ...\n")
    if cfg.get("coords", {}).get("enabled") and not cfg["coords"].get("bucket"):
        sys.exit(f"{p}: [coords] enabled = true but no bucket is set.")
    # An empty profile on a destination section means "same account as the crops
    # bucket". Resolved here, once, because an empty value otherwise falls through
    # to boto3's DEFAULT credential chain — which is the source account, and shows
    # up much later as an AccessDenied on the first write.
    for section in ("coords", "transcripts"):
        if section in cfg and not cfg[section].get("profile"):
            cfg[section]["profile"] = cfg["crops"]["profile"]
    return cfg


def _s3(profile):
    """A client for the named AWS profile (empty/None = default credential chain)."""
    return boto3.Session(profile_name=profile or None).client("s3")


# ----------------------------------------------------------------- key naming
def crop_key(source_key: str, label: str, prefix: str = "") -> str:
    """Mirror the source key, inserting the A/B label before the extension.

    Crops are re-encoded as JPEG, so the extension is always .jpg — for the flat
    `{volume}-{0000}.jpg` source bucket that means the key is byte-identical to
    the source except for the A/B suffix on split spreads.
    """
    root, _ext = posixpath.splitext(source_key)
    return f"{prefix}{root}{label}.jpg"


def coord_key(source_key: str, label: str, prefix: str = "") -> str:
    """Provenance key for a crop: the crop's key with the image extension replaced.

    e.g. source "176899-0002.jpg", label "A" -> "176899-0002A.json".
    """
    root, _ext = posixpath.splitext(source_key)
    return f"{prefix}{root}{label}.json"


def normalize_prefix(prefix: str, cfg: dict) -> str:
    """Drop a trailing key separator the caller may have typed.

    `prefix_suffix` is appended when listing, so `176899-` would otherwise list
    `176899--` and quietly match nothing. Normalising here keeps the listing, the
    resume check and the job title all working off the same string.
    """
    sep = cfg["source"].get("prefix_suffix", "")
    while sep and prefix.endswith(sep):
        prefix = prefix[:-len(sep)]
    return prefix


# ------------------------------------------------------------- memory budget
# Measured on this pipeline (folio.process build, CPU workers, synthetic spreads):
# a worker settles at ~1.2 GB resident once models are loaded and a few large
# images have passed through, plus a transient peak that scales with megapixels
# at ~42 MB/MP (680 MB at 16 MP, 1130 MB at 27 MP). One worker per core with
# 16 MP images therefore needs ~30 GB, which is how a 32 GB box dies.
WORKER_RESIDENT_MB = 1200.0
WORKER_MB_PER_MP = 42.0
MEMORY_SAFETY = 0.8            # leave headroom for the OS and the parent process


def worker_memory_mb(megapixels: float) -> float:
    """Estimated peak RSS of one crop worker on images of this size."""
    return WORKER_RESIDENT_MB + WORKER_MB_PER_MP * max(megapixels, 0.0)


def cap_jobs_for_memory(jobs: int, megapixels: float, available_mb=None) -> tuple:
    """Reduce `jobs` so the workers fit in RAM. Returns (jobs, note).

    Cropping is CPU-bound, so `jobs` is normally set to the core count — but the
    memory cost per worker scales with image size, and oversubscribing RAM does
    not merely slow things down, it takes the machine out.
    """
    if jobs <= 1:
        return jobs, None
    if available_mb is None:
        try:
            import psutil
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            return jobs, None          # cannot measure -> respect the config
    need = worker_memory_mb(megapixels)
    fits = max(1, int((available_mb * MEMORY_SAFETY) // need))
    if fits >= jobs:
        return jobs, None
    return fits, (f"jobs {jobs} -> {fits}: {megapixels:.1f} MP images need "
                  f"~{need/1024:.1f} GB/worker, {available_mb/1024:.1f} GB free")


def sample_megapixels(s3, bucket, key) -> float:
    """Decode one source image to learn the volume's working size."""
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return 0.0
        return img.shape[0] * img.shape[1] / 1e6
    except Exception:
        return 0.0


def _expected_crop_keys(source_key: str, prefix: str = "") -> set:
    """Every key this source image could plausibly have produced (1- or 2-folio)."""
    return {crop_key(source_key, lbl, prefix) for lbl in ("", "A", "B")}


def list_keys(s3, bucket, prefix, exts=None):
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if exts is None or k.lower().endswith(exts):
                out.append(k)
    return sorted(out)


# ------------------------------------------------------------ per-crop records
# One record per crop, carrying the pipeline's verdict on it. These drive both the
# blank-skipping at submission and the review report.
def _record(out_key, source_key, f) -> dict:
    return {
        "crop": out_key, "source_key": source_key, "label": f.label,
        "is_blank": bool(f.is_blank), "needs_review": bool(f.needs_review),
        "review_reasons": list(f.review_reasons or []),
        "blank_conf": round(float(f.blank_conf), 3),
        "orientation_conf": round(float(f.orientation_conf), 3),
        "text_frac": round(float(f.text_frac), 4),
        "rotation_deg": round(float(f.rotation_deg), 2),
        "ocr_rescued": bool(getattr(f, "ocr_rescued", False)),
    }


def _crop_metadata(f, source_key="") -> dict:
    """The full verdict, stored as S3 user metadata on the crop object.

    This is what a resumed run reads back, so it carries every field the review
    report prints — otherwise a resumed volume reports nulls for crops it did not
    reprocess. S3 caps user metadata at ~2 KB and requires ASCII, hence the
    truncation and the ascii coercion.
    """
    return {
        "folio-blank": "1" if f.is_blank else "0",
        "folio-review": "1" if f.needs_review else "0",
        "folio-reasons": ",".join(f.review_reasons or [])[:512],
        "folio-blank-conf": f"{float(f.blank_conf):.3f}",
        "folio-orient-conf": f"{float(f.orientation_conf):.3f}",
        "folio-source": str(source_key).encode("ascii", "ignore").decode()[:512],
        "folio-label": (f.label or ""),
        "folio-text-frac": f"{float(f.text_frac):.4f}",
        "folio-rotation": f"{float(f.rotation_deg):.2f}",
    }


def _record_from_metadata(out_key, md) -> dict:
    """Rebuild a record from a crop object's S3 user metadata (resume path)."""
    def _f(name):
        try:
            return float(md.get(name, "") or 0.0)
        except ValueError:
            return 0.0
    known = "folio-blank" in md
    reasons = [r for r in (md.get("folio-reasons", "") or "").split(",") if r]
    return {
        "crop": out_key,
        "source_key": md.get("folio-source") or None,
        "label": md.get("folio-label"),
        "is_blank": md.get("folio-blank") == "1",
        "needs_review": md.get("folio-review") == "1",
        "review_reasons": reasons,
        "blank_conf": _f("folio-blank-conf"),
        "orientation_conf": _f("folio-orient-conf"),
        "text_frac": _f("folio-text-frac") if "folio-text-frac" in md else None,
        "rotation_deg": _f("folio-rotation") if "folio-rotation" in md else None,
        "ocr_rescued": False,
        # Crops written before this metadata existed: treated as non-blank and
        # unflagged, but marked so the review report can say so out loud.
        "status_unknown": not known,
    }


def _records_for_existing(s3, bucket, keys, workers=16):
    """HEAD the already-present crops to recover their blank/review verdicts."""
    if not keys:
        return []
    from concurrent.futures import ThreadPoolExecutor

    def one(k):
        try:
            md = s3.head_object(Bucket=bucket, Key=k).get("Metadata", {}) or {}
        except Exception:
            md = {}
        return _record_from_metadata(k, md)

    with ThreadPoolExecutor(max_workers=min(workers, len(keys))) as ex:
        return list(ex.map(one, keys))


# ------------------------------------------------- per-worker state (spawned)
_W = {}


def _worker_init(cfg_kw, legacy, src_profile, tgt_profile, coords_profile=None):
    cfg = make_config(**cfg_kw)
    pipe, mode = build_pipeline(cfg, legacy)
    _W.update(pipe=pipe, mode=mode, src=_s3(src_profile), tgt=_s3(tgt_profile),
              coords=_s3(coords_profile) if coords_profile is not None else None)


def _crop_one(task):
    """Fetch one source image in memory, crop it, push the crops. Never raises."""
    key, src_bucket, tgt_bucket, tgt_prefix, coords_bucket, coords_prefix = task
    src, tgt, pipe = _W["src"], _W["tgt"], _W["pipe"]
    coords = _W.get("coords")
    try:
        body = src.get_object(Bucket=src_bucket, Key=key)["Body"].read()
        img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"key": key, "crops": [], "error": "unreadable"}
        res = pipe.process_image(posixpath.basename(key), img)
        written = []
        for f in res.folios:
            if f.crop is None:
                continue
            out_key = crop_key(key, f.label, tgt_prefix)
            ok, buf = cv2.imencode(".jpg", f.crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                continue
            # The blank/review verdict rides along as S3 user metadata so a resumed
            # run can recover it for crops it did not process this time (see
            # _records_for_existing) without re-running the pipeline.
            tgt.put_object(Bucket=tgt_bucket, Key=out_key,
                           Body=buf.tobytes(), ContentType="image/jpeg",
                           Metadata=_crop_metadata(f, key))
            if coords is not None:
                coord = {
                    "crop": out_key, "crops_bucket": tgt_bucket, "source_key": key,
                    "source_bucket": src_bucket,
                    "source_size": [int(img.shape[1]), int(img.shape[0])],
                    "crop_quad_norm": f.crop_quad_norm,
                    "corner_order": "TL,TR,BR,BL (x,y ratios of source image)",
                    "label": f.label, "page_count": res.page_count.value,
                    "rotation_deg": f.rotation_deg, "is_blank": f.is_blank,
                    "needs_review": f.needs_review,
                }
                # Mirror the crop key, minus the image extension, so a coord object
                # is trivially findable: "176899-0002A.json".
                coords.put_object(
                    Bucket=coords_bucket,
                    Key=coord_key(key, f.label, coords_prefix),
                    Body=json.dumps(coord).encode(), ContentType="application/json")
            written.append(_record(out_key, key, f))
        return {"key": key, "crops": written,
                "review": any(f.needs_review for f in res.folios), "error": res.error}
    except Exception as e:  # keep the run alive; the caller reports it
        return {"key": key, "crops": [], "error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------ stage 1
def crop_prefix(prefix, cfg, *, resume=True, limit=None, dry_run=False):
    """Crop every source image under `prefix`. Returns (crop_keys, stats)."""
    sc, cc, kc = cfg["source"], cfg["crops"], cfg["crop"]
    prefix = normalize_prefix(prefix, cfg)
    src = _s3(sc["profile"])
    s3_prefix = prefix + sc.get("prefix_suffix", "")
    keys = list_keys(src, sc["bucket"], s3_prefix, IMG_EXT)
    if limit:
        keys = keys[:limit]
    _log(f"{prefix}: {len(keys)} source image(s) under s3://{sc['bucket']}/{s3_prefix}")
    if not keys:
        return [], {"images": 0, "skipped": 0, "errors": 0, "review": 0}

    tgt_prefix = cc.get("prefix", "")
    tgt = _s3(cc["profile"])
    existing = set(list_keys(tgt, cc["bucket"], tgt_prefix + prefix, (".jpg",)))

    todo, done_keys, skipped = [], [], 0
    for k in keys:
        hit = _expected_crop_keys(k, tgt_prefix) & existing
        if resume and hit:
            done_keys.extend(sorted(hit))
            skipped += 1
        else:
            todo.append(k)
    if skipped:
        _log(f"{prefix}: {skipped} image(s) already cropped — resuming on {len(todo)}.")

    if dry_run:
        for k in todo[:10]:
            print("   ", k)
        _log(f"[dry-run] would crop {len(todo)} image(s) -> s3://{cc['bucket']}/{tgt_prefix}")
        return [], {"images": 0, "skipped": skipped, "errors": 0, "review": 0}

    coc = cfg.get("coords", {})
    coords_on = bool(coc.get("enabled"))
    coords_profile = coc.get("profile") if coords_on else None  # resolved in load_config
    coords_bucket = coc.get("bucket") if coords_on else None
    coords_prefix = coc.get("prefix", "") if coords_on else ""
    if coords_on:
        _log(f"{prefix}: coords -> s3://{coords_bucket}/{coords_prefix}")
    tasks = [(k, sc["bucket"], cc["bucket"], tgt_prefix, coords_bucket, coords_prefix)
             for k in todo]
    legacy = find_legacy_weights(None, str(_REPO))
    jobs = int(kc.get("jobs", 1) or 1)
    white_out = bool(kc.get("white_out", False))
    device = kc.get("device") or None
    rescue = bool(kc.get("rescue_partial_spread", False))
    if rescue:
        _log(f"{prefix}: partial-spread rescue ENABLED")

    # Size the pool to RAM, not just to cores: memory per worker scales with
    # megapixels, and oversubscribing it crashes the machine rather than swapping.
    if jobs > 1 and todo and kc.get("cap_jobs_to_memory", True):
        mp = sample_megapixels(src, sc["bucket"], todo[0])
        if mp > 0:
            jobs, note = cap_jobs_for_memory(jobs, mp)
            if note:
                _warn(f"{prefix}: {note}")
            else:
                _log(f"{prefix}: {mp:.1f} MP images, {jobs} worker(s) "
                     f"(~{worker_memory_mb(mp)/1024:.1f} GB each)")

    stats = {"images": 0, "skipped": skipped, "errors": 0, "review": 0}
    made = []

    def _tally(r, i, n):
        stats["images"] += 1
        stats["errors"] += bool(r.get("error"))
        stats["review"] += bool(r.get("review"))
        made.extend(r["crops"])
        if r.get("error"):
            _warn(f"{r['key']}: {r['error']}")
        if i % 50 == 0:
            _log(f"  {i}/{n}  crops={len(made)} errors={stats['errors']}")

    if tasks and jobs > 1:
        cfg_kw = dict(device=device or "cpu", mask_background=white_out,
                      crop_to_folio_mask=not white_out,
                      rescue_partial_spread=rescue)
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(jobs, initializer=_worker_init,
                      initargs=(cfg_kw, legacy, sc["profile"], cc["profile"],
                                coords_profile)) as pool:
            for i, r in enumerate(pool.imap_unordered(_crop_one, tasks), 1):
                _tally(r, i, len(tasks))
    elif tasks:
        _worker_init(dict(device=device, mask_background=white_out,
                          crop_to_folio_mask=not white_out,
                          rescue_partial_spread=rescue),
                     legacy, sc["profile"], cc["profile"], coords_profile)
        _log(f"config: {_W['mode']}")
        for i, t in enumerate(tasks, 1):
            _tally(_crop_one(t), i, len(tasks))

    # Crops carried over from an earlier run: recover their verdicts from S3
    # metadata so blank-skipping and the review report cover the whole volume.
    records = made + _records_for_existing(tgt, cc["bucket"], sorted(set(done_keys)))
    records.sort(key=lambda r: r["crop"])
    n_blank = sum(r["is_blank"] for r in records)
    stats["blank"] = n_blank
    stats["flagged"] = sum(r["needs_review"] for r in records)
    _log(f"{prefix}: {len(records)} crop(s) in s3://{cc['bucket']}/{tgt_prefix} "
         f"({stats['errors']} error(s), {stats['flagged']} flagged for review, "
         f"{n_blank} blank)")
    return records, stats


# ------------------------------------------------------------------ stage 2
def _build_metadata(cfg):
    """Mirror submit_job.main()'s metadata block, driven by pipeline.toml."""
    sj = _submit_job_module("build the job metadata")

    api = cfg["api"]
    job = api.get("job", {})
    models = api.get("models", {})
    prefs = api.get("transcription_preferences", {})
    return {
        "writing_style": job.get("writing_style", ""),
        "language": job.get("language", ""),
        "time_period": job.get("time_period", ""),
        "layout_structure": job.get("layout_structure", ""),
        "transcription_model": models.get("transcription", "gemini-3.6-flash"),
        "captioning_model": models.get("captioning", "gemini-3.5-flash-lite"),
        "foliation_model": models.get("foliation", "gemini-3.6-flash"),
        "aggregation_model": models.get("aggregation", "gemini-3.5-flash-lite"),
        "metadata_model": models.get("metadata", "gemini-3.6-flash"),
        "ner_model": models.get("ner", "gemini-3.6-flash"),
        "non_textual_elements": job.get("non_textual_elements", []),
        "transcription_preferences": {
            "expand_abbreviations": prefs.get("expand_abbreviations", False),
            "preserve_line_breaks": prefs.get("preserve_line_breaks", True),
            "retain_punctuation_and_spelling": prefs.get("retain_punctuation_and_spelling", True),
            "normalize_to_modern_language": prefs.get("normalize_to_modern_language", False),
            "ignore_marginalia": prefs.get("ignore_marginalia", False),
        },
        "metadata_schema": sj.DEFAULT_METADATA_SCHEMA,
        "additional_context_file": "",
        "additional_context_modules": [],
        "foliation_file": "",
        # submit_job.main() sets this when it auto-adds "foliate" for a metadata run.
        "foliation_override_discrete": ("metadata" in api.get("steps", [])
                                        and "foliate" not in api.get("steps", [])),
        "allow_subject_similarity": job.get("allow_subject_similarity", False),
        "delete_data": bool(api.get("delete_data", True)),
        "transcription_instructions": job.get("transcription_instructions", ""),
    }


def _submit_job_module(purpose: str):
    """Import SSDA's submit_job.py, or explain why it is needed.

    It is not vendored in this repo, so a fresh clone will not have it. Only the
    submission path needs it — cropping (--skip-submit / --dry-run) must keep
    working without it, which is why every import of it is lazy.
    """
    try:
        import submit_job as sj
    except ImportError:
        sys.exit(
            f"submit_job.py not found — needed to {purpose}.\n"
            "  It is SSDA's own script (slavesocieties/ssda-archivault), not vendored\n"
            f"  here; drop a copy at {_REPO / 'submit_job.py'}.\n"
            "  Cropping alone does not need it: try --skip-submit or --dry-run.")
    return sj


def _validate_models(cfg):
    sj = _submit_job_module("validate [api.models]")

    for module, chosen in cfg["api"].get("models", {}).items():
        allowed = sj.MODEL_OPTIONS.get(module)
        if allowed and chosen not in allowed:
            sys.exit(f"pipeline.toml: [api.models] {module} = {chosen!r} is not valid.\n"
                     f"  choose one of: {', '.join(allowed)}")


def submit_prefix(prefix, crop_keys, cfg, token, title):
    """Submit the crop keys via the API's S3 import flow. Returns (job_id, artifacts).

    `submit_job` is imported and called directly rather than shelled out to: the
    key list then has no OS argv length limit (a ~1k-image volume overflows
    Windows' ~32 KB cap), the password never reaches a process table, and a
    failure surfaces as an exception instead of an exit code.
    """
    sj = _submit_job_module("submit the job")

    api = cfg["api"]
    job = api.get("job", {})
    steps = set(api.get("steps", []))
    # submit_job.main() implies "foliate" whenever metadata is requested.
    if "metadata" in steps:
        steps.add("foliate")

    _log(f"{prefix}: submitting {len(crop_keys)} crop(s) as job {title!r}")
    return sj.submit_job(
        api_url=api["url"], token=token, directory=None, files_to_upload=[],
        title=title, steps=list(steps), country=job.get("country", ""),
        state=job.get("state", ""), description=job.get("description", ""),
        metadata=_build_metadata(cfg),
        source_bucket=cfg["crops"]["bucket"], keys=crop_keys,
    )


# ------------------------------------------------------------------ stage 3
def archive_artifacts(artifacts, cfg, title, manifest):
    """Stream each wanted artifact from its presigned URL into the transcripts bucket.

    Output keys are built from the *volume id* and the artifact kind alone —
    `<volume>.json` and `<volume>.md`. Whatever filename the API happens to use in
    its own `s3_key` is deliberately ignored, so a missing or surprising extension
    upstream cannot leak into the target bucket. Any other artifact kind (e.g. the
    tables ZIP) is discarded.
    """
    tc = cfg["transcripts"]
    s3 = _s3(tc["profile"])
    prefix = tc.get("prefix", "")
    stored = []

    for kind, info in (artifacts or {}).items():
        if kind not in ARTIFACT_EXT:
            _log(f"{title}: discarding {kind} artifact (not json/markdown)")
            continue
        if not isinstance(info, dict) or not info.get("presigned_url"):
            _warn(f"{title}: {kind} artifact has no presigned URL — skipped.")
            continue
        out_key = f"{prefix}{title}{ARTIFACT_EXT[kind]}"
        resp = requests.get(info["presigned_url"], timeout=120)
        if not resp.ok:
            _warn(f"{title}: could not fetch {kind} artifact ({resp.status_code})")
            continue
        s3.put_object(Bucket=tc["bucket"], Key=out_key, Body=resp.content,
                      ContentType=ARTIFACT_TYPE[kind])
        _log(f"{title}: {kind} -> s3://{tc['bucket']}/{out_key} ({len(resp.content)} bytes)")
        stored.append(out_key)

    for kind in ARTIFACT_EXT:
        if kind not in (artifacts or {}):
            _warn(f"{title}: the API returned no {kind} artifact.")

    if tc.get("write_manifest", True):
        manifest["artifacts"] = stored
        s3.put_object(Bucket=tc["bucket"], Key=f"{MANIFEST_DIR}{prefix}{title}.json",
                      Body=json.dumps(manifest, indent=2).encode(),
                      ContentType="application/json")
    return stored


def write_review_report(records, cfg, title, prefix_label):
    """Write the per-volume review report to the transcripts bucket.

    Written after cropping and *before* submission, so it exists even for a volume
    whose job later fails. Lists every crop held back as blank and every crop the
    pipeline flagged for review, with the reasons.
    """
    tc = cfg["transcripts"]
    prefix = tc.get("prefix", "")
    blanks = [r for r in records if r["is_blank"]]
    flagged = [r for r in records if r["needs_review"]]
    unknown = [r for r in records if r.get("status_unknown")]
    report = {
        "volume": title,
        "prefix": prefix_label,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "crops_total": len(records),
        "blank_skipped": len(blanks),
        "flagged_for_review": len(flagged),
        "submitted": len(records) - len(blanks),
        # Crops carried over from a run that predates the S3 verdict metadata:
        # counted as non-blank and unflagged, so say so rather than imply a clean bill.
        "status_unknown": len(unknown),
        "blank": [{k: r[k] for k in ("crop", "source_key", "blank_conf")} for r in blanks],
        "review": [{k: r[k] for k in ("crop", "source_key", "review_reasons",
                                      "orientation_conf", "blank_conf", "text_frac",
                                      "rotation_deg", "ocr_rescued")} for r in flagged],
    }
    key = f"{REVIEW_DIR}{prefix}{title}.json"
    _s3(tc["profile"]).put_object(Bucket=tc["bucket"], Key=key,
                                  Body=json.dumps(report, indent=2).encode(),
                                  ContentType="application/json")
    _log(f"{title}: review report -> s3://{tc['bucket']}/{key} "
         f"({len(flagged)} flagged, {len(blanks)} blank"
         + (f", {len(unknown)} unknown" if unknown else "") + ")")
    return key, report


# ------------------------------------------------------------------- preflight
def preflight(cfg, *, need_write=True):
    """Verify every bucket this run will touch is actually reachable, before the
    expensive cropping starts.

    Read access is not enough to predict write access — listing a bucket can
    succeed while PutObject is denied — so each destination is probed with a real
    put of a tiny object, which is then deleted.
    """
    sc, cc, tc = cfg["source"], cfg["crops"], cfg["transcripts"]
    coc = cfg.get("coords", {})
    checks = [("source", sc["profile"], sc["bucket"], False)]
    if need_write:
        checks.append(("crops", cc["profile"], cc["bucket"], True))
        if coc.get("enabled"):
            checks.append(("coords", coc["profile"], coc["bucket"], True))
        checks.append(("transcripts", tc["profile"], tc["bucket"], True))

    probe = f"_preflight-{uuid.uuid4().hex[:12]}.txt"
    bad = []
    for label, profile, bucket, writable in checks:
        try:
            s3 = _s3(profile)
            s3.list_objects_v2(Bucket=bucket, MaxKeys=1)
            if writable:
                s3.put_object(Bucket=bucket, Key=probe, Body=b"preflight")
                try:
                    s3.delete_object(Bucket=bucket, Key=probe)
                except Exception:  # the write is what matters; tidy-up is best effort
                    _warn(f"preflight: could not remove probe object {bucket}/{probe}")
            _log(f"preflight OK  {label:12} {'rw' if writable else 'r '}  s3://{bucket}"
                 f"  (profile {profile or 'default chain'})")
        except Exception as e:
            bad.append(f"  {label:12} s3://{bucket}  (profile {profile or 'default chain'})"
                       f"\n      {type(e).__name__}: {str(e).splitlines()[0][:140]}")
    if bad:
        sys.exit("preflight FAILED — fix access before running:\n" + "\n".join(bad))


# --------------------------------------------------------------------- driver
def run_prefix(prefix, cfg, *, token=None, resume=True, limit=None,
               dry_run=False, skip_crop=False, skip_submit=False):
    """Crop, submit and archive one source prefix. Returns a result dict."""
    started = time.time()
    # The job title IS the volume id: it names the API job and, downstream, the
    # artifact keys (<volume>.json / <volume>.md) in the transcripts bucket. A
    # trailing key separator is dropped so `176899` and `176899-` behave alike.
    api = cfg["api"]
    strip = api.get("title_strip", "")
    title = prefix[len(strip):] if strip and prefix.startswith(strip) else prefix
    sep = cfg["source"].get("prefix_suffix", "")
    if sep and title.endswith(sep):
        title = title[:-len(sep)]

    result = {"prefix": prefix, "title": title, "job_id": None,
              "started_utc": datetime.now(timezone.utc).isoformat()}

    if skip_crop:
        cc = cfg["crops"]
        s3c = _s3(cc["profile"])
        existing = list_keys(s3c, cc["bucket"], cc.get("prefix", "") + prefix, (".jpg",))
        _log(f"{prefix}: --skip-crop, found {len(existing)} existing crop(s)")
        records = _records_for_existing(s3c, cc["bucket"], existing)
        stats = {"images": 0, "skipped": len(existing), "errors": 0, "review": 0,
                 "blank": sum(r["is_blank"] for r in records),
                 "flagged": sum(r["needs_review"] for r in records)}
    else:
        records, stats = crop_prefix(prefix, cfg, resume=resume,
                                     limit=limit, dry_run=dry_run)
    result["crop_stats"] = stats
    result["crop_count"] = len(records)

    if dry_run:
        _log(f"[dry-run] would submit {len(records)} crop(s) as {title!r}; nothing spent.")
        return result
    if not records:
        _warn(f"{prefix}: no crops — skipping submission.")
        result["error"] = "no crops"
        return result

    # Written before submission so it survives a job failure.
    review_key, report = write_review_report(records, cfg, title, prefix)
    result.update(review_report=review_key, blank_skipped=report["blank_skipped"],
                  flagged_for_review=report["flagged_for_review"])

    # Blank folios are excluded from the job: the pipeline already decided there is
    # nothing on them, and transcribing them is pure spend.
    submit_keys = [r["crop"] for r in records if not r["is_blank"]]
    if report["blank_skipped"]:
        _log(f"{prefix}: holding back {report['blank_skipped']} blank crop(s); "
             f"submitting {len(submit_keys)}.")
    result["submitted_count"] = len(submit_keys)

    if skip_submit:
        _log(f"{prefix}: --skip-submit, stopping after cropping.")
        return result
    if not submit_keys:
        _warn(f"{prefix}: every crop is blank — nothing to submit.")
        result["error"] = "all crops blank"
        return result

    try:
        job_id, artifacts, up_s, inf_s = submit_prefix(prefix, submit_keys, cfg, token, title)
    except SystemExit as e:  # submit_job exits on API errors; keep the outer loop alive
        _warn(f"{prefix}: submission failed (submit_job exited {e.code}).")
        result["error"] = f"submit_job exit {e.code}"
        return result

    result.update(job_id=job_id, upload_seconds=round(up_s, 1),
                  inference_seconds=round(inf_s, 1),
                  elapsed_seconds=round(time.time() - started, 1),
                  source_bucket=cfg["source"]["bucket"],
                  crops_bucket=cfg["crops"]["bucket"])
    if not artifacts:
        _warn(f"{prefix}: job {job_id} returned no artifacts.")
        result["error"] = "no artifacts"
        return result

    archive_artifacts(artifacts, cfg, title, result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="+", help="source key prefix(es) — i.e. volume id(s)")
    ap.add_argument("--config", default=None, help=f"config file (default {DEFAULT_CONFIG.name})")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-crop images whose crops already exist in the crops bucket")
    ap.add_argument("--limit", type=int, default=None, help="crop at most N images (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="list the work, spend nothing")
    ap.add_argument("--skip-crop", action="store_true", help="submit crops already in the bucket")
    ap.add_argument("--skip-submit", action="store_true", help="crop only, do not submit")
    ap.add_argument("--no-preflight", action="store_true",
                    help="skip the up-front bucket read/write access check")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going if a prefix fails (default: stop, so a systemic "
                         "error cannot burn credits across dozens of volumes)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    # Only the submission path needs submit_job.py, so a crop-only run must not
    # be blocked by its absence.
    will_submit = not (args.dry_run or args.skip_submit)
    if will_submit:
        _validate_models(cfg)
    if not args.no_preflight:
        preflight(cfg, need_write=not args.dry_run)

    token = None
    if will_submit:
        sj = _submit_job_module("log in to the API")
        password = os.environ.get("ARCHIVAULT_PASSWORD") or getpass.getpass("Archivault password: ")
        token = sj.login(cfg["api"]["url"], cfg["api"]["email"], password)
        del password

    results, failed = [], 0
    for prefix in args.prefixes:
        print(f"\n=== {prefix} ===")
        r = run_prefix(prefix, cfg, token=token, resume=not args.no_resume,
                       limit=args.limit, dry_run=args.dry_run,
                       skip_crop=args.skip_crop, skip_submit=args.skip_submit)
        results.append(r)
        if r.get("error"):
            failed += 1
            if not args.continue_on_error:
                _warn(f"stopping after {prefix} ({r['error']}). "
                      "Re-run to resume, or pass --continue-on-error.")
                break

    print("\n--- summary ---")
    for r in results:
        status = r.get("error") or f"job {r.get('job_id')}"
        print(f"  {r['prefix']:<20} {r.get('crop_count', 0):>5} crop(s)  {status}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
