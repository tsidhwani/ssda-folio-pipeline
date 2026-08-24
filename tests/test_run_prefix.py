"""Tests for scripts/run_prefix.py — the crop -> submit -> archive orchestrator.

Everything here runs against an in-memory fake S3 and a fake submit_job module, so
no credentials, no network and no weights are needed. What is covered is the part
that is expensive to get wrong in production:

  * crop keys mirror the flat source bucket, with A/B only on split spreads,
  * resume skips already-cropped images but still submits the whole volume,
  * blank folios are held back from the API job,
  * artifact keys depend only on the volume id, never on what the API returns,
  * coords go to their own bucket,
  * an empty destination profile means "the crops profile", not the default chain.
"""
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

run_prefix = pytest.importorskip("run_prefix")
rp = run_prefix

JPEG = cv2.imencode(".jpg", np.full((60, 40, 3), 200, np.uint8))[1].tobytes()


# --------------------------------------------------------------------- fakes
class FakeS3:
    """Minimal S3: get/put/head/delete/list over a dict of dicts."""

    def __init__(self, buckets, meta):
        self.b, self.meta = buckets, meta

    def get_object(self, Bucket, Key):
        return {"Body": types.SimpleNamespace(read=lambda: self.b[Bucket][Key])}

    def put_object(self, Bucket, Key, Body, **kw):
        self.b.setdefault(Bucket, {})[Key] = Body
        if kw.get("Metadata"):
            self.meta[(Bucket, Key)] = kw["Metadata"]

    def head_object(self, Bucket, Key):
        if Key not in self.b.get(Bucket, {}):
            raise KeyError(Key)
        return {"Metadata": self.meta.get((Bucket, Key), {})}

    def delete_object(self, Bucket, Key):
        self.b.get(Bucket, {}).pop(Key, None)
        self.meta.pop((Bucket, Key), None)

    def list_objects_v2(self, Bucket, **kw):
        keys = sorted(self.b.get(Bucket, {}))
        return {"KeyCount": len(keys), "Contents": [{"Key": k} for k in keys]}

    def get_paginator(self, _name):
        outer = self

        class P:
            def paginate(self, Bucket, Prefix="", **kw):
                yield {"Contents": [{"Key": k} for k in sorted(outer.b.get(Bucket, {}))
                                    if k.startswith(Prefix)]}
        return P()


class FakeFolio:
    def __init__(self, label="", blank=False, review=False, reasons=()):
        self.label = label
        self.crop = np.zeros((40, 30, 3), np.uint8)
        self.crop_quad_norm = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.rotation_deg, self.is_blank, self.needs_review = 90.0, blank, review
        self.review_reasons = list(reasons)
        self.blank_conf = 0.97 if blank else 0.02
        self.orientation_conf = 0.41 if review else 0.99
        self.text_frac = 0.0 if blank else 0.13
        self.ocr_rescued = False


class FakeRes:
    page_count = types.SimpleNamespace(value=1)

    def __init__(self, folios):
        self.folios, self.error = folios, None


class FakePipe:
    """0002 is a two-folio spread; 0003 is blank; everything else is one folio."""

    def process_image(self, name, img):
        if "0002" in name:
            return FakeRes([FakeFolio("A"),
                            FakeFolio("B", review=True, reasons=["low_orientation_conf"])])
        if "0003" in name:
            return FakeRes([FakeFolio("", blank=True)])
        return FakeRes([FakeFolio("")])


@pytest.fixture
def env(monkeypatch):
    """Wire run_prefix to fakes and hand back the buckets plus a loaded config."""
    buckets = {"src": {}, "crops": {}, "coords": {}, "trans": {}}
    meta = {}
    submitted = {}
    profiles = []

    def fake_s3(profile):
        profiles.append(profile)
        return FakeS3(buckets, meta)

    monkeypatch.setattr(rp, "_s3", fake_s3)
    monkeypatch.setattr(rp, "find_legacy_weights", lambda *a, **k: None)

    def fake_worker_init(cfg_kw, legacy, sp, tp, cp=None):
        rp._W.update(pipe=FakePipe(), mode="fake", src=FakeS3(buckets, meta),
                     tgt=FakeS3(buckets, meta),
                     coords=FakeS3(buckets, meta) if cp is not None else None)
    monkeypatch.setattr(rp, "_worker_init", fake_worker_init)

    sj = types.ModuleType("submit_job")
    sj.DEFAULT_METADATA_SCHEMA = {"title": "x"}
    sj.MODEL_OPTIONS = {}
    sj.login = lambda *a: "TOKEN"

    def fake_submit_job(**kw):
        submitted.update(kw)
        return ("job-123",
                {"json": {"presigned_url": "http://x/1", "s3_key": "jobs/9f8e7d"},
                 "markdown": {"presigned_url": "http://x/2", "s3_key": "jobs/out.txt"},
                 "tables_zip": {"presigned_url": "http://x/3", "s3_key": "t.zip"}},
                1.0, 2.0)
    sj.submit_job = fake_submit_job
    monkeypatch.setitem(sys.modules, "submit_job", sj)
    monkeypatch.setattr(rp, "requests", types.SimpleNamespace(
        get=lambda url, **kw: types.SimpleNamespace(
            ok=True, status_code=200, content=b'{"transcript":"hi"}')))

    for i in (1, 2, 3):
        buckets["src"][f"176899-{i:04d}.jpg"] = JPEG
    buckets["src"]["1768-0001.jpg"] = JPEG        # must NOT match prefix 176899

    cfg = rp.load_config(_REPO / "pipeline.example.toml")
    cfg["source"].update(bucket="src", profile="p-src")
    cfg["crops"].update(bucket="crops", profile="p-dst")
    cfg["transcripts"].update(bucket="trans", profile="p-dst")
    cfg["coords"].update(enabled=False, bucket="coords", profile="p-dst")
    cfg["crop"]["jobs"] = 1
    return types.SimpleNamespace(buckets=buckets, meta=meta, cfg=cfg,
                                 submitted=submitted, profiles=profiles)


# ----------------------------------------------------------------- key naming
@pytest.mark.parametrize("src,expected", [
    ("176899-0001.jpg", ["176899-0001.jpg", "176899-0001A.jpg", "176899-0001B.jpg"]),
    ("GMRV_005090007-0142.jpg", ["GMRV_005090007-0142.jpg", "GMRV_005090007-0142A.jpg",
                                 "GMRV_005090007-0142B.jpg"]),
    ("176899-0001.tif", ["176899-0001.jpg", "176899-0001A.jpg", "176899-0001B.jpg"]),
])
def test_crop_key_mirrors_the_source_key(src, expected):
    assert [rp.crop_key(src, lbl) for lbl in ("", "A", "B")] == expected


def test_coord_key_drops_the_image_extension():
    assert rp.coord_key("176899-0002.jpg", "A") == "176899-0002A.json"
    assert rp.coord_key("176899-0001.jpg", "") == "176899-0001.json"


def test_crop_key_honours_a_prefix():
    assert rp.crop_key("176899-0001.jpg", "A", "folios/") == "folios/176899-0001A.jpg"


# ------------------------------------------------------------------ cropping
def test_crop_writes_mirrored_keys_and_ignores_other_volumes(env):
    records, stats = rp.crop_prefix("176899", env.cfg, resume=False)
    assert sorted(env.buckets["crops"]) == [
        "176899-0001.jpg", "176899-0002A.jpg", "176899-0002B.jpg", "176899-0003.jpg"]
    # prefix_suffix "-" must stop 176899 from sweeping up 1768-0001.jpg
    assert not any(k.startswith("1768-") for k in env.buckets["crops"])
    assert stats["blank"] == 1 and stats["flagged"] == 1


def test_resume_skips_cropped_images_but_still_submits_the_volume(env):
    rp.run_prefix("176899", env.cfg, token="T")
    env.submitted.clear()
    r = rp.run_prefix("176899", env.cfg, token="T")
    assert r["crop_stats"]["images"] == 0
    assert r["crop_stats"]["skipped"] == 3
    assert r["crop_count"] == 4                     # whole volume, not just new work
    assert len(env.submitted["keys"]) == 3          # minus the blank


def test_resume_recovers_verdicts_from_s3_metadata(env):
    rp.run_prefix("176899", env.cfg, token="T")
    rp.run_prefix("176899", env.cfg, token="T")
    rep = json.loads(env.buckets["trans"]["review/176899.json"])
    assert rep["blank_skipped"] == 1 and rep["flagged_for_review"] == 1
    assert rep["status_unknown"] == 0
    assert rep["review"][0]["review_reasons"] == ["low_orientation_conf"]
    assert rep["review"][0]["source_key"] == "176899-0002.jpg"


def test_crops_without_metadata_are_reported_unknown_not_clean(env):
    rp.run_prefix("176899", env.cfg, token="T")
    env.meta.clear()                                 # simulate pre-metadata crops
    rp.run_prefix("176899", env.cfg, token="T")
    rep = json.loads(env.buckets["trans"]["review/176899.json"])
    assert rep["status_unknown"] == 4 and rep["blank_skipped"] == 0


# ------------------------------------------------------------------- coords
def test_coords_go_to_their_own_bucket(env):
    env.cfg["coords"].update(enabled=True)
    rp.crop_prefix("176899", env.cfg, resume=False)
    assert sorted(env.buckets["coords"]) == [
        "176899-0001.json", "176899-0002A.json", "176899-0002B.json", "176899-0003.json"]
    c = json.loads(env.buckets["coords"]["176899-0002A.json"])
    assert c["crop"] == "176899-0002A.jpg" and c["source_key"] == "176899-0002.jpg"


def test_coords_disabled_leaves_the_bucket_untouched(env):
    rp.crop_prefix("176899", env.cfg, resume=False)
    assert env.buckets["coords"] == {}
    assert len(env.buckets["crops"]) == 4


# ------------------------------------------------------- blanks + submission
def test_blank_crops_are_written_but_not_submitted(env):
    rp.run_prefix("176899", env.cfg, token="T")
    assert "176899-0003.jpg" in env.buckets["crops"]      # still cropped
    assert "176899-0003.jpg" not in env.submitted["keys"]  # but not transcribed
    assert env.submitted["keys"] == ["176899-0001.jpg", "176899-0002A.jpg",
                                     "176899-0002B.jpg"]


def test_submission_imports_from_the_crops_bucket(env):
    rp.run_prefix("176899", env.cfg, token="T")
    assert env.submitted["source_bucket"] == "crops"
    assert env.submitted["title"] == "176899"


def test_dry_run_submits_nothing(env):
    r = rp.run_prefix("176899", env.cfg, token="T", dry_run=True)
    assert not env.submitted and r["job_id"] is None


# -------------------------------------------------------------- artifacts
def test_artifact_keys_ignore_whatever_the_api_names_them(env):
    """s3_key values above are junk ("jobs/9f8e7d", "out.txt") on purpose."""
    rp.run_prefix("176899", env.cfg, token="T")
    assert "176899.json" in env.buckets["trans"]
    assert "176899.md" in env.buckets["trans"]
    assert not any(k.endswith(".zip") for k in env.buckets["trans"])


def test_manifest_and_review_use_directory_prefixes(env):
    rp.run_prefix("176899", env.cfg, token="T")
    assert "manifest/176899.json" in env.buckets["trans"]
    assert "review/176899.json" in env.buckets["trans"]
    man = json.loads(env.buckets["trans"]["manifest/176899.json"])
    assert man["job_id"] == "job-123" and man["crop_count"] == 4
    assert man["artifacts"] == ["176899.json", "176899.md"]


def test_review_report_is_written_before_submission_survives_no_artifacts(env, monkeypatch):
    import submit_job as sj
    monkeypatch.setattr(sj, "submit_job", lambda **kw: ("job-9", {}, 1.0, 1.0))
    r = rp.run_prefix("176899", env.cfg, token="T")
    assert r["error"] == "no artifacts"
    assert "review/176899.json" in env.buckets["trans"]   # written anyway


def test_title_strips_prefix_and_trailing_separator(env):
    env.cfg["api"]["title_strip"] = "GMRV_"
    env.buckets["src"].clear()
    env.buckets["src"]["GMRV_00509-0001.jpg"] = JPEG
    r = rp.run_prefix("GMRV_00509-", env.cfg, token="T")
    assert r["title"] == "00509"
    assert "00509.json" in env.buckets["trans"]


# ----------------------------------------------------------------- config
def test_empty_destination_profile_falls_back_to_the_crops_profile(tmp_path):
    """An empty profile must NOT mean boto3's default chain (a different account)."""
    src = (_REPO / "pipeline.example.toml").read_text(encoding="utf-8")
    p = tmp_path / "pipeline.toml"
    p.write_text(src.replace('profile = "CHANGEME-write-profile"',
                             'profile = "the-write-profile"'), encoding="utf-8")
    cfg = rp.load_config(p)
    assert cfg["transcripts"]["profile"] == "the-write-profile"
    assert cfg["coords"]["profile"] == "the-write-profile"
    assert cfg["source"]["profile"] == "default"      # source is left alone


def test_stale_write_coords_key_is_rejected(tmp_path):
    src = (_REPO / "pipeline.example.toml").read_text(encoding="utf-8")
    p = tmp_path / "pipeline.toml"
    p.write_text(src.replace("[coords]", "write_coords = true\n\n[coords]"),
                 encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        rp.load_config(p)
    assert "obsolete" in str(e.value)


def test_example_config_is_loadable():
    """The tracked template must stay valid — it is what a new clone starts from."""
    cfg = rp.load_config(_REPO / "pipeline.example.toml")
    for section in ("source", "crops", "coords", "transcripts", "crop", "api"):
        assert section in cfg
    assert cfg["crop"]["rescue_partial_spread"] is False


# ------------------------------------------------------------ memory budget
def test_worker_memory_scales_with_megapixels():
    small, large = rp.worker_memory_mb(3.0), rp.worker_memory_mb(27.0)
    assert large > small
    # anchored to measurements: ~1.9 GB at 16 MP, ~2.3 GB at 27 MP
    assert 1700 < rp.worker_memory_mb(16.1) < 2100
    assert 2200 < large < 2500


def test_jobs_are_capped_when_ram_is_short():
    """The crash case: 16 workers x 16 MP images on a 32 GB box."""
    jobs, note = rp.cap_jobs_for_memory(16, 16.1, available_mb=23 * 1024)
    assert jobs < 16 and note and "16 ->" in note
    assert jobs * rp.worker_memory_mb(16.1) <= 23 * 1024


def test_jobs_are_untouched_when_ram_is_ample():
    jobs, note = rp.cap_jobs_for_memory(8, 3.0, available_mb=256 * 1024)
    assert jobs == 8 and note is None


def test_small_images_still_get_full_parallelism():
    jobs, _n = rp.cap_jobs_for_memory(16, 3.4, available_mb=23 * 1024)
    assert jobs >= 12, "small images should not be throttled on a 32 GB box"


def test_cap_never_returns_zero_workers():
    jobs, _n = rp.cap_jobs_for_memory(16, 200.0, available_mb=512)
    assert jobs == 1


def test_single_job_is_never_capped():
    assert rp.cap_jobs_for_memory(1, 99.0, available_mb=64)[0] == 1
