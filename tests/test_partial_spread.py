"""Tests for the partial-spread rescue (folio.stages.partial_spread)."""
import numpy as np
import pytest

from folio.config import PipelineConfig
from folio.stages.partial_spread import (find_spine, rescue_partial_spread,
                                         spine_score, split_quad)


def _page(w, h, ink_rows=28, seed=0):
    """A synthetic page: light paper with dark, ragged horizontal text lines.

    The ragged line ends matter — perfectly uniform lines give every column an
    identical darkness profile, which is nothing like a real page of handwriting.
    """
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 232, np.uint8)
    for y in range(20, h - 20, max(h // ink_rows, 4)):
        x0 = int(rng.uniform(0.06, 0.12) * w)
        x1 = int(rng.uniform(0.80, 0.94) * w)
        img[y:y + max(h // 200, 2), x0:x1] = 40
        # break each line into words so columns differ from one another
        for _ in range(rng.integers(3, 7)):
            g = int(rng.uniform(x0, x1))
            img[y:y + max(h // 200, 2), g:g + max(w // 90, 2)] = 232
    return img


def _spread(full_w, partial_w, h, spine=10):
    """A complete folio, a dark spine column, then a partial facing leaf."""
    left = _page(full_w, h, seed=1)
    right = _page(partial_w, h, seed=2)
    gutter = np.full((h, spine, 3), 45, np.uint8)
    return np.hstack([left, gutter, right])


def test_spine_score_peaks_at_the_spine():
    img = _spread(600, 260, 900)
    gray = img[:, :, 0]
    s = spine_score(gray)
    x = int(np.argmax(s))
    assert 595 <= x <= 615, f"spine found at {x}, expected ~600-610"


def test_find_spine_returns_ratio_above_page_baseline():
    img = _spread(600, 260, 900)
    x, ratio = find_spine(img[:, :, 0])
    assert x is not None and 590 <= x <= 620
    assert ratio > 1.5, f"spine ratio {ratio} should clear the default gate"


def test_rescue_keeps_the_complete_folio():
    img = _spread(600, 260, 900)          # 870 wide / 900 high -> aspect ~0.97
    kept, quad, info = rescue_partial_spread(img, None)
    assert info["fired"], info
    assert info["kept"] == "left"
    # the complete folio is ~600 wide; the partial leaf must be gone
    assert 560 <= kept.shape[1] <= 640, kept.shape
    assert 0.6 <= info["aspect_after"] <= 0.75


def test_rescue_keeps_right_when_the_partial_leaf_is_on_the_left():
    img = _spread(600, 260, 900)[:, ::-1]  # mirror: partial leaf now on the left
    kept, quad, info = rescue_partial_spread(img, None)
    assert info["fired"] and info["kept"] == "right", info
    assert 560 <= kept.shape[1] <= 640, kept.shape


def test_rescue_declines_when_aspect_already_in_range():
    img = _page(600, 900)                  # 0.67 aspect, nothing to fix
    kept, quad, info = rescue_partial_spread(img, None)
    assert kept is None and not info["fired"]
    assert "already in range" in info["why"]


def test_rescue_declines_on_a_blank_cover():
    img = np.full((900, 900, 3), 225, np.uint8)   # wide, but no text at all
    kept, quad, info = rescue_partial_spread(img, None)
    assert kept is None and not info["fired"]
    assert "too little text" in info["why"]


def test_rescue_declines_when_no_confident_spine():
    """An over-wide page with no spine must not have one invented for it.

    Uses flat texture rather than ruled text: the detector scores a column by how
    much darker it is than its horizontal neighbourhood, so the ragged edge of a
    text block can score spine-like. A two-sided test rejects that but performs
    measurably worse on real pages (see spine_score), so that trade is taken
    deliberately and this test does not assert against it.
    """
    rng = np.random.default_rng(3)
    img = (np.full((900, 880, 3), 228, np.uint8)
           + rng.integers(-6, 7, (900, 880, 3))).astype(np.uint8)
    kept, quad, info = rescue_partial_spread(img, None)
    assert kept is None and not info["fired"]


def test_min_spine_gate_is_respected():
    img = _spread(600, 260, 900)
    _k, _q, ok = rescue_partial_spread(img, None, min_spine=1.2)
    assert ok["fired"]
    _k2, _q2, strict = rescue_partial_spread(img, None, min_spine=99.0)
    assert not strict["fired"] and "no spine-like column" in strict["why"]


# ---- provenance ------------------------------------------------------------
QUAD = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]   # TL, TR, BR, BL


def test_split_quad_left_half():
    q = split_quad(QUAD, 0.5, keep_left=True)
    assert q == [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]


def test_split_quad_right_half():
    q = split_quad(QUAD, 0.5, keep_left=False)
    assert q == [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]


def test_split_quad_follows_a_sheared_quad():
    """On a non-rectangular quad the new corners ride the top and bottom edges."""
    quad = [[10.0, 0.0], [110.0, 10.0], [100.0, 210.0], [0.0, 200.0]]
    q = split_quad(quad, 0.25, keep_left=True)
    assert q[1] == pytest.approx([35.0, 2.5])     # along TL->TR
    assert q[2] == pytest.approx([25.0, 202.5])   # along BL->BR
    assert q[0] == quad[0] and q[3] == quad[3]


def test_rescue_narrows_the_quad_with_the_crop():
    img = _spread(600, 260, 900)
    kept, quad, info = rescue_partial_spread(img, QUAD)
    assert info["fired"] and info["kept"] == "left"
    # kept the left side, so the quad must shrink from the right, not the left
    assert quad[0] == [0.0, 0.0] and quad[3] == [0.0, 1.0]
    assert 0.6 < quad[1][0] < 0.78, quad
    assert quad[1][0] == pytest.approx(quad[2][0])


def test_rescue_is_off_by_default():
    """It must not change production behaviour until deliberately enabled."""
    assert PipelineConfig().quality.rescue_partial_spread is False


def test_channel_distance_matches_linalg_norm_exactly():
    """The in-place distance must be identical to the expression it replaced."""
    from folio.stages.foreground import _channel_distance
    rng = np.random.default_rng(0)
    lab = (rng.random((120, 90, 3)) * 255).astype(np.float32)
    bg = np.array([31.0, 128.0, 127.0], np.float32)
    ref = np.linalg.norm(lab - bg[None, None, :], axis=2)
    assert np.array_equal(_channel_distance(lab, bg), ref)
