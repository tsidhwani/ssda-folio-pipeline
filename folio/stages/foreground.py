"""Classical, background-agnostic foreground/page detection.

This is the dependency-light stand-in for the RT-DETR detector used in the
classical-fallback mode (no GPU / no weights). It estimates the background
colour from the image border, builds a foreground (paper) mask that works on
dark, coloured, or light backgrounds, then proposes 1 or 2 page boxes by
testing for a central gutter valley.

Used only by folio.models.classical; the production path uses the neural
detector + SAM 2.1 instead. Pure NumPy + OpenCV, fully testable.
"""
from __future__ import annotations

from typing import List, Tuple
import cv2
import numpy as np

from ..schemas import PageBox


def _largest_components(mask: np.ndarray, min_area_frac: float, top_k: int = 4):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_frac * h * w:
            out.append((area, stats[i], cents[i]))
    out.sort(key=lambda t: -t[0])
    return out[:top_k]


def _channel_distance(lab: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Euclidean distance from `bg` per pixel, without full-frame temporaries.

    Equivalent to ``np.linalg.norm(lab - bg[None, None, :], axis=2)`` and verified
    bit-identical, but that expression materialises FOUR full-frame float32 arrays
    (the broadcast subtraction, the squaring inside norm, the reduction, plus
    `lab`). At 16 MP that is ~490 MB for one line; accumulating channel by channel
    into two reused buffers costs ~120 MB. With one worker per core that
    difference is measured in tens of gigabytes.
    """
    acc = np.zeros(lab.shape[:2], np.float32)
    tmp = np.empty(lab.shape[:2], np.float32)
    for i in range(lab.shape[2]):
        np.subtract(lab[:, :, i], bg[i], out=tmp)
        np.multiply(tmp, tmp, out=tmp)
        np.add(acc, tmp, out=acc)
    np.sqrt(acc, out=acc)
    return acc


def foreground_mask(image: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/1) of the paper region, robust to background colour.

    Strategy: estimate background from a border frame, threshold the per-pixel
    colour distance from it; if that fails (paper ~ background, e.g. white page
    on a white desk), fall back to an ink/edge-density mask.
    """
    h, w = image.shape[:2]
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # border frame samples -> background colour
    b = max(2, int(0.02 * min(h, w)))
    frame = np.concatenate([
        lab[:b, :].reshape(-1, 3), lab[-b:, :].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3),
    ], axis=0)
    bg = np.median(frame, axis=0)
    dist = _channel_distance(lab, bg)
    dist_n = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, fg = cv2.threshold(dist_n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = _clean(fg)
    cover = fg.mean() / 255.0
    if cover < 0.15 or cover > 0.97:
        # low contrast paper-vs-bg: fall back to ink/edge density
        fg2 = _ink_edge_mask(bgr)
        if 0.15 <= fg2.mean() / 255.0 <= 0.99:
            fg = fg2
    return (fg > 0).astype(np.uint8)


def _ink_edge_mask(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dens = cv2.dilate(edges, k)
    dens = cv2.morphologyEx(dens, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))
    return _clean(dens)


def _clean(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    k = max(3, int(0.01 * min(h, w)) | 1)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, el)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, el)
    # keep large blobs only
    out = np.zeros_like(m)
    for _, st, _ in _largest_components((m > 0).astype(np.uint8), 0.02, top_k=4):
        x, y, ww, hh = st[cv2.CC_STAT_LEFT], st[cv2.CC_STAT_TOP], \
            st[cv2.CC_STAT_WIDTH], st[cv2.CC_STAT_HEIGHT]
        out[y:y+hh, x:x+ww] |= m[y:y+hh, x:x+ww]
    return out


def _fg_col_mean(vals, mask):
    """Per-column mean over foreground pixels; NaN for columns that have none.

    Equivalent to masking the background to NaN and calling ``np.nanmean``, but
    without the "Mean of empty slice" RuntimeWarning that an all-background
    column raises. Such columns are expected (the page bbox is not the page), and
    callers already substitute a median for the NaNs. Done arithmetically rather
    than by filtering warnings, which would not be thread-safe.
    """
    cnt = mask.sum(axis=0)
    tot = np.where(mask, vals, 0.0).sum(axis=0)
    out = np.full(vals.shape[1], np.nan, np.float32)
    nz = cnt > 0
    out[nz] = tot[nz] / cnt[nz]
    return out


def _fg_median(vals, mask, default=0.0):
    """Median over foreground pixels; `default` when the mask is empty."""
    sel = vals[mask]
    return float(np.median(sel)) if sel.size else float(default)


def trim_facing_sliver(gray, fg, fx1, fy1, fx2, fy2,
                       band: float = 0.22, max_strip: float = 0.16,
                       prom_frac: float = 0.14):
    """If a single-page capture includes a thin strip of the FACING page along
    one outer edge (separated by a dark gutter), trim it off. Looks only in the
    outer `band` fraction and only removes a strip narrower than `max_strip` of
    the width. Returns a possibly-tightened (fx1, fx2).

    The defaults reproduce the original fixed thresholds; they are parameters so
    that tools/eval_partial_spread.py can sweep them against a real volume.
    """
    region = gray[fy1:fy2 + 1, fx1:fx2 + 1].astype(np.float32)
    fgreg = fg[fy1:fy2 + 1, fx1:fx2 + 1].astype(bool)
    bw = fx2 - fx1
    if bw < 20:
        return fx1, fx2
    col = _fg_col_mean(region, fgreg)
    finite = col[np.isfinite(col)]
    med = float(np.median(finite)) if finite.size else 0.0
    col = np.nan_to_num(col, nan=med)
    col_s = cv2.GaussianBlur(col.reshape(1, -1), (0, 0),
                             sigmaX=max(bw * 0.006, 1)).ravel()
    off = max(int(0.10 * bw), 4)
    med = float(np.median(col_s)) + 1e-6

    def deepest(band_lo, band_hi):
        idx = np.arange(band_lo, band_hi)
        if idx.size == 0:
            return None
        left = col_s[np.clip(idx - off, 0, bw - 1)]
        right = col_s[np.clip(idx + off, 0, bw - 1)]
        prom = np.minimum(left, right) - col_s[idx]
        b = int(np.argmax(prom))
        return band_lo + b, float(prom[b])

    new_l, new_r = fx1, fx2
    # left band [0.04, band]
    res = deepest(int(0.04 * bw), int(band * bw))
    if res and res[1] > prom_frac * med:
        new_l = fx1 + res[0]
    # right band [1-band, 0.96]
    res = deepest(int((1.0 - band) * bw), int(0.96 * bw))
    if res and res[1] > prom_frac * med:
        new_r = fx1 + res[0]
    # only accept trims that remove a NARROW strip (facing-page sliver)
    if (new_l - fx1) > max_strip * bw:
        new_l = fx1
    if (fx2 - new_r) > max_strip * bw:
        new_r = fx2
    return new_l, new_r


def detect_pages(image: np.ndarray, two_folio_valley: float = 0.6,
                 spine_band: Tuple[float, float] = (0.32, 0.68),
                 aspect_two: float = 1.15, aspect_two_valley: float = 1.05,
                 valley_shadow: float = 0.92,
                 trim_band: float = 0.22, trim_max_strip: float = 0.16,
                 ) -> Tuple[List[PageBox], np.ndarray, float]:
    """Return (page_boxes, foreground_mask, gutter_valley_ratio).

    Decides 1 vs 2 pages from the depth of a central vertical valley in the
    foreground column profile. Boxes are split at the valley; spine.detect_gutter
    later refines the exact (curved) seam.
    """
    h, w = image.shape[:2]
    fg = foreground_mask(image)
    comps = _largest_components(fg, min_area_frac=0.05, top_k=4)
    if not comps:
        return [], fg, 1.0

    # overall foreground bbox
    xs = np.where(fg.any(axis=0))[0]
    ys = np.where(fg.any(axis=1))[0]
    fx1, fx2 = int(xs.min()), int(xs.max())
    fy1, fy2 = int(ys.min()), int(ys.max())

    # A bound book photographed open is CONTINUOUS paper across the spread, so
    # the gutter is NOT a gap in the foreground - it is the SPINE SHADOW, a dark
    # vertical valley in intensity. Detect the spine via the darkest central
    # column within the page region (restricted to the central band).
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    region = gray[fy1:fy2+1, fx1:fx2+1].astype(np.float32)
    fgreg = fg[fy1:fy2+1, fx1:fx2+1].astype(bool)
    col = _fg_col_mean(region, fgreg)  # background columns come back as NaN
    col = np.nan_to_num(col, nan=_fg_median(region, fgreg))
    bw = fx2 - fx1
    col_s = cv2.GaussianBlur(col.reshape(1, -1), (0, 0),
                             sigmaX=max(bw * 0.008, 1)).ravel()
    # The spine is a LOCALIZED dark dip flanked by brighter pages - not simply
    # the globally darkest column (page edges / vignetting can be darker). Score
    # each candidate by its PROMINENCE: how much darker it is than the brighter
    # of its two flanks, sampled ~12% of the spread width to either side.
    off = max(int(0.12 * bw), 5)
    lo, hi = int(spine_band[0] * bw), int(spine_band[1] * bw)
    if hi > lo and col_s.size > 2 * off + 2:
        idx = np.arange(lo, hi)
        left_flank = col_s[np.clip(idx - off, 0, bw - 1)]
        right_flank = col_s[np.clip(idx + off, 0, bw - 1)]
        flank = np.minimum(left_flank, right_flank)   # brighter side must exceed dip
        prominence = flank - col_s[idx]               # large => sharp dark spine
        best = int(np.argmax(prominence))
        valley_x = fx1 + lo + best
        med = float(np.median(col_s)) + 1e-6
        shadow = float(col_s[lo + best] / med)
        # if no real dip stands out, treat as weak (low two-folio confidence)
        if prominence[best] < 0.04 * med:
            shadow = max(shadow, 0.95)
    else:
        shadow = 1.0
        valley_x = (fx1 + fx2) // 2

    aspect = (fx2 - fx1) / float(max(fy2 - fy1, 1))
    # Two folios when the spread is landscape; a shadow valley confirms it and
    # rescues borderline-aspect spreads. Single pages are portrait.
    is_two = (aspect > aspect_two) or (aspect > aspect_two_valley and shadow < valley_shadow)

    if is_two:
        conf = float(np.clip(1.0 - shadow + 0.5, 0.5, 0.99))
        left = PageBox(fx1, fy1, valley_x, fy2, conf)
        right = PageBox(valley_x, fy1, fx2, fy2, conf)
        return [left, right], fg, shadow
    tl, tr = trim_facing_sliver(gray, fg, fx1, fy1, fx2, fy2,
                                band=trim_band, max_strip=trim_max_strip)
    return [PageBox(tl, fy1, tr, fy2, 1.0)], fg, shadow
