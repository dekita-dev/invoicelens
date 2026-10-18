"""Document image preprocessing for the OCR path.

Why this exists: a photo of a paper invoice is not a scan. It has
perspective skew (the camera was not parallel to the paper), uneven light
(shadow of the hand that held the phone), and often a busy background. OCR
engines were trained on flat, axis-aligned, evenly lit pages, so the raw
photo under-reads. Three cheap deterministic passes fix most of that.

Design constraints, in order of importance:

1. **Never make things worse.** Every pass is individually fallible: no
   quadrilateral found, degenerate transform, unwritable temp file, cv2
   missing. Every one of those returns the ORIGINAL image unchanged. A
   slightly imperfect result is acceptable; a destroyed input is not.
2. **cv2 is optional.** The module imports it lazily. Not installed (or
   import fails) means the whole file degrades to a passthrough, and no
   other part of the product notices.
3. **Deterministic.** Same input bytes, same output bytes. No RNG, no
   model, no network, no iterative refinement over scales.
4. **Bounded work.** One load, one edge pass, one warp, one division, a
   size cap on the output. A phone photo is at most a few seconds here.

Passes:
  quad      find the page outline (largest 4-gon covering 20-95% of frame)
  warp      perspective-correct it onto an axis-aligned rectangle
  light     flat-field division: estimate the background with a heavy
            blur and divide it out, killing shadow and vignetting

The frame-coverage gate is measured, not cautious: on the synthetic set
(2026-10-18, rapidocr, 32 fixtures) the pass GAINED 12 points of
total-usable on the degraded phone tier and COST 12 points on pristine
full-frame pages, where there is no desk to crop and no shadow to flatten.
So the pass runs exactly when it was built to run: a page sitting inside a
larger frame. A page that already fills the frame is handed over untouched.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

# Long-side cap for the warped output. A 12 MP phone photo does not need to
# stay 12 MP for OCR, and an unbounded warp is slow and memory-hungry.
MAX_SIDE = 2400


@dataclass
class PreprocessResult:
    """What the passes did. *changed* is False when the original is best."""

    image: object = None            # the array to hand to the OCR engine
    changed: bool = False           # False = use the original, nothing applied
    quad_found: bool = False
    warped: bool = False
    illumination: bool = False
    notes: str = ""


# A page outline covering more than this share of the frame is the frame
# itself: a full-page scan or an edge-to-edge photo. Nothing to crop, and
# the light pass measured as a net loss on exactly those inputs.
FULL_FRAME_RATIO = 0.95


def _import_cv():
    """Return (cv2, np) or (None, None). Never raises."""
    try:
        import cv2
        import numpy as np
        return cv2, np
    except Exception:
        return None, None


def _order_corners(pts, np):
    """Order 4 points as tl, tr, br, bl (the order getPerspectiveTransform wants)."""
    s = pts[np.argsort(pts[:, 1])]
    top = s[:2]
    bottom = s[2:]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")


def _find_quad(img, cv2, np):
    """Largest convex 4-gon covering at least 20% of the frame, or None.

    The epsilon sweep is not superstition: on the blurred + JPEG'd fixtures
    the page ring approximates to five points at eps=0.02 (one wobble from
    the compression) and a clean four at eps=0.03. Trying a few and taking
    the first clean quad finds pages a single epsilon loses.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Close small gaps in the page border so approxPolyDP sees one outline.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    frame_area = float(h * w)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(c) < 0.20 * frame_area:
            break
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return _order_corners(approx.reshape(4, 2).astype("float32"), np)
    return None


def _warp(img, quad, cv2, np):
    """Perspective-correct onto an axis-aligned rectangle, or None."""
    tl, tr, br, bl = (np.asarray(p, dtype="float32") for p in quad)
    width = max(float(np.linalg.norm(br - bl)), float(np.linalg.norm(tr - tl)))
    height = max(float(np.linalg.norm(tr - br)), float(np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        return None
    scale = min(1.0, MAX_SIDE / max(width, height))
    dst_w, dst_h = int(round(width * scale)), int(round(height * scale))
    if dst_w < 50 or dst_h < 50:
        return None
    dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1],
                    [0, dst_h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img, matrix, (dst_w, dst_h))
    if warped is None or warped.size == 0:
        return None
    return warped


def _normalize_illumination(img, cv2, np):
    """Flat-field division: divide out a heavy-blur background estimate.

    A shadow is a slow, large-scale brightness change. Blurring hard enough
    keeps the text (fine detail) out of the estimate, so dividing by it
    removes the shadow while leaving the ink. Black areas would blow up on
    division, so the result is clipped and the divisor is floored.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    long_side = max(gray.shape[:2])
    k = max(15, (long_side // 12) | 1)  # force odd
    background = cv2.GaussianBlur(gray, (k, k), 0).astype("float32")
    out = img.astype("float32")
    for ch in range(min(3, img.shape[2])):
        out[:, :, ch] = out[:, :, ch] / (background + 1.0) * 128.0
    return np.clip(out, 0, 255).astype("uint8")


def preprocess_array(img):
    """Run the passes on a BGR array. Returns a PreprocessResult.

    The whole pass is gated on one measured fact: it only fires when a page
    quad sits INSIDE the frame (20-95% coverage). Full-frame pages come
    back untouched, and so does anything without a detectable page.

    On any unexpected failure the input array comes back unchanged with
    changed=False; callers can trust the image field either way.
    """
    cv2, np = _import_cv()
    if cv2 is None or img is None or getattr(img, "size", 0) == 0:
        return PreprocessResult(image=img, changed=False,
                               notes="cv2 missing or empty image")

    result = PreprocessResult(image=img)
    try:
        quad = _find_quad(img, cv2, np)
        if quad is None:
            return PreprocessResult(image=img, changed=False,
                                    notes="no page outline inside the frame")
        result.quad_found = True
        frame_area = float(img.shape[0] * img.shape[1])
        quad_area = float(cv2.contourArea(quad.astype("float32")))
        if quad_area >= FULL_FRAME_RATIO * frame_area:
            return PreprocessResult(image=img, changed=False, quad_found=True,
                                    notes="page fills the frame; nothing to "
                                          "correct")
        warped = _warp(img, quad, cv2, np)
        if warped is None:
            return PreprocessResult(image=img, changed=False, quad_found=True,
                                    notes="degenerate warp")
        result.image = warped
        result.warped = True
        lit = _normalize_illumination(result.image, cv2, np)
        if lit is not None and lit.shape == result.image.shape:
            result.image = lit
            result.illumination = True
        result.changed = result.warped and result.illumination
    except Exception as exc:  # a pass must never take the caller down
        result = PreprocessResult(image=img, changed=False,
                                  notes="pass failed: %s" % exc)
    return result


def preprocess_image(path, out_dir=None):
    """Return a path to OCR: a preprocessed temp PNG, or *path* unchanged.

    The return value is the original path whenever nothing could be applied
    (cv2 missing, unreadable file, every pass fell back, write failed).
    Callers can compare the result to *path* to know whether they own a
    temp file they should delete.
    """
    cv2, np = _import_cv()
    if cv2 is None:
        return path
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return path
        result = preprocess_array(img)
        if not result.changed:
            return path
        target_dir = out_dir or tempfile.gettempdir()
        fd, tmp = tempfile.mkstemp(prefix="invoicelens_prep_",
                                   suffix=".png", dir=target_dir)
        os.close(fd)
        if not cv2.imwrite(tmp, result.image):
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return path
        return tmp
    except Exception:
        return path
