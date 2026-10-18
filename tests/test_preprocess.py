"""Tests for the OCR preprocessing pass.

Every case here is synthetic but built to fail the way a real phone photo
fails: perspective skew, shadow gradient, no page border at all. The
property under test is the one the module docstring promises: the pass
either demonstrably helps, or hands the original back untouched.

Runs wherever cv2 is installed; skips cleanly where it is not (the product
treats cv2 as optional, so the test suite must too).
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_CV2 = False

from invoicelens import preprocess  # noqa: E402


def _make_photo(angle_deg=8.0, gradient=False, page=True, seed=7,
                paper_shadow=False):
    """A stand-in for a phone photo: dark desk, white page, black text.

    The page is rotated by *angle_deg* around its centre, so a successful
    perspective correction has to both find it and straighten it.
    *paper_shadow* lays a real brightness ramp across the paper itself
    (a hand or a lamp at the edge of the frame), which is what the
    illumination pass exists to remove.
    """
    rng = np.random.default_rng(seed)
    frame = np.full((600, 800, 3), 48, dtype=np.uint8)  # dark desk
    if gradient:
        ramp = np.linspace(90, 250, 800).astype(np.uint8)
        frame[:, :, 0] = ramp[None, :]
        frame[:, :, 1] = ramp[None, :]
        frame[:, :, 2] = ramp[None, :]
    else:
        frame = frame + rng.integers(0, 10, frame.shape, dtype=np.uint8)
    if page:
        page_img = np.full((360, 500, 3), 255, dtype=np.uint8)
        for row in range(40, 320, 40):  # text lines
            cv2.rectangle(page_img, (50, row), (450, row + 12), (10, 10, 10), -1)
        if paper_shadow:
            ramp = np.linspace(150, 255, 500).astype(np.uint8)
            for ch in range(3):
                page_img[:, :, ch] = (page_img[:, :, ch].astype(int)
                                      * ramp // 255).astype(np.uint8)
        matrix = cv2.getRotationMatrix2D((250, 180), angle_deg, 1.0)
        rotated = cv2.warpAffine(page_img, matrix, (500, 360),
                                 borderValue=(48, 48, 48))
        x0, y0 = (800 - 500) // 2, (600 - 360) // 2
        frame[y0:y0 + 360, x0:x0 + 500] = rotated
    return frame


@unittest.skipUnless(HAS_CV2, "opencv is not installed in this environment")
class TestPreprocessArray(unittest.TestCase):
    def test_rotated_page_is_warped_upright(self):
        img = _make_photo(angle_deg=8.0)
        res = preprocess.preprocess_array(img)
        self.assertTrue(res.quad_found, "page outline should be found")
        self.assertTrue(res.warped, "page should be perspective-corrected")
        self.assertTrue(res.changed)
        out = res.image
        h, w = out.shape[:2]
        # The output is the page, not the frame: about 500x360, not 800x600.
        self.assertAlmostEqual(w / h, 500 / 360, delta=0.15)
        self.assertLess(abs(w - 500), 80)
        self.assertLess(abs(h - 360), 80)
        # After correction the page fills the frame. Flat-field division
        # maps paper white to ~128 by design, so "bright" is a majority of
        # pixels above mid grey, not a mean above 200.
        self.assertGreater(float((out[:, :, 0] > 100).mean()), 0.5)

    def test_illumination_gradient_is_reduced(self):
        # A page sitting inside the frame with a real shadow across the
        # paper. Compare the SAME pixels twice: warp only, then warp plus
        # the flat-field division. The metric is the fitted brightness
        # SLOPE across the page, not raw std: the division also lifts the
        # pixels around ink (the blur estimate is darkened by the text),
        # which adds local noise on purpose. What must die is the ramp.
        img = _make_photo(angle_deg=0.0, paper_shadow=True)
        res = preprocess.preprocess_array(img)
        self.assertTrue(res.warped)
        self.assertTrue(res.illumination)
        quad = preprocess._find_quad(img, cv2, np)
        warped_only = preprocess._warp(img, quad, cv2, np)

        def slope(im):
            mask = im[:, :, 0] > 100
            xs = np.nonzero(mask.any(axis=0))[0]
            col_mean = np.array(
                [im[:, int(x), 0][mask[:, int(x)]].mean() for x in xs])
            return float(np.polyfit(xs.astype(float), col_mean, 1)[0])

        slope_before, slope_after = slope(warped_only), slope(res.image)
        self.assertGreater(abs(slope_before), 0.1,
                           "the on-paper shadow should be a real ramp")
        self.assertLess(abs(slope_after), abs(slope_before) * 0.25,
                        "flat-field division should flatten the ramp")

    def test_clean_page_text_survives(self):
        # A CLEAN page photographed on a desk: sub-frame, so the pass runs,
        # and the ink plus the paper must both come through it.
        img = _make_photo(angle_deg=0.0)
        res = preprocess.preprocess_array(img)
        self.assertTrue(res.warped)
        self.assertTrue(res.illumination)
        out = res.image
        # Paper survives as the bright majority (flat-field maps it to ~128).
        self.assertGreater(float((out[:, :, 0] > 100).mean()), 0.5)
        # Ink survives as a real dark component: the text lines cover tens
        # of thousands of pixels of the ~180k page; the desk is cropped
        # away by the warp, so it cannot fake this number.
        self.assertGreater(int((out[:, :, 0] < 60).sum()), 20000)

    def test_no_page_outline_means_no_change(self):
        # Noise frame with no quadrilateral anywhere: there is no page to
        # correct, so the pass must decline to run (measured: the light
        # pass alone was a net loss on full-frame inputs).
        rng = np.random.default_rng(3)
        img = rng.integers(120, 200, (400, 600, 3), dtype=np.uint8)
        res = preprocess.preprocess_array(img)
        self.assertFalse(res.quad_found)
        self.assertFalse(res.changed)
        self.assertIs(res.image, img)
        self.assertIn("no page outline", res.notes)

    def test_full_frame_page_is_untouched(self):
        # A page that fills the frame with a visible border (an edge-to-edge
        # photo or a scan with a dark rim): the quad is found, but there is
        # nothing to crop and the light pass measured as a loss here.
        img = np.full((900, 780, 3), 40, dtype=np.uint8)
        cv2.rectangle(img, (8, 8), (771, 891), (255, 255, 255), -1)
        for row in range(60, 800, 60):
            cv2.rectangle(img, (60, row), (720, row + 16), (10, 10, 10), -1)
        res = preprocess.preprocess_array(img)
        self.assertTrue(res.quad_found)
        self.assertFalse(res.changed)
        self.assertIs(res.image, img)
        self.assertIn("fills the frame", res.notes)

    def test_deterministic_output(self):
        img = _make_photo(angle_deg=6.0)
        first = preprocess.preprocess_array(img).image
        second = preprocess.preprocess_array(img).image
        self.assertTrue(np.array_equal(first, second))

    def test_unexpected_failure_returns_original(self):
        img = _make_photo(angle_deg=5.0)
        with mock.patch.object(preprocess, "_find_quad",
                               side_effect=RuntimeError("boom")):
            res = preprocess.preprocess_array(img)
        self.assertFalse(res.changed)
        self.assertIs(res.image, img)
        self.assertIn("pass failed", res.notes)


@unittest.skipUnless(HAS_CV2, "opencv is not installed in this environment")
class TestPreprocessImage(unittest.TestCase):
    def _write(self, tmpdir, name, img):
        path = os.path.join(tmpdir, name)
        cv2.imwrite(path, img)
        return path

    def test_corrupt_file_returns_original_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "broken.png")
            with open(path, "wb") as f:
                f.write(b"this is not an image")
            self.assertEqual(preprocess.preprocess_image(path), path)

    def test_photo_gets_a_temp_file_and_round_trips(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, "photo.png", _make_photo(angle_deg=8.0))
            out = preprocess.preprocess_image(path, out_dir=d)
            self.assertNotEqual(out, path, "a real photo should be rewritten")
            self.assertTrue(os.path.isfile(out))
            self.assertLess(os.path.getsize(out), 8 * 1024 * 1024)

    def test_cv2_missing_is_a_passthrough(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, "photo.png", _make_photo(angle_deg=8.0))
            with mock.patch.dict(sys.modules, {"cv2": None, "numpy": None}):
                self.assertEqual(preprocess.preprocess_image(path), path)


if __name__ == "__main__":
    unittest.main()
