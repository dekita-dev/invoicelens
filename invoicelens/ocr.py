"""OCR adapter: turn an invoice image into text, without committing to one engine.

Design constraints, in order of importance:

1. **The product is local-first.** No network call, no API key, no upload.
   Every engine here runs on the customer's own CPU.
2. **Fail loudly.** If no engine is installed, raise instead of returning "".
   An empty string looks like "this invoice has no total", which is a wrong
   answer the user cannot see. A raised error is a wrong answer they can.
3. **Engines are interchangeable.** v1 ships one that works today; the eval
   harness in tools/eval_ocr.py scores any engine that implements the same
   call, so a better one can be swapped in without touching the pipeline.

Engines
-------
rapidocr   PP-OCR family models (Apache-2.0) exported to ONNX, shipped as a
           pip wheel. CPU-only, no compiler, no model download at runtime.
tesseract  Optional. Only useful if the Tesseract binary and the CJK language
           packs are installed; treats CJK as a lower-quality fallback.
none       No engine. Explicit, so a missing install is never mistaken for a
           clean extraction.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile


class OCRUnavailable(Exception):
    """No usable OCR engine on this machine."""


class OCRResult:
    __slots__ = ("text", "lines", "engine", "confidence")

    def __init__(self, text: str, lines: list[str], engine: str,
                 confidence: float | None = None):
        self.text = text
        self.lines = lines
        self.engine = engine
        self.confidence = confidence

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<OCRResult engine=%s lines=%d chars=%d>" % (
            self.engine, len(self.lines), len(self.text))


def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def available_engines() -> list[str]:
    found = []
    if _has_module("rapidocr_onnxruntime"):
        found.append("rapidocr")
    if _has_module("paddleocr"):
        found.append("paddleocr")
    if shutil.which("tesseract"):
        found.append("tesseract")
    return found


# --------------------------------------------------------------------------
# rapidocr
# --------------------------------------------------------------------------
_RAPID = None


def _rapid_engine():
    global _RAPID
    if _RAPID is None:
        from rapidocr_onnxruntime import RapidOCR
        _RAPID = RapidOCR()
    return _RAPID


def _ocr_rapidocr(path: str) -> OCRResult:
    engine = _rapid_engine()
    result, _elapse = engine(path)
    if not result:
        return OCRResult("", [], "rapidocr", None)
    lines: list[str] = []
    scores: list[float] = []
    for item in result:
        # rapidocr returns [box, text, score]
        if len(item) >= 3:
            lines.append(str(item[1]))
            try:
                scores.append(float(item[2]))
            except (TypeError, ValueError):
                pass
        else:  # pragma: no cover - shape drift guard
            lines.append(str(item))
    conf = sum(scores) / len(scores) if scores else None
    return OCRResult("\n".join(lines), lines, "rapidocr", conf)


# --------------------------------------------------------------------------
# paddleocr (optional, language-aware)
# --------------------------------------------------------------------------
_PADDLE_LANG = {
    "zh-Hans": "ch",
    "zh-Hant": "chinese_cht",
    "ja": "japan",
    "ko": "korean",
}
_PADDLE_CACHE: dict = {}


def _paddle_engine(lang: str):
    cached = _PADDLE_CACHE.get(lang)
    if cached is not None:
        return cached
    # Must be set before paddle is imported. On Windows, PaddlePaddle 3.3.1 with
    # oneDNN enabled dies on the first inference with
    #   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
    #   [pir::ArrayAttribute<pir::DoubleAttribute>]  (onednn_instruction.cc:118)
    # Turning oneDNN off costs speed but is the difference between 0% and 100%
    # recognition on Japanese. Measured 2026-10-17.
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    from paddleocr import PaddleOCR
    kwargs = {"lang": lang, "enable_mkldnn": False}
    # 3.x renamed the switches; pass them only if the signature takes them.
    try:
        import inspect
        sig = inspect.signature(PaddleOCR.__init__)
        if "use_doc_orientation_classify" in sig.parameters:
            kwargs["use_doc_orientation_classify"] = False
        if "use_doc_unwarping" in sig.parameters:
            kwargs["use_doc_unwarping"] = False
        if "use_textline_orientation" in sig.parameters:
            kwargs["use_textline_orientation"] = False
    except (TypeError, ValueError):  # pragma: no cover - exotic signatures
        pass
    _PADDLE_CACHE[lang] = PaddleOCR(**kwargs)
    return _PADDLE_CACHE[lang]


def _paddle_lines(result) -> tuple[list[str], list[float]]:
    """Normalise the 2.x and 3.x result shapes into (texts, scores)."""
    texts: list[str] = []
    scores: list[float] = []
    if not result:
        return texts, scores
    first = result[0]
    # 3.x: result is a list of dicts (or objects with .json)
    if isinstance(first, dict) or hasattr(first, "json"):
        for page in result:
            payload = page if isinstance(page, dict) else page.json
            if isinstance(payload, dict) and "res" in payload:
                payload = payload["res"]
            for t, s in zip(payload.get("rec_texts", []),
                            payload.get("rec_scores", [])):
                texts.append(str(t))
                scores.append(float(s))
        return texts, scores
    # 2.x: [[ [box, (text, score)], ... ]]
    for page in result:
        for item in page or []:
            try:
                texts.append(str(item[1][0]))
                scores.append(float(item[1][1]))
            except (IndexError, TypeError, ValueError):
                continue
    return texts, scores


def _ocr_paddleocr(path: str, lang: str | None) -> OCRResult:
    key = _PADDLE_LANG.get(lang or "", "ch")
    engine = _paddle_engine(key)
    # Calling the object directly is stable across 2.x and 3.x; predict() only
    # exists in 3.x. Try the callable first.
    try:
        result = engine(path)
    except TypeError:  # pragma: no cover - 3.x-only builds
        result = engine.predict(path)
    texts, scores = _paddle_lines(result)
    conf = sum(scores) / len(scores) if scores else None
    return OCRResult("\n".join(texts), texts, "paddleocr(%s)" % key, conf)


# --------------------------------------------------------------------------
# tesseract (optional)
# --------------------------------------------------------------------------
TESS_LANG = {
    "zh-Hans": "chi_sim",
    "zh-Hant": "chi_tra",
    "ja": "jpn",
    "ko": "kor",
}


def _ocr_tesseract(path: str, lang: str | None = None) -> OCRResult:
    tlang = TESS_LANG.get(lang or "", "eng")
    cmd = ["tesseract", path, "stdout", "-l", tlang, "--psm", "6"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise OCRUnavailable("tesseract failed to run: %s" % exc) from exc
    if out.returncode != 0:
        raise OCRUnavailable(
            "tesseract failed (is the '%s' language pack installed?): %s"
            % (tlang, out.stderr.decode("utf-8", "replace")[:200]))
    text = out.stdout.decode("utf-8", "replace")
    lines = [ln for ln in (l.strip() for l in text.splitlines()) if ln]
    return OCRResult("\n".join(lines), lines, "tesseract", None)


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def _maybe_preprocess(path: str):
    """Run the optional OCR prep pass; return a temp path or None.

    Preprocessing is a best-effort helper: any failure inside it (cv2
    missing, unreadable image, every pass fell back) must leave the OCR
    path exactly as it was, so this never raises.
    """
    try:
        from . import preprocess as _prep
        candidate = _prep.preprocess_image(path)
        return candidate if candidate != path else None
    except Exception:
        return None


def ocr_image(path: str, engine: str | None = None,
              lang: str | None = None,
              preprocess: bool = False) -> OCRResult:
    """OCR one image.

    *lang* only matters for engines that are language-aware.

    *preprocess* opts into the deskew/light pass (see invoicelens.preprocess).
    Off by default: measured on the 60-page corpus, it improves character
    accuracy on photos by 15pp but does not move any field the product
    reports, and costs a little vendor-name recall. It is the right switch
    for a badly skewed photo, not for every photo. The pass falls back to the
    raw image on any problem, so turning it on never adds a failure mode.
    """
    if not os.path.isfile(path):
        raise OCRUnavailable("not a file: %s" % path)

    usable = available_engines()
    if not usable:
        raise OCRUnavailable(
            "no OCR engine installed. Install one with:\n"
            "  pip install rapidocr-onnxruntime onnxruntime")

    engine = engine or usable[0]

    tmp_path = _maybe_preprocess(path) if preprocess else None
    use_path = tmp_path or path
    try:
        if engine == "rapidocr":
            if not _has_module("rapidocr_onnxruntime"):
                raise OCRUnavailable("rapidocr not installed")
            return _ocr_rapidocr(use_path)
        if engine == "paddleocr":
            if not _has_module("paddleocr"):
                raise OCRUnavailable("paddleocr not installed")
            return _ocr_paddleocr(use_path, lang)
        if engine == "tesseract":
            if not shutil.which("tesseract"):
                raise OCRUnavailable("tesseract binary not on PATH")
            return _ocr_tesseract(use_path, lang)
        raise OCRUnavailable("unknown engine: %s" % engine)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def dispose() -> None:
    """Drop the cached engine so a long-lived process can free its memory."""
    global _RAPID
    _RAPID = None
