"""Turn an input file into plain text.

Supported in v1: .txt and text-based .pdf.
Supported as of 2026-10-17: images (.png/.jpg/.jpeg/.webp/.bmp) through the
OCR path in invoicelens.ocr, with the deterministic repair pass applied first.
Scanned PDFs (a PDF with no text layer) still raise, because turning PDF pages
into images needs a renderer this package does not ship. Refusing loudly beats
silently extracting nothing and reporting a confident zero.
"""

from __future__ import annotations

import io
import os

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class UnsupportedSource(Exception):
    """Raised when a file cannot be turned into text without OCR."""


def _read_image(path: str, preprocess: bool = False) -> str:
    from . import ocr, textnorm

    if ocr.available_engines() == []:
        raise UnsupportedSource(
            "this is an image and no OCR engine is installed. Install one with: "
            "pip install rapidocr-onnxruntime onnxruntime"
        )
    try:
        res = ocr.ocr_image(path, preprocess=preprocess)
    except ocr.OCRUnavailable as exc:
        raise UnsupportedSource(str(exc)) from exc
    except Exception as exc:
        # A corrupt or truncated image surfaces as whatever the engine raises
        # (rapidocr raises LoadImageError). The caller only knows one error
        # type, and a stack trace from a third-party library is not an answer
        # a customer can act on.
        raise UnsupportedSource(
            "could not read %s: %s" % (os.path.basename(path), exc)
        ) from exc
    text = textnorm.repair(res.text).strip()
    if not text:
        raise UnsupportedSource(
            "OCR found no text in %s. The image may be blank, rotated, or "
            "too low-resolution." % os.path.basename(path)
        )
    return text


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise UnsupportedSource(
            "pypdf is not installed. Run: pip install pypdf"
        ) from exc

    reader = PdfReader(io.open(path, "rb"))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - malformed pdf
            raise UnsupportedSource("could not read PDF: %s" % exc) from exc
    text = "\n".join(pages).strip()
    if not text:
        raise UnsupportedSource(
            "PDF has no extractable text layer. It is probably a scan: the "
            "pages are images, and turning them back into images needs a "
            "renderer this package does not ship. Export the pages as PNG or "
            "JPG and pass those instead -- the image path reads them."
        )
    return text


def read_text(path: str, preprocess: bool = False) -> str:
    """Return the text content of *path*, or raise UnsupportedSource.

    *preprocess* only affects images: the optional deskew/light pass before
    OCR, off by default (see ocr.ocr_image for the measurement behind that).
    Text and text-based PDFs ignore it.
    """
    if not os.path.isfile(path):
        raise UnsupportedSource("not a file: %s" % path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with io.open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
        if not text:
            raise UnsupportedSource("empty text file: %s" % path)
        return text
    if ext == ".pdf":
        return _read_pdf(path)
    if ext in IMAGE_EXTS:
        return _read_image(path, preprocess)
    raise UnsupportedSource(
        "unsupported file type '%s'. Handles .txt, text-based .pdf and images "
        "(%s)." % (ext, " ".join(sorted(IMAGE_EXTS)))
    )
