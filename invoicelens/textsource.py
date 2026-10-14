"""Turn an input file into plain text.

Supported in v1: .txt and text-based .pdf.
Deliberately NOT supported: images and scanned PDFs. Those need an OCR pass,
which is a separate problem (see README roadmap). Refusing loudly beats
silently extracting nothing and reporting a confident zero.
"""

from __future__ import annotations

import io
import os


class UnsupportedSource(Exception):
    """Raised when a file cannot be turned into text without OCR."""


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
            "PDF has no extractable text layer. It is probably a scan and "
            "needs OCR, which v1 does not do."
        )
    return text


def read_text(path: str) -> str:
    """Return the text content of *path*, or raise UnsupportedSource."""
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
    raise UnsupportedSource(
        "unsupported file type '%s'. v1 handles .txt and text-based .pdf only."
        % ext
    )
