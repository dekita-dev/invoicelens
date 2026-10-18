"""Tests for the PDF path, which until now had none.

`.pdf` is the format most invoices arrive in, and it was the only reader in
`textsource` with zero coverage -- the image path had a test file, the text
path had a test file, and the PDF path had a docstring. These tests exist
because of that gap, not because anything was seen failing.

Two properties matter to a user:

  * pages are joined in order, so a two-page invoice reads as one document;
  * a PDF with no text layer is refused loudly instead of returning nothing
    and letting the caller report a confident zero.

The fixtures are built by the same generator the corpus uses
(`tools/gen_multipage_pdf.py`), so these tests fail if that writer breaks.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from invoicelens.textsource import UnsupportedSource, read_text  # noqa: E402
import gen_multipage_pdf as mp  # noqa: E402


def test_two_pages_are_joined_in_order(tmp_path):
    path = str(tmp_path / "two.pdf")
    mp.write_pdf(path, [
        ["HEAD:Vendor One", "page one marker"],
        ["HEAD:Vendor One", "page two marker"],
    ])
    text = read_text(path)
    assert "page one marker" in text
    assert "page two marker" in text
    assert text.index("page one marker") < text.index("page two marker")


def test_every_page_survives(tmp_path):
    # Three pages, because an off-by-one that drops the last page is the
    # most likely way a join goes wrong and a two-page fixture cannot see it.
    path = str(tmp_path / "three.pdf")
    mp.write_pdf(path, [["page %d body" % i] for i in (1, 2, 3)])
    text = read_text(path)
    for i in (1, 2, 3):
        assert "page %d body" % i in text


def test_lines_within_a_page_keep_their_order(tmp_path):
    path = str(tmp_path / "lines.pdf")
    mp.write_pdf(path, [["first line", "second line", "third line"]])
    text = read_text(path)
    assert text.index("first line") < text.index("second line") \
        < text.index("third line")


def test_pdf_without_a_text_layer_is_refused(tmp_path):
    # An empty page is the shape a scanned PDF presents to pypdf: the file
    # is valid, the page has no text. Returning "" here would let the caller
    # report a zero-total invoice with full confidence.
    path = str(tmp_path / "scan.pdf")
    mp.write_pdf(path, [[]])
    with pytest.raises(UnsupportedSource) as exc:
        read_text(path)
    assert "no extractable text layer" in str(exc.value)


def test_missing_file_is_refused_before_parsing(tmp_path):
    with pytest.raises(UnsupportedSource):
        read_text(str(tmp_path / "nope.pdf"))


def test_parentheses_do_not_break_extraction(tmp_path):
    # Unescaped ( or ) in a PDF literal string corrupts the content stream,
    # and the symptom is a page that looks fine in a viewer but extracts
    # nothing. The writer escapes them; this pins that it does.
    path = str(tmp_path / "parens.pdf")
    mp.write_pdf(path, [["Consulting (phase 1) - 2 days"]])
    assert "Consulting (phase 1)" in read_text(path)
