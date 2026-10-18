"""Generate multi-page text-layer PDF invoices, with exact ground truth.

Why this exists
---------------
Two things were untested before this script:

  * `textsource._read_pdf` had no test at all, despite being the path every
    PDF invoice takes;
  * "one PDF, several pages, one document total" was an assumption nobody
    had checked. A tool that silently sums two pages' totals, or reports
    only the last page, is wrong in a way a single-page fixture can never
    show.

Why English only
----------------
These PDFs use the base-14 Helvetica font with WinAnsi encoding. That font
is guaranteed present in every reader and needs no embedding, which keeps
this generator pure standard library -- but it only covers Latin-1. A CJK
text layer requires embedding a font file (several MB) and a ToUnicode
map; that is a different piece of work, and faking it would produce a PDF
whose extracted text is mojibake, i.e. a fixture that measures nothing.

Usage
  python tools/gen_multipage_pdf.py            # writes fixtures/multipage/
  python tools/gen_multipage_pdf.py --out DIR
"""

from __future__ import annotations

import argparse
import io
import json
import os

# --------------------------------------------------------------------------
# Five groups on purpose: one is an anecdote, five can show whether a failure
# is systematic. Each is a real invoice shape -- services, goods, retainer,
# expenses, subscription -- and each puts its total on page 2, because a
# reader that stops after page 1 would otherwise look correct.
# --------------------------------------------------------------------------
GROUPS = [
    {
        "id": "mp-legal-services",
        "vendor": "Ellery & Pike LLP",
        "address": "1201 Third Avenue, Suite 2200, Seattle, WA 98101",
        "doc_no": "EP-2026-0914",
        "date": "09/14/2026",
        "tax_label": "Sales Tax (8.75%)",
        "items_p1": [
            ["Contract drafting", "8", "425.00", "3,400.00"],
            ["Client call attendance", "3", "425.00", "1,275.00"],
        ],
        "items_p2": [
            ["Document production", "2", "180.00", "360.00"],
            ["Courier charges", "1", "45.00", "45.00"],
        ],
        "subtotal": "5,080.00",
        "tax": "444.50",
        "total": "5,524.50",
    },
    {
        "id": "mp-lab-supplies",
        "vendor": "Corvid Lab Supply",
        "address": "77 Research Drive, Cambridge, MA 02139",
        "doc_no": "CL-4471",
        "date": "09/21/2026",
        "tax_label": "Sales Tax (6.25%)",
        "items_p1": [
            ["Glassware set, 24 pieces", "2", "310.00", "620.00"],
            ["Reagent grade ethanol, 4L", "3", "96.00", "288.00"],
        ],
        "items_p2": [
            ["Nitrile gloves, box of 100", "10", "18.50", "185.00"],
            ["Cold shipping surcharge", "1", "65.00", "65.00"],
        ],
        "subtotal": "1,158.00",
        "tax": "72.38",
        "total": "1,230.38",
    },
    {
        "id": "mp-design-retainer",
        "vendor": "Fieldnote Studio",
        "address": "9 Ossington Avenue, Toronto, ON M6J 2Y8",
        "doc_no": "FS-2026-118",
        "date": "09/30/2026",
        "tax_label": "HST (13.00%)",
        "items_p1": [
            ["Brand system, phase 1", "1", "4,200.00", "4,200.00"],
            ["Icon set, 40 glyphs", "1", "1,150.00", "1,150.00"],
        ],
        "items_p2": [
            ["Figma library handoff", "1", "650.00", "650.00"],
            ["Additional revision round", "2", "225.00", "450.00"],
        ],
        "subtotal": "6,450.00",
        "tax": "838.50",
        "total": "7,288.50",
    },
    {
        "id": "mp-travel-expenses",
        "vendor": "Meridian Travel Desk",
        "address": "455 Market Street, Floor 8, San Francisco, CA 94105",
        "doc_no": "MT-2026-2210",
        "date": "10/02/2026",
        "tax_label": "Tax and fees (9.50%)",
        "items_p1": [
            ["Airfare, SFO to NRT, round trip", "1", "1,180.00", "1,180.00"],
            ["Hotel, 4 nights", "4", "265.00", "1,060.00"],
        ],
        "items_p2": [
            ["Airport transfer", "2", "68.00", "136.00"],
            ["Travel insurance", "1", "94.00", "94.00"],
        ],
        "subtotal": "2,470.00",
        "tax": "234.65",
        "total": "2,704.65",
    },
    {
        "id": "mp-software-annual",
        "vendor": "Ridgeline Software Inc.",
        "address": "300 Congress Avenue, Suite 1400, Austin, TX 78701",
        "doc_no": "RS-2026-8842",
        "date": "10/05/2026",
        "tax_label": "Sales Tax (8.25%)",
        "items_p1": [
            ["Workspace plan, annual, 25 seats", "25", "84.00", "2,100.00"],
            ["Audit log retention, annual", "1", "480.00", "480.00"],
        ],
        "items_p2": [
            ["Onboarding workshop", "1", "750.00", "750.00"],
            ["Priority support, annual", "1", "600.00", "600.00"],
        ],
        "subtotal": "3,930.00",
        "tax": "324.23",
        "total": "4,254.23",
    },
]


def _escape(text: str) -> str:
    """Escape the three characters that would break a PDF literal string."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines) -> bytes:
    """One text object per line, top down. No wrapping: the lines are short."""
    parts = []
    y = 780
    for line in lines:
        size = 17 if line.startswith("HEAD:") else 11
        text = line[5:] if line.startswith("HEAD:") else line
        parts.append("BT /F1 %d Tf 50 %d Td (%s) Tj ET"
                     % (size, y, _escape(text)))
        y -= 12 + size
    return "\n".join(parts).encode("latin-1")


def write_pdf(path: str, pages) -> None:
    """Write a minimal, valid PDF: Helvetica, WinAnsi, one page per entry.

    Built by hand because the alternative is a dependency (reportlab) that
    this project does not otherwise need, for 60 lines of output.
    """
    objs: list[bytes] = []

    def add(body: bytes) -> int:
        objs.append(body)
        return len(objs)          # 1-based object number

    # Reserve numbers first: catalog, pages, font, then page/content pairs.
    n_pages = len(pages)
    page_nums = [4 + 2 * i for i in range(n_pages)]
    add(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join("%d 0 R" % n for n in page_nums)
    add(("<< /Type /Pages /Kids [%s] /Count %d >>"
         % (kids, n_pages)).encode("latin-1"))
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>")

    for i, lines in enumerate(pages):
        content_num = page_nums[i] + 1
        add(("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
             "/Resources << /Font << /F1 3 0 R >> >> "
             "/Contents %d 0 R >>" % content_num).encode("latin-1"))
        stream = _content_stream(lines)
        add(b"<< /Length %d >>\nstream\n" % len(stream) + stream
            + b"\nendstream")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for num, body in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % num + body + b"\nendobj\n")
    xref_at = out.tell()
    out.write(b"xref\n0 %d\n" % (len(objs) + 1))
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(b"%010d 00000 n \n" % off)
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
              % (len(objs) + 1, xref_at))
    with open(path, "wb") as f:
        f.write(out.getvalue())


def page_lines(group: dict, page: int) -> list[str]:
    """The visible text of one page, in reading order."""
    head = "HEAD:" + group["vendor"]
    if page == 1:
        return [
            head,
            group["address"],
            "INVOICE    Invoice No.: %s" % group["doc_no"],
            "Invoice Date: %s" % group["date"],
            "",
            "Description                Qty   Unit Price      Amount",
        ] + ["%-26s %5s %10s %12s" % tuple(row) for row in group["items_p1"]] + [
            "",
            "Subtotal (page 1 of 2)                              %12s"
            % _page1_subtotal(group),
            "",
            "Continued on page 2.",
        ]
    return [
        head,
        "INVOICE %s - page 2 of 2" % group["doc_no"],
        "",
        "Description                Qty   Unit Price      Amount",
    ] + ["%-26s %5s %10s %12s" % tuple(row) for row in group["items_p2"]] + [
        "",
        "Subtotal                                            %12s"
        % group["subtotal"],
        "%s                     %12s" % (group["tax_label"], group["tax"]),
        "Total Due                                           %12s"
        % group["total"],
        "",
        "Payable within 30 days. Thank you.",
    ]


def _page1_subtotal(group: dict) -> str:
    """Sum of page 1 only, printed on page 1 as a running figure.

    Deliberately NOT the document total: the point of the fixture is that a
    reader joining both pages must end up with the document total, and one
    that stops early must be visibly wrong rather than accidentally right.
    """
    cents = 0
    for row in group["items_p1"]:
        cents += int(round(float(row[3].replace(",", "")) * 100))
    return "{:,}".format(cents // 100) + ".%02d" % (cents % 100)


def _cents(text: str) -> int:
    return int(round(float(text.replace(",", "")) * 100))


def check_group_arithmetic() -> None:
    """Stop if a group's numbers disagree with its own line items.

    Two separate claims are checked: the line items across BOTH pages sum to
    the subtotal (this is what makes "join the pages" the correct behaviour),
    and subtotal + tax equals the total. Getting either wrong would turn a
    generator typo into an OCR failure in the report.
    """
    bad = []
    for g in GROUPS:
        items = sum(_cents(r[3]) for r in g["items_p1"] + g["items_p2"])
        if items != _cents(g["subtotal"]):
            bad.append((g["id"], "items %d != subtotal %d"
                        % (items, _cents(g["subtotal"]))))
        if _cents(g["subtotal"]) + _cents(g["tax"]) != _cents(g["total"]):
            bad.append((g["id"], "%s + %s != %s"
                        % (g["subtotal"], g["tax"], g["total"])))
    if bad:
        raise SystemExit("FATAL: group arithmetic is wrong: %s" % bad)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    check_group_arithmetic()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out or os.path.join(root, "fixtures", "multipage")
    os.makedirs(out_dir, exist_ok=True)

    records = []
    for group in GROUPS:
        name = "%s.pdf" % group["id"]
        write_pdf(os.path.join(out_dir, name),
                  [page_lines(group, 1), page_lines(group, 2)])
        records.append({
            "file": name,
            "lang": "en",
            "pages": 2,
            "fields": {
                "vendor": group["vendor"],
                "doc_no": group["doc_no"],
                "date": group["date"],
                "subtotal": group["subtotal"],
                "tax": group["tax"],
                "total": group["total"],
            },
            # Text that can only come from one specific page. Used to prove
            # page order survives extraction, which a concatenated blob alone
            # cannot show.
            "page1_marker": "Continued on page 2.",
            "page2_marker": "page 2 of 2",
            "page1_subtotal": _page1_subtotal(group),
        })

    manifest = os.path.join(out_dir, "manifest.json")
    with io.open(manifest, "w", encoding="utf-8") as f:
        json.dump({"name": "multi-page text-layer invoices",
                   "count": len(records), "records": records},
                  f, ensure_ascii=False, indent=1)
    print("wrote %d PDFs -> %s" % (len(records), out_dir))
    print("manifest -> %s" % manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
