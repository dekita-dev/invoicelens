"""Check that a multi-page PDF reads as one document, not as one page.

The claim under test is narrow and easy to get wrong: when an invoice spans
two pages, the tool must join the pages and report the document total, not
the running subtotal printed on page 1 and not the last page's rows alone.

Each group is scored on five independent checks, and the pages are read
through the same public entry point a user hits (`textsource.read_text`),
not through pypdf directly -- a check that bypasses the product's own path
can pass while the product is broken.

Usage
  python tools/check_multipage.py
  python tools/check_multipage.py --report report/multipage.md
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from invoicelens import textnorm  # noqa: E402
from invoicelens.textsource import read_text  # noqa: E402


def _count(hay: str, needle: str) -> int:
    """Occurrence count of *needle*, after the same normalisation the eval
    harness uses (NFKC + all whitespace removed)."""
    def norm(s: str) -> str:
        s = textnorm.repair(str(s))
        for ch in " \t\u3000\u00a0\n\r":
            s = s.replace(ch, "")
        return s
    return norm(hay).count(norm(needle))


def check_group(directory: str, rec: dict) -> dict:
    path = os.path.join(directory, rec["file"])
    problems = []
    try:
        text = read_text(path)
    except Exception as exc:  # noqa: BLE001 - report, do not crash the run
        return {"file": rec["file"], "ok": False,
                "problems": ["read_text raised %s: %s"
                             % (type(exc).__name__, exc)]}

    # 1 + 2. Both pages present, and in the right order. A concatenated blob
    # can contain both strings and still be reversed, which would silently
    # put the payment terms before the amounts.
    p1 = _count(text, rec["page1_marker"])
    p2 = _count(text, rec["page2_marker"])
    if not p1 or not p2:
        problems.append("page marker missing (p1=%d p2=%d)" % (p1, p2))
    elif text.find("Continued on page 2") > text.find("page 2 of 2"):
        problems.append("page order reversed")

    # 3. The document total appears exactly once. Twice means the join
    # duplicated a page; zero means the join dropped one.
    money = rec["fields"]
    n_total = _count(text, money["total"])
    if n_total != 1:
        problems.append("total appears %d times (expected 1)" % n_total)

    # 4. Every money field is present.
    for key in ("subtotal", "tax", "total"):
        if _count(text, money[key]) < 1:
            problems.append("%s missing from text" % key)

    # 5. The page-1 running subtotal must differ from the document total,
    # otherwise the fixture cannot tell "stopped early" from "read it all".
    if _count(text, rec["page1_subtotal"]) < 1:
        problems.append("page-1 running subtotal missing")
    if textnorm.parse_amount(rec["page1_subtotal"]) == \
            textnorm.parse_amount(money["total"]):
        problems.append("page-1 subtotal equals the document total: "
                        "fixture cannot detect an early stop")

    sub = textnorm.parse_amount(money["subtotal"])
    tax = textnorm.parse_amount(money["tax"])
    tot = textnorm.parse_amount(money["total"])
    if None in (sub, tax, tot) or sub + tax != tot:
        problems.append("ground truth does not add up")

    return {"file": rec["file"], "ok": not problems, "problems": problems,
            "chars": len(text)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    directory = args.dir or os.path.join(ROOT, "fixtures", "multipage")
    manifest_path = os.path.join(directory, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise SystemExit("manifest missing: %s (run tools/gen_multipage_pdf.py)"
                         % manifest_path)
    with io.open(manifest_path, encoding="utf-8") as f:
        records = json.load(f)["records"]

    results = [check_group(directory, rec) for rec in records]
    failed = [r for r in results if not r["ok"]]

    lines = ["# Multi-page PDF check", "",
             "Read through `textsource.read_text`, the same call a user's run "
             "makes. Each group is a two-page invoice whose total sits on "
             "page 2, so a reader that stops after page 1 is visibly wrong.",
             "",
             "| File | Chars | Result |", "|---|---|---|"]
    for r in results:
        lines.append("| `%s` | %d | %s |"
                     % (r["file"], r.get("chars", 0),
                        "pass" if r["ok"] else "; ".join(r["problems"])))
    lines.append("")
    lines.append("%d of %d groups pass." % (len(results) - len(failed),
                                            len(results)))
    report = "\n".join(lines) + "\n"
    print(report)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)),
                    exist_ok=True)
        with io.open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        print("report -> %s" % args.report)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
