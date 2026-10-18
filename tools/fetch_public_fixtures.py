#!/usr/bin/env python
"""Download the CORD v2 public receipt fixtures and build an eval manifest.

InvoiceLens measures itself on two corpora. The synthetic one
(`fixtures/cjk_synthetic`, from `tools/gen_cjk_fixtures.py`) has known ground
truth but was drawn by this project, so it cannot catch a bias the project
does not know it has. CORD v2 is the independent half: 1,000 real receipts
collected from shops and restaurants, published by NAVER CLOVA under CC BY
4.0. Only the 100-page test split is fetched here, because that is the part
with annotations meant to be held out.

The script downloads through the HuggingFace datasets-server rows API rather
than the 234 MB parquet file: the API hands back each row with a signed CDN
URL for its image, so the fetch stays at the size of the images actually
wanted and needs no parquet reader.

Environment note (measured 2026-10-18): `datasets-server.huggingface.co`
sits behind a flaky edge -- roughly one connection in three stalls instead
of answering, through curl from a shell and through curl from Python alike.
So every fetch here is a curl call with a short `--max-time` (a 7 KB JSON
that cannot arrive in 20 s will not arrive in 120 s) plus curl's own
`--retry-all-errors` loop, downloads run four at a time, already-fetched
images are skipped so a killed run resumes, and any image that never
arrives is reported at the end instead of being silently dropped.

What it writes into `fixtures/public/cord-v2/`:

  images/receipt_NNNN.png   the receipt scans, one per test row
  manifest.json             same shape as the synthetic manifest, so
                            `tools/eval_ocr.py --manifest <this file>` just
                            works: `lines` is the page text rebuilt from the
                            word boxes, `fields` pulls subtotal / tax / total
                            out of the dataset's own parse annotation
  ATTRIBUTION.md            the CC BY 4.0 attribution the license requires,
                            with the retrieval date

Usage:
    python tools/fetch_public_fixtures.py                # full test split
    python tools/fetch_public_fixtures.py --limit 20     # a quick subset
    python tools/fetch_public_fixtures.py --dest somewhere/else
    python tools/fetch_public_fixtures.py --rows-json rows.json  # re-assemble
                            # from rows already on disk (no network for the
                            # API part; images already in images/ are kept)
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

DATASET = "naver-clova-ix/cord-v2"
# The dataset ships a single config (dataset_infos.json calls it "cord-v2"),
# and the rows API accepts "default" as the alias for it -- verified against
# the live API 2026-10-18, where config=default returns 200.
CONFIG = "default"
SPLIT = "test"
ROWS_API = ("https://datasets-server.huggingface.co/rows"
            "?dataset=%s&config=%s&split=%s&offset=%d&length=%d")
PAGE = 50           # the rows API caps a single request below the full split
WORKERS = 4         # image downloads in parallel; the edge stalls, not the CPU
MIN_IMAGE_BYTES = 1024

ATTRIBUTION = """# CORD v2 — attribution and license

These fixtures are from **CORD: A Consolidated Receipt Dataset for Post-OCR
Parsing**, version 2, by Park et al. (NAVER CLOVA), 2019.

- Source repository: https://github.com/clovaai/cord
- HuggingFace dataset: https://huggingface.co/datasets/naver-clova-ix/cord-v2
- License: **Creative Commons Attribution 4.0 International** (CC BY 4.0).
  Verified twice on 2026-10-18: the repository carries `LICENSE-CC-BY` with
  the full CC BY 4.0 text, and the HuggingFace dataset metadata declares
  `license: Creative Commons Attribution 4.0 International License`.
  Full text: https://creativecommons.org/licenses/by/4.0/legalcode
- Retrieved: {date} (test split, {n} receipts, Indonesian and some English)

Only the test split is vendored. Note the upstream caveat, quoted from the
CORD repository README: "1,000 sample dataset will be available soon. Some
class labels shown in the original paper were removed due to Indonesian
legal issues. In particular, the `store_info`, `payment_info`, and `etc`
fields have been removed from the target class to be published." That is
why 1,000 images and not the 11,000 the paper describes.

Citation (from the upstream repository):

    @article{{park2019cord,
      title={{CORD: A Consolidated Receipt Dataset for Post-OCR Parsing}},
      author={{Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop
               and Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk}},
      booktitle={{Document Intelligence Workshop at Neural Information
                  Processing Systems}},
      year={{2019}}
    }}
"""


def _curl(url: str, out_path: str, max_time: int) -> None:
    """Fetch a URL into a file with curl, retrying transient stalls.

    curl rather than urllib on purpose: on a normal machine either works,
    but curl gives the `--retry-all-errors` loop that this flaky edge needs.
    The body goes to a file, never a pipe: subprocess pipes deadlock on
    large responses on this machine. `--noproxy '*'` because the system
    proxy answers 502 for hosts it does not terminate, and HuggingFace is
    reachable directly.
    """
    cmd = ["curl", "-sS", "--noproxy", "*",
           "--connect-timeout", "15", "--max-time", str(max_time),
           "--retry", "10", "--retry-all-errors", "--retry-delay", "1",
           "--retry-max-time", str(max_time * 3),
           "-o", out_path, url]
    last = None
    for attempt in range(2):  # curl's own retry loop is the inner one
        try:
            proc = subprocess.run(cmd, timeout=max_time * 3 + 60)
        except subprocess.TimeoutExpired:
            proc = None
        if proc is not None and proc.returncode == 0 \
                and os.path.isfile(out_path):
            return
        last = "rc=%s" % (proc.returncode if proc else "outer-timeout")
        if attempt == 0:
            time.sleep(3)
    raise RuntimeError("curl could not fetch %s (%s)" % (url, last))


def _get_json(url: str) -> dict:
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        _curl(url, tmp, max_time=30)
        with io.open(tmp, encoding="utf-8") as f:
            return json.load(f)
    except ValueError as exc:
        raise RuntimeError("rows API returned non-JSON for %s (%s)"
                           % (url, exc))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _page_lines(gt: dict) -> list[str]:
    """Rebuild page text from the word boxes, one line per annotation row.

    CORD annotates per word with a quad and a `row_id`; grouping by row_id
    and sorting by the left edge of the quad gives a reading order close
    enough for a page-level CER reference. This is an approximation of the
    dataset's own evaluation, which is word-level, and the manifest README
    says so.
    """
    by_row: dict[int, list[tuple[float, str]]] = {}
    for li, line in enumerate(gt.get("valid_line", [])):
        # row_id lives on the word, not the line; fall back to the line's
        # group_id and finally to the line index, so a missing id still
        # keeps words from two visual rows in one bucket.
        line_fallback = line.get("group_id", li)
        for w in line.get("words", []):
            text = (w.get("text") or "").strip()
            if not text:
                continue
            row = w.get("row_id", line_fallback)
            quad = w.get("quad") or {}
            xs = [v for k, v in quad.items()
                  if k.startswith("x") and isinstance(v, (int, float))]
            by_row.setdefault(row, []).append((min(xs) if xs else 0, text))
    out = []
    for row in sorted(by_row):
        words = [t for _, t in sorted(by_row[row], key=lambda p: p[0])]
        out.append(" ".join(words))
    return out


def _money_fields(gt: dict) -> dict[str, str]:
    """Pull the three money fields the product cares about out of gt_parse.

    Indonesian amounts use the dot as thousands separator (`60.000`), which
    is why the manifest marks the language `id` and eval_ocr passes the
    matching locale hint to the repair layer.

    Two annotation quirks are normalised here, both measured on the live
    test split: a field can be annotated twice, and when the two values are
    identical that is one value; when they conflict (`tax: ['6,818', '0']`)
    the annotation is ambiguous, and guessing which one is right would put
    a wrong number into the ground truth -- so the field is dropped and the
    eval scores the record without it rather than against a coin flip.
    """
    parse = gt.get("gt_parse") or {}
    sub = parse.get("sub_total") or {}
    tot = parse.get("total") or {}
    raw = {
        "subtotal": sub.get("subtotal_price"),
        "tax": sub.get("tax_price"),
        "total": tot.get("total_price"),
    }
    out: dict[str, str] = {}
    for key, v in raw.items():
        if v is None or v == "":
            continue
        if isinstance(v, (list, tuple)):
            uniq = []
            for item in v:
                s = str(item).strip()
                if s and s not in uniq:
                    uniq.append(s)
            if len(uniq) != 1:
                continue
            v = uniq[0]
        s = str(v).strip()
        if s:
            out[key] = s
    return out


def _fetch_rows(limit) -> list[dict]:
    """Page the rows API until `limit` rows (or the whole split) are in hand."""
    first = _get_json(ROWS_API % (DATASET, CONFIG, SPLIT, 0, 1))
    available = first.get("num_rows_total", 0)
    want = available if limit is None else min(limit, available)
    if not want:
        raise RuntimeError("rows API reported an empty %s split" % SPLIT)
    print("dataset : %s (%s split, %d rows available)"
          % (DATASET, SPLIT, available))
    print("fetching: %d receipts" % want)
    rows: list[dict] = []
    while len(rows) < want:
        length = min(PAGE, want - len(rows))
        page = _get_json(ROWS_API % (DATASET, CONFIG, SPLIT, len(rows), length))
        batch = page.get("rows") or []
        if not batch:
            raise RuntimeError("rows API returned no rows at offset %d"
                               % len(rows))
        rows.extend(batch)
        print("  rows: %d/%d" % (len(rows), want))
    return rows[:want]


def _load_rows_json(path: str) -> list[dict]:
    with io.open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"] if isinstance(data, dict) else data
    print("rows    : %d read from %s" % (len(rows), path))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="fetch only the first N rows (default: whole split)")
    ap.add_argument("--dest", default=None,
                    help="target directory (default: "
                         "fixtures/public/cord-v2 next to this repo)")
    ap.add_argument("--rows-json", default=None,
                    help="assemble from a rows JSON already on disk instead "
                         "of calling the API (images already in images/ are "
                         "kept; missing ones are still fetched)")
    args = ap.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest = args.dest or os.path.join(root, "fixtures", "public", "cord-v2")
    img_dir = os.path.join(dest, "images")
    print("dest    : %s" % dest)
    os.makedirs(img_dir, exist_ok=True)

    if args.rows_json:
        rows = _load_rows_json(args.rows_json)
    else:
        rows = _fetch_rows(args.limit)

    # Download phase: four at a time, resume-aware, failure-tolerant per
    # image (the run reports what never arrived instead of dying quietly).
    jobs = []
    for idx, row in enumerate(rows):
        src = ((row.get("row") or {}).get("image") or {}).get("src")
        if not src:
            print("  row %d: no image URL" % idx)
            continue
        name = "receipt_%04d.png" % idx
        path = os.path.join(img_dir, name)
        if os.path.isfile(path) and \
                os.path.getsize(path) >= MIN_IMAGE_BYTES:
            jobs.append((idx, name, path, src, True))   # already on disk
        else:
            jobs.append((idx, name, path, src, False))

    todo = [j for j in jobs if not j[4]]
    if todo:
        print("images  : %d to fetch, %d already on disk"
              % (len(todo), len(jobs) - len(todo)))

    def _one(job):
        idx, name, path, src, _cached = job
        _curl(src, path, max_time=120)
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        if size < MIN_IMAGE_BYTES:
            # A signed URL that expired mid-run returns a small error page,
            # not an image. Refusing it keeps a truncated fetch from
            # silently poisoning the corpus.
            if os.path.exists(path):
                os.unlink(path)
            return idx, name, "too small (%d bytes)" % size
        return idx, name, None

    failures = []
    if todo:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for n, (idx, name, err) in enumerate(
                    pool.map(_one, todo), 1):
                if err:
                    failures.append((idx, name, err))
                if n % 10 == 0 or n == len(todo):
                    print("  images: %d/%d%s"
                          % (n, len(todo), " (%d failed)" % len(failures)
                             if failures else ""))

    # Manifest: only rows whose image actually landed, so the corpus and
    # the manifest can never disagree.
    records = []
    for idx, name, path, _src, cached in jobs:
        if not os.path.isfile(path) or \
                os.path.getsize(path) < MIN_IMAGE_BYTES:
            continue
        gt = (rows[idx].get("row") or {}).get("ground_truth")
        if isinstance(gt, str):
            gt = json.loads(gt)
        gt = gt or {}
        records.append({
            "file": "images/" + name,
            "lang": "id",
            "tier": "real",
            "currency": "IDR",
            "lines": _page_lines(gt),
            "fields": _money_fields(gt),
        })

    manifest = {
        "name": "the CORD v2 public receipt test split",
        "source": "https://huggingface.co/datasets/%s" % DATASET,
        "license": "CC BY 4.0",
        "count": len(records),
        "records": records,
    }
    with io.open(os.path.join(dest, "manifest.json"), "w",
                 encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    today = datetime.date.today().isoformat()
    with io.open(os.path.join(dest, "ATTRIBUTION.md"), "w",
                 encoding="utf-8") as f:
        f.write(ATTRIBUTION.format(date=today, n=len(records)))

    no_total = sum(1 for r in records if "total" not in r["fields"])
    print("manifest: %d records (%d without a total annotation)"
          % (len(records), no_total))
    print("wrote    : %s" % os.path.join(dest, "manifest.json"))
    print("wrote    : %s" % os.path.join(dest, "ATTRIBUTION.md"))
    if failures:
        print("\n%d image(s) never arrived; re-run to resume:" % len(failures))
        for idx, name, err in failures:
            print("  row %d (%s): %s" % (idx, name, err))
        return 1
    print("\nMeasure it with:")
    print("  python tools/eval_ocr.py --manifest %s --prep both"
          % os.path.join("fixtures", "public", "cord-v2", "manifest.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
