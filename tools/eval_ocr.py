"""Score an OCR engine against a fixture set: synthetic CJK or public.

Three numbers, because one is never enough:

  CER       character error rate on the full page text. Catches "the engine
            read something, but the wrong something".
  norm-CER  same, after NFKC + whitespace removal. Separates "the engine
            inserts spaces between CJK words" (harmless, normalised away)
            from "the engine confuses 発 and 秃" (not harmless).
  fields    per-field exact-substring hit rate for vendor, invoice number,
            date and total. This is the number that matters to the product:
            a user does not care that 97% of glyphs were right, they care
            that the total came out as 12,296.00 and not 12,296.00-.

Usage
  python tools/eval_ocr.py                      # engine auto-detect
  python tools/eval_ocr.py --engine tesseract
  python tools/eval_ocr.py --prep both           # raw vs prepped gain table
  python tools/eval_ocr.py --report report/ocr_baseline.md
  python tools/eval_ocr.py --manifest fixtures/public/cord-v2/manifest.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from invoicelens import ocr, textnorm  # noqa: E402


def norm(s) -> str:
    # str() coercion on purpose: a hand-written manifest can carry a list or
    # a number where a string is expected, and the harness should score that
    # field as a miss rather than die with a TypeError halfway through a
    # seven-minute run. (The CORD v2 fetcher normalises at the source; this
    # is the boundary guard.)
    s = unicodedata.normalize("NFKC", str(s))
    for ch in " \t\u3000\u00a0\n\r":
        s = s.replace(ch, "")
    return s


def cer(ref: str, hyp: str) -> float:
    """Levenshtein over characters, divided by reference length."""
    r, h = list(ref), list(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def arith_rate(pool) -> float | None:
    """Share of checkable pages whose three money fields add up.

    None when no page in the pool had all three fields (a corpus without
    money annotations cannot be scored on this, and printing 0% would be a
    lie). Reported next to the per-field rates because it catches a
    different failure: every field string can match while one digit is
    wrong, and the sum is what a reader checks.
    """
    checkable = [r for r in pool if r["arithmetic"] is not None]
    if not checkable:
        return None
    return sum(1 for r in checkable if r["arithmetic"]) / len(checkable)


def _pct(x) -> str:
    return "n/a" if x is None else "%.0f%%" % (x * 100)


def field_hit_rate(pool, key: str):
    """Share of pages that hit *key*, or None when nobody annotates it.

    None is not 0%. A public corpus may carry no vendor labels at all
    (CORD v2 does not), and printing 0% would report a total engine failure
    where the corpus simply made no claim. The divide-by-zero is the same
    bug class as an empty pool in the threshold gate.
    """
    annotated = [r for r in pool if key in r["fields"]]
    if not annotated:
        return None
    return sum(1 for r in annotated if r["fields"][key]) / len(annotated)


def usable_hit_rate(pool, key: str):
    """Same, for the post-repair verdict of one money field."""
    annotated = [r for r in pool if key in r["usable"]]
    if not annotated:
        return None
    return sum(1 for r in annotated if r["usable"][key]) / len(annotated)


def total_usable_rate(pool) -> float:
    """Percent of pages whose total survives the repair layer and parses."""
    rate = usable_hit_rate(pool, "total")
    return 0.0 if rate is None else rate * 100


def threshold_exit_code(pool, threshold):
    """(exit code, rate) for the --threshold gate. None means it passed.

    2 is a refusal, not a failure: the rate is a percentage and the flag
    takes a percentage, but a value below 1 is almost certainly a caller
    passing a fraction (0.95) out of habit, and comparing 95.0 against
    0.95 would make the gate pass on a completely failing corpus. A gate
    that cannot go red is worse than no gate, so the input is rejected
    instead of quietly reinterpreted.
    """
    rate = total_usable_rate(pool)
    if 0 < threshold < 1:
        return 2, rate
    return (None if rate >= threshold else 10), rate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--tier", default=None, help="clean | scan | photo | hard")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--prep", choices=["on", "off", "both"], default="on",
                    help="preprocessing pass before OCR: on (default) | off "
                         "(raw) | both (same image twice, gain table)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="exit 10 when the prepped total-usable rate (the "
                         "product metric) falls below this percentage; for "
                         "CI gates and before/after comparisons")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest_path = args.manifest or os.path.join(
        root, "fixtures", "cjk_synthetic", "manifest.json")
    if not os.path.isfile(manifest_path):
        raise SystemExit("manifest missing: %s (run tools/gen_cjk_fixtures.py)"
                         % manifest_path)

    with io.open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    records = manifest["records"]
    if args.tier:
        records = [r for r in records if r["tier"] == args.tier]
    if args.lang:
        records = [r for r in records if r["lang"] == args.lang]
    if not records:
        raise SystemExit("no fixtures matched the filters")

    engines = ocr.available_engines()
    if not engines:
        raise SystemExit(
            "no OCR engine installed.\n"
            "  pip install rapidocr-onnxruntime onnxruntime")
    engine = args.engine or engines[0]
    print("engine: %s   fixtures: %d   (available: %s)"
          % (engine, len(records), ", ".join(engines)))

    img_dir = os.path.dirname(manifest_path)
    rows = []
    t0 = time.time()
    for idx, rec in enumerate(records, 1):
        path = os.path.join(img_dir, rec["file"].replace("/", os.sep))
        # --prep both runs the same image twice so the gain table compares
        # one page against itself, not two different noise draws.
        prep_modes = [True, False] if args.prep == "both" else \
            [args.prep == "on"]
        for prep in prep_modes:
            started = time.time()
            try:
                res = ocr.ocr_image(path, engine=engine, lang=rec["lang"],
                                    preprocess=prep)
                err = None
            except ocr.OCRUnavailable as exc:
                raise SystemExit("engine unavailable: %s" % exc)
            except Exception as exc:  # a crash on one page must not lose the run
                res, err = None, "%s: %s" % (type(exc).__name__, exc)
            elapsed = time.time() - started

            ref = "\n".join(rec["lines"])
            hyp = res.text if res else ""
            raw = cer(ref, hyp)
            nrm = cer(norm(ref), norm(hyp))
            hay = norm(hyp)
            fields = {k: (norm(v) in hay) for k, v in rec["fields"].items()}
            # Same page, after the deterministic repair layer. Reported separately
            # so the repair cannot quietly inflate the text metric.
            # Locale hint for the thousands separator. CJK locales use comma
            # thousands, so `115.500` is admissible as 115,500 there and
            # nowhere else. English belongs in the comma group too: leaving it
            # out meant `1,200.96` misread as `1.200.96` could not be repaired
            # at all (measured 2026-10-18 on the English hard fixture, where
            # that single omission cost the page its total). Indonesian
            # receipts (the CORD v2 public fixtures) are the dot-decimal
            # mirror: `60.000` there means 60,000.
            thousands = {"ja": ",", "ko": ",", "zh-Hans": ",",
                         "zh-Hant": ",", "en": ",", "id": "."}.get(rec["lang"])
            repaired_hay = norm(textnorm.repair(hyp))
            amounts = textnorm.amounts_in_text(hyp, thousands)
            money = {"subtotal", "tax", "total"}
            usable = {}
            parsed = {}
            for k, v in rec["fields"].items():
                if k in money:
                    want = textnorm.parse_amount(v, thousands)
                    usable[k] = want is not None and want in amounts
                    if usable[k]:
                        parsed[k] = want
                else:
                    usable[k] = norm(v) in repaired_hay
            # The second verification: when all three money fields survived
            # parsing, do they add up? Independent of per-field matching --
            # a page can match every field string and still have one misread
            # digit that breaks the sum, and a reader checks the sum.
            if len(parsed) == len(money):
                arith = (parsed["subtotal"] + parsed["tax"]
                         == parsed["total"])
            else:
                arith = None
            rows.append({
                "file": rec["file"], "lang": rec["lang"], "tier": rec["tier"],
                "engine": engine, "prep": prep, "cer": raw, "norm_cer": nrm,
                "fields": fields, "usable": usable, "arithmetic": arith,
                "seconds": elapsed,
                "confidence": (res.confidence if res else None),
                "error": err,
                "text": hyp,
            })
            # A 100-page run takes tens of minutes and used to print nothing
            # until the end, which made "still working" indistinguishable from
            # "hung". One line per pass costs nothing and answers that.
            print("  %3d/%3d  %-44s prep=%-3s %5.1fs"
                  % (idx, len(records), rec["file"],
                     "on" if prep else "off", elapsed), flush=True)

    total = time.time() - t0

    # In --prep both mode the headline tables show the prepped numbers (the
    # product default); the raw run feeds the gain section below.
    headline = [r for r in rows if r["prep"]] if args.prep == "both" else rows

    # Dump the raw rows BEFORE rendering the tables. The measurement is the
    # expensive part (minutes of OCR); the report is a view of it. A crash
    # while formatting once threw away a full run, and the fix is ordering,
    # not a broader try/except: evidence first, presentation second.
    if args.json_out:
        with io.open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"engine": engine, "rows": rows}, f,
                      ensure_ascii=False, indent=1)
        print("json -> %s" % args.json_out)

    def bucket(key: str, pool=None) -> dict:
        out: dict[str, list] = {}
        for r in (pool if pool is not None else headline):
            out.setdefault(r[key], []).append(r)
        return out

    lines_out = []
    lines_out.append("# OCR baseline on %s\n"
                     % (manifest.get("name") or "synthetic CJK invoices"))
    lines_out.append("Engine: `%s`  |  fixtures: %d  |  wall time: %.1fs\n"
                     % (engine, len(rows), total))
    lines_out.append("\n## By language\n")
    lines_out.append("`vendor`/`doc no`/`date` are raw substring hits. `total raw` is a raw")
    lines_out.append("substring hit, `total usable` is what survives the repair layer plus")
    lines_out.append("amount parsing -- i.e. whether the product would get the right number.")
    lines_out.append("`arith` is the share of pages whose subtotal + tax + total all survived")
    lines_out.append("and add up; `n/a` when the corpus has no money annotations.\n")
    lines_out.append("| Language | n | CER | norm-CER | vendor | doc no | date | total raw | total usable | arith |")
    lines_out.append("|---|---|---|---|---|---|---|---|---|---|")
    for lang, group in sorted(bucket("lang").items()):
        n = len(group)
        cer_mean = sum(r["cer"] for r in group) / n
        nrm_mean = sum(r["norm_cer"] for r in group) / n
        lines_out.append("| %s | %d | %.2f%% | %.2f%% | %s | %s | %s | %s | %s | %s |"
                         % (lang, n, cer_mean * 100, nrm_mean * 100,
                            _pct(field_hit_rate(group, "vendor")),
                            _pct(field_hit_rate(group, "doc_no")),
                            _pct(field_hit_rate(group, "date")),
                            _pct(field_hit_rate(group, "total")),
                            _pct(usable_hit_rate(group, "total")),
                            _pct(arith_rate(group))))

    lines_out.append("\n## By tier\n")
    lines_out.append("| Tier | n | CER | norm-CER | total raw | total usable | arith |")
    lines_out.append("|---|---|---|---|---|---|---|")
    for tier, group in sorted(bucket("tier").items()):
        n = len(group)
        cer_mean = sum(r["cer"] for r in group) / n
        nrm_mean = sum(r["norm_cer"] for r in group) / n
        lines_out.append("| %s | %d | %.2f%% | %.2f%% | %s | %s | %s |"
                         % (tier, n, cer_mean * 100, nrm_mean * 100,
                            _pct(field_hit_rate(group, "total")),
                            _pct(usable_hit_rate(group, "total")),
                            _pct(arith_rate(group))))

    if args.prep == "both":
        # Same page, twice: raw vs preprocessed. A negative CER delta means
        # preprocessing helped; a negative total-usable delta means it cost
        # money fields. Both directions are reported, no exceptions.
        raw_rows = [r for r in rows if not r["prep"]]
        prep_rows = [r for r in rows if r["prep"]]
        by_file = {r["file"]: r for r in raw_rows}
        lines_out.append("\n## Gain: raw vs prepped (same images, two passes)\n")
        lines_out.append("| Tier | n | CER raw | CER prepped | dCER | total usable raw | total usable prepped | dTotal |")
        lines_out.append("|---|---|---|---|---|---|---|---|")
        for tier in sorted({r["tier"] for r in rows}):
            pairs = [(by_file[r["file"]], r) for r in prep_rows
                     if r["tier"] == tier and r["file"] in by_file]
            if not pairs:
                continue
            n = len(pairs)
            cer_raw = sum(p["cer"] for p, _ in pairs) / n
            cer_prep = sum(p["cer"] for _, p in pairs) / n
            # Only pages that actually carry a total annotation can move this
            # number. Counting an unannotated page as a miss would blame the
            # engine for a gap in the corpus.
            scored = [(p, q) for p, q in pairs if "total" in p["usable"]]
            m = max(1, len(scored))
            tot_raw = sum(1 for p, _ in scored if p["usable"]["total"]) / m
            tot_prep = sum(1 for _, q in scored if q["usable"]["total"]) / m
            lines_out.append(
                "| %s | %d | %.2f%% | %.2f%% | %+.2fpp | %.0f%% | %.0f%% | %+.0fpp |"
                % (tier, n, cer_raw * 100, cer_prep * 100,
                   (cer_prep - cer_raw) * 100, tot_raw * 100, tot_prep * 100,
                   (tot_prep - tot_raw) * 100))
        # Wall-clock cost of the pass itself.
        t_raw = sum(r["seconds"] for r in raw_rows) / max(1, len(raw_rows))
        t_prep = sum(r["seconds"] for r in prep_rows) / max(1, len(prep_rows))
        lines_out.append("\nMean seconds per image: raw %.1fs, prepped %.1fs "
                         "(the pass itself is inside the prepped number).\n"
                         % (t_raw, t_prep))

    worst = sorted(headline, key=lambda r: -r["norm_cer"])[:5]
    lines_out.append("\n## Worst five pages (norm-CER)\n")
    lines_out.append("| File | norm-CER | total usable | seconds |")
    lines_out.append("|---|---|---|---|")
    for r in worst:
        lines_out.append("| `%s` | %.2f%% | %s | %.1fs |"
                         % (r["file"], r["norm_cer"] * 100,
                            "yes" if r["usable"].get("total") else "**no**",
                            r["seconds"]))
    errs = [r for r in headline if r["error"]]
    if errs:
        lines_out.append("\n## Crashes\n")
        for r in errs:
            lines_out.append("- `%s` — %s" % (r["file"], r["error"]))

    report = "\n".join(lines_out) + "\n"
    print(report)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with io.open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        print("report -> %s" % args.report)
    if args.threshold is not None:
        code, rate = threshold_exit_code(headline, args.threshold)
        if code == 2:
            print("THRESHOLD: --threshold takes a percentage (95), not a "
                  "fraction (0.95); got %r. Refusing to report a pass from a "
                  "unit mistake." % args.threshold)
            return 2
        if code == 10:
            print("THRESHOLD: prepped total usable %.1f%% < %.1f%% "
                  "-> exit 10" % (rate, args.threshold))
            return 10
        print("THRESHOLD: prepped total usable %.1f%% >= %.1f%% -> pass"
              % (rate, args.threshold))
    return 0


if __name__ == "__main__":
    sys.exit(main())
