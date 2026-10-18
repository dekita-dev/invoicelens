"""Output writers: CSV, JSON, and a human-readable summary."""

from __future__ import annotations

import csv
import io
import json
import os


CSV_COLUMNS = (
    "file", "vendor", "invoice_number", "invoice_date", "due_date",
    "currency", "subtotal", "tax", "total", "category", "deductible",
    "rule_note", "extraction_attempts", "extraction_repaired", "error",
)


def _row(record):
    f = record["fields"]
    return {
        "file": os.path.basename(record["file"]),
        "vendor": f.get("vendor"),
        "invoice_number": f.get("invoice_number"),
        "invoice_date": f.get("invoice_date"),
        "due_date": f.get("due_date"),
        "currency": f.get("currency"),
        "subtotal": f.get("subtotal"),
        "tax": f.get("tax"),
        "total": f.get("total"),
        "category": record["category"],
        "deductible": record["deductible"],
        "rule_note": record["rule_note"],
        "extraction_attempts": record["attempts"],
        "extraction_repaired": record["repaired"],
        "error": record["error"] or "",
    }


def write_csv(records, path):
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for rec in records:
            writer.writerow(_row(rec))


def write_json(records, path):
    payload = []
    for rec in records:
        f = rec["fields"]
        payload.append({
            "file": os.path.basename(rec["file"]),
            "fields": f,
            "category": rec["category"],
            "deductible": rec["deductible"],
            "rule_note": rec["rule_note"],
            "extraction": {
                "attempts": rec["attempts"],
                "repaired": rec["repaired"],
                "errors": rec["errors"],
            },
            "arithmetic": rec.get("arithmetic"),
            "error": rec["error"],
        })
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _fmt_money(rec):
    f = rec["fields"]
    total = f.get("total")
    cur = f.get("currency") or ""
    if total is None:
        return "n/a"
    try:
        return "%s %.2f" % (cur, float(total))
    except (TypeError, ValueError):
        return str(total)


def write_summary(records, path, model_desc, rules_desc):
    lines = []
    lines.append("# InvoiceLens run summary")
    lines.append("")
    lines.append("- model: %s" % model_desc)
    lines.append("- rules: %s" % rules_desc)
    lines.append("- files processed: %d" % len(records))
    ok = [r for r in records if not r["error"]]
    lines.append("- extracted: %d" % len(ok))
    lines.append("- needed a repair prompt: %d"
                 % sum(1 for r in records if r["repaired"]))
    lines.append("- failed: %d" % (len(records) - len(ok)))
    lines.append("- arithmetic mismatches: %d"
                 % sum(1 for r in records if r.get("arithmetic")))
    lines.append("")

    by_cat = {}
    for rec in ok:
        by_cat.setdefault(rec["category"], []).append(rec)

    lines.append("## By category")
    lines.append("")
    if not by_cat:
        lines.append("_No categories matched._")
    for cat in sorted(by_cat):
        recs = by_cat[cat]
        lines.append("### %s (%d)" % (cat, len(recs)))
        lines.append("")
        for rec in recs:
            f = rec["fields"]
            lines.append("- **%s** | %s | %s | invoice date %s"
                         % (f.get("vendor") or "unknown vendor",
                            _fmt_money(rec),
                            rec["deductible"],
                            f.get("invoice_date") or "unknown"))
        lines.append("")

    lines.append("## Needs review")
    lines.append("")
    review = [r for r in records
              if r["error"] or r["deductible"] is None
              or r.get("arithmetic")]
    if not review:
        lines.append("_Nothing._")
    for rec in review:
        if rec["error"]:
            lines.append("- `%s`: %s" % (os.path.basename(rec["file"]),
                                         rec["error"]))
        elif rec.get("arithmetic"):
            lines.append("- `%s`: %s" % (os.path.basename(rec["file"]),
                                         rec["arithmetic"]))
        else:
            lines.append("- `%s`: %s" % (os.path.basename(rec["file"]),
                                         rec["rule_note"]))
    lines.append("")

    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
