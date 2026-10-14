"""Tests for rule matching, the text reader, and the report writers."""

import csv
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from invoicelens.classify import classify, load_rules  # noqa: E402
from invoicelens.report import write_csv, write_json, write_summary  # noqa: E402
from invoicelens.textsource import UnsupportedSource, read_text  # noqa: E402

RULES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "rules", "rules.example.yaml")
FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "fixtures")


@pytest.fixture(scope="module")
def rules():
    return load_rules(RULES)


def test_rules_load(rules):
    assert rules.categories
    assert rules.default_deductible is None


def test_classify_matches_software(rules):
    cat, ded, note = classify(
        {"vendor": "GitHub, Inc.", "line_items": [
            {"description": "GitHub Team subscription, October 2026"}]},
        rules)
    assert cat == "software_subscriptions"
    assert ded is True


def test_classify_matches_professional_services(rules):
    cat, ded, _ = classify(
        {"vendor": "Northwind Studio", "line_items": [
            {"description": "Brand identity retainer"}]}, rules)
    assert cat == "professional_services"
    assert ded is True


def test_classify_personal_is_not_deductible(rules):
    cat, ded, _ = classify(
        {"vendor": "Corner Grocery supermarket", "line_items": []}, rules)
    assert cat == "personal"
    assert ded is False


def test_classify_no_match_returns_uncategorized(rules):
    cat, ded, note = classify({"vendor": "Zzyzx LLC", "line_items": []}, rules)
    assert cat == "uncategorized"
    assert ded is None
    assert "Review manually" in note


def test_classify_highest_keyword_count_wins(rules):
    # "office supplies" hits office_supplies 3x; also contains "hardware"
    cat, _, _ = classify(
        {"vendor": "Meridian Office Supply",
         "line_items": [{"description": "office stationery supplies"}]},
        rules)
    assert cat == "office_supplies"


# ---------------------------------------------------------------------------
# Known limitation, pinned on purpose.
#
# This asserts behaviour that is WRONG, not behaviour that is wanted. It is
# here so that anyone who fixes mixed-receipt classification gets a failing
# test telling them to look here, instead of silently changing an output the
# README documents as a known limitation.
#
# The case is real: it is the bundled office-mixed-2026-09.txt fixture, whose
# extracted fields put "Coffee beans, 1kg" in a line item. Keyword scoring
# gives office_supplies 2 hits ("office" from the vendor name, "monitor" from
# "Monitor arm, dual") against meals_entertainment's 1 ("coffee"), so the
# whole invoice, coffee included, is filed as office supplies.
#
# Fixing it needs per-line-item classification or a model pass over the rules.
# When that lands, change this test to assert the fixed outcome and delete the
# matching bullet from the README's Limitations section.
# ---------------------------------------------------------------------------
def test_mixed_receipt_is_misfiled_known_limitation(rules):
    fields = {
        "vendor": "MERIDIAN OFFICE SUPPLY",
        "line_items": [
            {"description": "Standing desk frame"},
            {"description": "Monitor arm, dual"},
            {"description": "A4 paper, 5 reams"},
            {"description": "Ergonomic chair"},
            {"description": "Coffee beans, 1kg"},
        ],
    }
    cat, ded, note = classify(fields, rules)
    # Coffee should be meals_entertainment (deductible=None -> needs review).
    # Today it is not, because one invoice gets exactly one category.
    assert cat == "office_supplies"
    assert ded is True
    assert "capitalized" in note


def test_read_text_txt():
    text = read_text(os.path.join(FIXTURES, "saas-subscription-2026-10.txt"))
    assert "GitHub Team subscription" in text


def test_read_text_rejects_unknown_extension(tmp_path):
    p = tmp_path / "scan.png"
    p.write_bytes(b"\x89PNG\r\n")
    with pytest.raises(UnsupportedSource) as ei:
        read_text(str(p))
    assert "OCR" in str(ei.value) or "unsupported" in str(ei.value)


def test_read_text_rejects_missing_file(tmp_path):
    with pytest.raises(UnsupportedSource):
        read_text(str(tmp_path / "nope.txt"))


def _record(**kw):
    base = {"file": "/tmp/x.txt", "fields": {"vendor": "ACME", "total": 10.0,
                                             "currency": "USD",
                                             "invoice_date": "2026-09-01"},
            "category": "software_subscriptions", "deductible": True,
            "rule_note": "ok", "attempts": 1, "repaired": False,
            "errors": [], "error": None}
    base.update(kw)
    return base


def test_write_csv_and_json(tmp_path):
    recs = [_record(), _record(file="/tmp/y.txt", error="boom",
                               category="uncategorized", deductible=None)]
    write_csv(recs, str(tmp_path / "a.csv"))
    write_json(recs, str(tmp_path / "a.json"))

    with io.open(tmp_path / "a.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["vendor"] == "ACME"
    assert rows[1]["error"] == "boom"

    data = json.load(io.open(tmp_path / "a.json", encoding="utf-8"))
    assert data[0]["fields"]["total"] == 10.0
    assert data[1]["error"] == "boom"


def test_write_summary_lists_review_items(tmp_path):
    recs = [_record(), _record(file="/tmp/y.txt", error="no text layer")]
    write_summary(recs, str(tmp_path / "s.md"), "gemma.gguf", "rules.yaml")
    body = io.open(tmp_path / "s.md", encoding="utf-8").read()
    assert "software_subscriptions" in body
    assert "Needs review" in body
    assert "no text layer" in body
