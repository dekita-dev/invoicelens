"""Reverse-regression tests for the --threshold gate in tools/eval_ocr.py.

The gate is the only automated go/no-go on OCR quality, so the thing that
must be pinned is not that it can pass (it obviously can) but that it can
go red. Two ways it could silently stay green:

  * the comparison drifting across units -- 95.0 compared against 0.95
    passes on a corpus where every single page failed;
  * the pool being empty -- dividing by max(1, 0) yields 0.0, so the
    arithmetic still runs and a fraction-style threshold still passes.

Both are covered here. A gate that cannot fail is worse than no gate.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import eval_ocr  # noqa: E402


def _row(total_usable):
    return {"usable": {"total": total_usable}}


def test_gate_passes_on_a_clean_pool():
    pool = [_row(True)] * 20
    code, rate = eval_ocr.threshold_exit_code(pool, 95)
    assert code is None
    assert rate == 100.0


def test_gate_goes_red_when_the_metric_drops():
    # 4 of 20 pages keep their total: 20%, far under a 95% bar.
    pool = [_row(True)] * 4 + [_row(False)] * 16
    code, rate = eval_ocr.threshold_exit_code(pool, 95)
    assert rate == 20.0
    assert code == 10


def test_fraction_threshold_is_refused_not_reinterpreted():
    # The exact silent-green bug: 0.95 vs a 95.0-scale rate would pass even
    # with a zero-rate corpus. Refusing (2) is the only safe reading.
    pool = [_row(False)] * 10
    code, rate = eval_ocr.threshold_exit_code(pool, 0.95)
    assert code == 2
    assert rate == 0.0


def test_empty_pool_cannot_pass():
    code, rate = eval_ocr.threshold_exit_code([], 95)
    assert rate == 0.0
    assert code == 10


def test_boundary_is_inclusive():
    # 19/20 is exactly 95.0. The bar is "at least", not "more than".
    pool = [_row(True)] * 19 + [_row(False)] * 1
    code, rate = eval_ocr.threshold_exit_code(pool, 95)
    assert rate == 95.0
    assert code is None


def test_unannotated_field_reports_na_not_zero():
    # CORD v2 carries no vendor/date/doc-no labels at all. Reporting 0% would
    # claim a total engine failure where the corpus made no claim at all.
    pool = [{"fields": {"total": "1.000"}, "usable": {"total": True}}]
    assert eval_ocr.field_hit_rate(pool, "vendor") is None
    assert eval_ocr.usable_hit_rate(pool, "tax") is None


def test_field_rate_uses_only_annotated_rows():
    pool = [
        {"fields": {"vendor": "a"}, "usable": {}},
        {"fields": {"vendor": "a"}, "usable": {}},
        {"fields": {}, "usable": {}},
        {"fields": {}, "usable": {}},
    ]
    # Two of four pages carry a vendor label and both are correct: the rate
    # is 100%, not 50%. The denominator is the evidence, not the corpus size.
    assert eval_ocr.field_hit_rate(pool, "vendor") == 1.0


def test_unannotated_pages_do_not_drag_the_threshold_gate():
    # A corpus where only 2 of 10 pages annotate a total, and both are right,
    # must pass a 95% bar. Otherwise a sparse corpus fails for being sparse.
    pool = [{"usable": {"total": True}}] * 2 + [{"usable": {}}] * 8
    code, rate = eval_ocr.threshold_exit_code(pool, 95)
    assert rate == 100.0
    assert code is None


def test_threshold_gate_on_a_corpus_with_no_totals_at_all():
    # Nobody annotates a total: 0.0 with no evidence behind it. Refusing to
    # call that a pass is the conservative reading.
    pool = [{"usable": {}}] * 5
    code, rate = eval_ocr.threshold_exit_code(pool, 95)
    assert rate == 0.0
    assert code == 10
