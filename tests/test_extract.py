"""Tests for the extraction helpers. These never load a model."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from invoicelens.extract import (  # noqa: E402
    _coerce, _extract_json_block, arithmetic_note, extract_fields,
)


class StubLLM:
    """Returns queued replies, one per call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user, max_tokens=512, temp=0.0):
        self.calls.append(user)
        return self.replies.pop(0)


def test_extract_json_block_plain():
    block = _extract_json_block('{"a": 1, "b": [1, 2]}')
    assert json.loads(block) == {"a": 1, "b": [1, 2]}


def test_extract_json_block_with_surrounding_prose():
    text = 'Sure, here you go:\n{"a": 1}\nLet me know if you need more.'
    assert json.loads(_extract_json_block(text)) == {"a": 1}


def test_extract_json_block_ignores_braces_inside_strings():
    text = '{"note": "use {curly} braces", "total": 5}'
    assert json.loads(_extract_json_block(text))["note"] == "use {curly} braces"


def test_extract_json_block_rejects_unbalanced():
    import pytest
    with pytest.raises(ValueError):
        _extract_json_block('{"a": 1')


def test_extract_json_block_rejects_no_object():
    import pytest
    with pytest.raises(ValueError):
        _extract_json_block("no json here at all")


def test_coerce_fills_missing_keys_with_none():
    out = _coerce({"vendor": "ACME"})
    assert out["vendor"] == "ACME"
    assert out["total"] is None
    assert out["line_items"] == []
    assert set(out) == {
        "vendor", "invoice_number", "invoice_date", "due_date", "currency",
        "subtotal", "tax", "total", "line_items", "payment_method", "notes",
    }


def test_extract_first_try_success():
    llm = StubLLM(['{"vendor": "ACME", "total": 10.5, "currency": "USD"}'])
    res = extract_fields(llm, "some invoice text")
    assert res.ok
    assert res.attempts == 1
    assert res.repaired is False
    assert res.fields["vendor"] == "ACME"
    assert res.fields["total"] == 10.5


def test_extract_repairs_after_bad_json():
    llm = StubLLM([
        "Vendor: ACME, total 10.5",                      # not JSON
        '{"vendor": "ACME", "total": 10.5}',             # repaired
    ])
    res = extract_fields(llm, "some invoice text")
    assert res.ok
    assert res.attempts == 2
    assert res.repaired is True
    assert res.fields["vendor"] == "ACME"
    # the repair prompt must carry the parse error and the bad reply
    assert "not valid JSON" in llm.calls[1]
    assert "Vendor: ACME" in llm.calls[1]


def test_extract_gives_up_after_max_attempts():
    llm = StubLLM(["nope", "still nope", "nope again"])
    res = extract_fields(llm, "text", max_attempts=3)
    assert res.ok is False
    assert res.attempts == 3
    assert len(res.errors) == 3
    assert res.fields == {}


def test_extract_stops_on_engine_error_without_retry_storm():
    class Boom(StubLLM):
        def complete(self, *a, **kw):
            raise RuntimeError("model crashed")

    res = extract_fields(Boom([]), "text", max_attempts=3)
    assert res.ok is False
    assert res.attempts == 1
    assert "model crashed" in res.errors[0]


def test_not_an_invoice_is_still_ok():
    llm = StubLLM(['{"vendor": null, "notes": "not an invoice"}'])
    res = extract_fields(llm, "a poem")
    assert res.ok
    assert res.fields["vendor"] is None
    assert res.fields["notes"] == "not an invoice"


def test_arithmetic_consistent_passes():
    # The bundled office-mixed fixture: 1041.5 + 83.32 = 1124.82. In binary
    # floating point that sum is 1124.8199999999998, which is why the check
    # goes through Decimal on the string form.
    fields = {"subtotal": 1041.5, "tax": 83.32, "total": 1124.82}
    assert arithmetic_note(fields) is None


def test_arithmetic_mismatch_is_flagged_with_the_numbers():
    fields = {"subtotal": 11600.0, "tax": 696.0, "total": 122960.0}
    note = arithmetic_note(fields)
    assert note is not None
    assert "11600" in note and "696" in note and "122960" in note


def test_arithmetic_skips_incomplete_rows():
    # A missing field is the extraction's problem, not arithmetic's: there
    # is nothing to cross-check, so no flag.
    assert arithmetic_note({"subtotal": 1, "tax": 2}) is None
    assert arithmetic_note({"total": 3}) is None
    assert arithmetic_note({}) is None
    assert arithmetic_note(None) is None


def test_arithmetic_ignores_non_numbers():
    # The model is told to use null, not to invent; a string value means the
    # reply did not follow the schema, which the parser reports instead.
    assert arithmetic_note({"subtotal": "x", "tax": 1, "total": 2}) is None


def test_arithmetic_handles_string_numbers():
    # Models do return "1124.82" as a string sometimes; str() round-trips.
    fields = {"subtotal": "1041.5", "tax": "83.32", "total": "1124.82"}
    assert arithmetic_note(fields) is None


def test_arithmetic_tax_zero_is_consistent():
    fields = {"subtotal": 2400.0, "tax": 0.0, "total": 2400.0}
    assert arithmetic_note(fields) is None
