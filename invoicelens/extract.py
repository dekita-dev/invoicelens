"""Field extraction from invoice text using the local model.

The pattern that matters here is the repair loop: the model is asked for a
strict JSON object, and if the reply does not parse, the *parse error itself*
is fed back as the next prompt. That is cheaper and more honest than a
hand-written regex fallback, because the model still owns the extraction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation


MONEY_KEYS = ("subtotal", "tax", "total")


def arithmetic_note(fields) -> str | None:
    """Cross-check subtotal + tax == total. A reason string when they disagree.

    The one verification a reader can do in their head, done deterministically
    so it costs no model call. It catches the failure mode that per-field
    checks miss: the model can return three plausible-looking numbers that
    simply do not add up, which means at least one of them is misread even
    though each one looks fine alone.

    Returns None when the fields are absent (nothing to check -- a missing
    field is the extraction's problem, not arithmetic's), when they are not
    numbers, or when they are consistent. Decimal is used on the string form
    of each value on purpose: 1041.5 + 83.32 != 1124.82 in binary floating
    point, and a check that cries wolf on every float total is worse than no
    check at all.
    """
    values = []
    for key in MONEY_KEYS:
        v = (fields or {}).get(key)
        if v is None:
            return None
        try:
            values.append(Decimal(str(v)))
        except (InvalidOperation, ValueError):
            return None
    sub, tax, total = values
    if sub + tax == total:
        return None
    return ("subtotal %s + tax %s does not equal total %s"
            % (sub, tax, total))


SYSTEM_PROMPT = """You are an invoice field extractor. You are given the raw \
text of one invoice or receipt. You reply with a single JSON object and \
nothing else.

Use exactly this schema:
{
  "vendor": string,
  "invoice_number": string or null,
  "invoice_date": "YYYY-MM-DD" or null,
  "due_date": "YYYY-MM-DD" or null,
  "currency": ISO-4217 code or null,
  "subtotal": number or null,
  "tax": number or null,
  "total": number or null,
  "line_items": [{"description": string, "quantity": number, \
"unit_price": number, "amount": number}],
  "payment_method": string or null,
  "notes": string
}

Rules:
- Reply with the JSON object only. No markdown fences, no commentary, no \
leading or trailing text.
- Use null for anything you cannot find. Never invent a value.
- Normalize every date to YYYY-MM-DD.
- Numbers are plain decimals: no currency symbols, no thousands separators.
- If the document is not an invoice or receipt, reply with \
{"vendor": null, "notes": "not an invoice"} and nothing else."""


REPAIR_TEMPLATE = """Your previous reply was not valid JSON.

Parse error: {error}

Your previous reply was:
---
{bad}
---

Reply again with only the corrected JSON object matching the schema. No \
commentary, no markdown fences."""


REQUIRED_KEYS = (
    "vendor", "invoice_number", "invoice_date", "due_date", "currency",
    "subtotal", "tax", "total", "line_items", "payment_method", "notes",
)


@dataclass
class ExtractionResult:
    fields: dict = field(default_factory=dict)
    attempts: int = 0
    repaired: bool = False
    errors: list = field(default_factory=list)
    raw_last: str = ""

    @property
    def ok(self):
        """True when the model gave us something usable.

        Strictly a bool: an empty result must read as False, not None, so
        callers can branch on it without a truthiness surprise.
        """
        return bool(self.fields.get("vendor") or self.fields.get("notes"))


def _extract_json_block(text):
    """Pull the outermost balanced {...} object out of a model reply."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in reply")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unbalanced braces in reply")


def _coerce(payload):
    """Fill in missing keys with null and keep the shape predictable."""
    out = {k: payload.get(k) for k in REQUIRED_KEYS}
    items = out.get("line_items")
    if not isinstance(items, list):
        out["line_items"] = []
    return out


def build_prompt(text):
    """The user-side prompt for one invoice.

    Kept as a function so the benchmarks (tools/bench_backends.py) measure
    the prompt the product actually sends instead of a lookalike that can
    drift from it.
    """
    return (
        "Extract the fields from this invoice text. Reply with JSON only.\n\n"
        "=== INVOICE TEXT ===\n%s\n=== END ===" % text
    )


def extract_fields(llm, text, max_attempts=3, max_tokens=512):
    """Extract structured fields from invoice *text* with a repair loop."""
    result = ExtractionResult()
    prompt = build_prompt(text)

    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        try:
            raw = llm.complete(SYSTEM_PROMPT, prompt, max_tokens=max_tokens)
        except Exception as exc:  # engine level failure, do not retry blindly
            result.errors.append("engine: %s" % exc)
            return result

        result.raw_last = raw
        try:
            block = _extract_json_block(raw)
            payload = json.loads(block)
            if not isinstance(payload, dict):
                raise ValueError("top level is not an object")
            result.fields = _coerce(payload)
            result.repaired = attempt > 1
            return result
        except Exception as exc:
            err = "attempt %d: %s" % (attempt, exc)
            result.errors.append(err)
            prompt = REPAIR_TEMPLATE.format(error=exc, bad=raw[:1500])

    result.fields = {}
    return result
