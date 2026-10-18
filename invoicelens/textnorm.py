"""Deterministic repair of OCR output before it reaches the model.

Why this exists
---------------
The 2B model that extracts fields is good at reading sense and bad at being
told "316_800 is a number". A single misread separator turns a correct read
into a null field, and the user sees "no total found" on an invoice that is
sitting right in front of them.

Measured case, 2026-10-17: PP-OCR read the Japanese fixture totals as
`316_800`, `115_500`, `240, 000` -- every digit right, every separator wrong.
A parser that only accepts `,` throws all of it away.

So: fix what is unambiguously a separator or a digit, refuse to guess at
anything else. No spell correction, no dictionary, no language model. Every
rule below is one an invoice reader can explain to a customer.

Refused on purpose
------------------
  * 请求書 -> 請求書     (script substitution; needs a language model, and a
                          wrong "correction" here changes a legal document)
  * 0/O in words         (repair only between digits)
  * dropped glyphs       (cannot tell "8" from "3" without context)
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

# Separators that OCR confuses with a thousands comma. Grouped by what they
# actually are: punctuation that sits in a number, and characters that come
# from the same glyph neighbourhood on a low-DPI scan.
_SEP = "_,\u201a\u2019\u00b7\u2027\uff0c\u3001;:|"  # , , ' · ， 、 ; : |
_SPACED_SEP = re.compile(r"(\d)\s*[" + re.escape(_SEP) + r"]\s*(\d{3})")

# Digit/letter pairs that share a shape. Only ever applied when the character
# is wedged between digits, so "B2B" and "S1" survive untouched.
_DIGIT_FIX = {"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1",
              "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2"}

_CURRENCY = "¥$€£₩￥元"

# A money token. Whitespace is NOT allowed inside the run: OCR emits
# "1 6,500,000 6,500,000" for a quantity followed by two amounts, and a regex
# that swallows the spaces glues them into one impossible number, which then
# fails parsing and silently loses the real totals. Measured 2026-10-17.
NUM_TOKEN = re.compile(
    r"(?:[" + _CURRENCY + r"]\s*)?\d[\d.,]*")


def fold_digits(text: str) -> str:
    """Full-width -> half-width, and letter/digit confusions between digits."""
    text = unicodedata.normalize("NFKC", text)
    out = []
    for i, ch in enumerate(text):
        if ch in _DIGIT_FIX:
            prev = text[i - 1] if i else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            # Both sides must exist and be numeric-looking. The empty-string
            # check is explicit: "" is a substring of ".," so a naive
            # `prev in ".,"` test silently rewrites the first and last
            # character of every token (B2B -> 828).
            if prev and nxt and (prev.isdigit() or prev in ".,") \
                    and (nxt.isdigit() or nxt in ".,"):
                out.append(_DIGIT_FIX[ch])
                continue
        out.append(ch)
    return "".join(out)


def repair_separators(text: str) -> str:
    """Make every thousand-separator look like a comma, and drop stray spaces.

    `316_800` -> `316,800`;  `240, 000` -> `240,000`;  `1.056.000` is left
    alone on purpose -- it is the European decimal format, not an error, and
    deciding which one it is requires the locale, not the text.
    """
    prev = None
    cur = text
    # Two passes: the first fixes "240, 000", the second catches a separator
    # that only becomes unambiguous after the space is gone.
    while prev != cur:
        prev = cur
        cur = _SPACED_SEP.sub(r"\1,\2", cur)
    # The same break with a dot instead of a comma: `48. 000`. Measured on
    # CORD v2 receipt_0062, where the engine put a space after the thousands
    # dot and the amount stopped parsing. Only an exact three-digit group is
    # joined, so a decimal split across a space (`12. 50`) is left alone.
    cur = re.sub(r"(\d)\s*\.\s*(\d{3})(?!\d)", r"\1.\2", cur)
    # A separator directly after the decimal part is never a comma.
    cur = re.sub(r"(\d),(\d{4,})", r"\1\2", cur)
    return cur


def repair(text: str) -> str:
    """Full OCR repair pass. Safe to call on text that needs no repair."""
    return repair_separators(fold_digits(text))


def parse_amount(token: str, thousands: str | None = None) -> Decimal | None:
    """Parse a money token. Returns None instead of guessing.

    Accepts  `¥12,296.00`, `NT$1,500`, `₩8,635,000`, `12,296-`, `(12,296)`.

    *thousands* resolves the genuinely ambiguous shapes, in the direction
    the locale needs. `115.500` on a Japanese invoice is 115,500 (comma
    misread as a dot); on a German one it is 115.5. Without a hint this
    function returns None rather than pick a reading, because a wrong total
    on a tax document is worse than a missing one. Callers that know the
    locale pass `thousands=","` for comma-decimal locales or
    `thousands="."` for dot-decimal ones -- the second is what the
    Indonesian receipts in fixtures/public need, where `60.000` means
    60,000.

    Not accepted: `12 296,00` (space separator, ambiguous).

    Mixed separators are read by position: whichever comes last is the
    decimal mark, and the other must be in the thousands position. So
    `1,056.00` and `9.500,00` both reduce to a value, and the second one is
    not hypothetical -- CORD v2 receipt_0067 prints the Dutch form on an
    Indonesian receipt (measured 2026-10-18). The one exception is a
    three-digit tail, which is read as another thousands group: no currency
    prints three decimals, and `8.635,000` (Korean photo fixture, comma
    misread as dot) is 8,635,000, not 8635.000.

    Known gap (measured 2026-10-18): the mirror of the comma-locale repair
    -- `6,124,68` under `thousands="."`, comma groups plus a two-digit
    tail -- is NOT implemented. The comma branch runs before any
    dot-locale branch could, so the string dies there. No fixture has
    produced it yet; tests/test_textnorm.py pins the current behaviour so
    the gap cannot close by accident.
    """
    if token is None:
        return None
    s = repair(str(token)).strip()
    if not s:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if s.endswith("-"):
        neg, s = True, s[:-1]
    s = re.sub(r"^[^0-9]+", "", s)          # currency symbols and prefixes
    s = re.sub(r"[^0-9.,]+$", "", s)        # trailing units
    s = s.replace(" ", "").replace("\u3000", "")
    if not re.fullmatch(r"\d[\d.,]*", s):
        return None
    # Malformed grouping must fail: `12,,296` is not 12296, it is a sign that
    # the OCR dropped a glyph and we do not know which one.
    if re.search(r"[.,]{2}", s) or re.search(r"[.,]$", s):
        return None
    if "," in s and "." in s:
        groups = re.split(r"[.,]", s)
        if (len(groups[-1]) == 3 and all(len(g) == 3 for g in groups[1:])
                and 1 <= len(groups[0]) <= 3):
            # A three-digit tail is a thousands group, not cents. No currency
            # prints three decimals, and both corpora produce this shape when
            # a comma is misread as a dot: `8.635,000` on a Korean invoice
            # (measured 2026-10-18) is 8,635,000, and reading it by position
            # would return 8635.000 -- a thousand-fold error, which is the one
            # mistake this module exists to prevent.
            s = "".join(groups)
        else:
            # Both separators present and the tail is short: the LAST one is
            # the decimal mark, the other one sits in the thousands position.
            # Under that rule `1,056.00` (US) and `9.500,00` (thousands dot
            # plus decimal comma, CORD v2 receipt_0067) are the same case.
            if s.rfind(",") > s.rfind("."):
                intpart, dec = s.rsplit(",", 1)
                grp = "."                 # dot is the thousands mark here
            else:
                intpart, dec = s.rsplit(".", 1)
                grp = ","                 # comma is the thousands mark here
            if not re.fullmatch(r"\d{1,3}(?:%s\d{3})+" % re.escape(grp),
                                intpart):
                return None
            if not re.fullmatch(r"\d{1,2}", dec):
                return None
            s = intpart.replace(grp, "") + "." + dec
    elif "," in s:
        parts = s.split(",")
        if all(len(p) == 3 for p in parts[1:]):
            s = "".join(parts)              # thousands separator
        elif len(parts) == 2 and 1 <= len(parts[1]) <= 2:
            s = parts[0] + "." + parts[1]   # decimal comma
        else:
            return None
    elif thousands == ",":
        # Dot-as-thousands, admitted only because the caller said the locale
        # uses a comma. `115.500` on a Japanese invoice is 115,500. The same
        # misread applied to a decimal amount gives `6.124.68` (measured
        # 2026-10-18 on a preprocessed photo fixture): thousand groups plus a
        # two-digit tail, which has exactly one value-preserving reading.
        m = re.fullmatch(r"(\d{1,3}(?:\.\d{3})+)(?:\.(\d{1,2}))?", s)
        if m:
            s = m.group(1).replace(".", "")
            if m.group(2):
                s += "." + m.group(2)
    elif thousands == ".":
        # Dot-as-thousands in a dot-decimal locale. Measured 2026-10-18 on
        # the CORD v2 public fixtures: Indonesian receipts print `60.000`
        # for 60,000, and without this branch the parser reads 60.000 -- a
        # thousand-fold error in the direction that matters. Pure group
        # runs only; the comma-group mirror (`6,124,68`) stays unimplemented
        # because the comma branch above runs first.
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
            s = s.replace(".", "")
    if not re.fullmatch(r"\d+(\.\d+)?", s):
        return None
    try:
        val = Decimal(s)
    except InvalidOperation:
        return None
    return -val if neg else val


def amount_in_text(text: str, thousands: str | None = None) -> Decimal | None:
    """Last (and therefore usually grand-total-most) money token on the page."""
    best = None
    for val in amounts_in_text(text, thousands):
        best = val
    return best


def amounts_in_text(text: str, thousands: str | None = None) -> list[Decimal]:
    """Every money token on the page, in reading order, after repair."""
    out: list[Decimal] = []
    for m in NUM_TOKEN.finditer(repair(text)):
        val = parse_amount(m.group(0), thousands)
        if val is not None:
            out.append(val)
    return out
