"""Tests for the OCR repair layer.

Every case here is one that actually failed or nearly failed on a real fixture
in fixtures/cjk_synthetic, not an invented edge case.
"""

import os
import sys
import unittest
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from invoicelens.textnorm import (  # noqa: E402
    amount_in_text, amounts_in_text, fold_digits, parse_amount, repair,
    repair_separators,
)


class TestSeparators(unittest.TestCase):
    def test_underscore_read_as_comma(self):
        # Measured: PP-OCR read 316,800 as 316_800 on both ja fixtures.
        self.assertEqual(repair_separators("316_800"), "316,800")

    def test_space_after_separator(self):
        # Measured: 240,000 came back as "240, 000".
        self.assertEqual(repair_separators("240, 000"), "240,000")

    def test_space_after_a_thousands_dot(self):
        # Measured 2026-10-18 on CORD v2 receipt_0062: the same break, but
        # with the dot an Indonesian receipt uses, so the comma-only rule
        # never saw it and the amount stopped parsing.
        self.assertEqual(repair_separators("48. 000"), "48.000")
        # A two-digit tail is a decimal split across a space, not a group,
        # and joining it would change the value by a factor of a thousand.
        self.assertEqual(repair_separators("12. 50"), "12. 50")

    def test_fullwidth_comma(self):
        self.assertEqual(repair_separators("11，600"), "11,600")

    def test_european_decimal_untouched(self):
        # 1.056.000 is a valid de-DE amount; rewriting it would corrupt money.
        self.assertEqual(repair_separators("1.056.000"), "1.056.000")

    def test_clean_input_is_stable(self):
        for s in ("12,296.00", "316,800", "8,000"):
            self.assertEqual(repair_separators(s), s)


class TestDigits(unittest.TestCase):
    def test_letter_between_digits(self):
        self.assertEqual(fold_digits("31O,8O0"), "310,800")

    def test_letter_in_word_survives(self):
        self.assertEqual(fold_digits("B2B"), "B2B")
        self.assertEqual(fold_digits("S1 プラン"), "S1 プラン")

    def test_fullwidth_digits(self):
        self.assertEqual(fold_digits("￥１２，２９６"), "¥12,296")


class TestParseAmount(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(parse_amount("12,296.00"), Decimal("12296.00"))

    def test_currency_prefix_forms(self):
        cases = {
            "¥12,296": "12296",
            "NT$1,500": "1500",
            "₩8,635,000": "8635000",
            "￥12,296": "12296",
            "12,296元": "12296",
        }
        for raw, want in cases.items():
            self.assertEqual(parse_amount(raw), Decimal(want), raw)

    def test_underscore_amount(self):
        self.assertEqual(parse_amount("316_800"), Decimal("316800"))

    def test_accounting_negative(self):
        self.assertEqual(parse_amount("(1,234)"), Decimal("-1234"))
        self.assertEqual(parse_amount("1,234-"), Decimal("-1234"))

    def test_decimal_comma(self):
        self.assertEqual(parse_amount("12,50"), Decimal("12.50"))

    def test_dot_as_thousands_only_with_locale_hint(self):
        # Measured: PaddleOCR read 115,500 as 115.500 on the ja scan fixture.
        self.assertEqual(parse_amount("115.500", thousands=","), Decimal("115500"))
        # Same string, no hint: must not be guessed at all.
        self.assertEqual(parse_amount("115.500"), Decimal("115.500"))
        # And the hint must not turn a real decimal into an integer.
        self.assertEqual(parse_amount("115.50", thousands=","), Decimal("115.50"))

    def test_dot_thousands_with_two_digit_tail(self):
        # Measured 2026-10-18 on a preprocessed photo fixture (zh-Hans
        # hans-software-renewal, total 6,124.68): PP-OCR read "6.124.68" --
        # BOTH commas came back as dots. The gain table caught it as photo-tier
        # money dropping 100% -> 88%; this branch is the fix.
        self.assertEqual(parse_amount("6.124.68", thousands=","),
                         Decimal("6124.68"))
        # No hint: still refused, because "6.124.68" is not a number in any
        # locale without knowing which separator is which.
        self.assertIsNone(parse_amount("6.124.68"))
        # A genuine decimal must survive the hint untouched.
        self.assertEqual(parse_amount("12.50", thousands=","), Decimal("12.50"))
        # Multi-group form keeps working through the same branch.
        self.assertEqual(parse_amount("1.056.000", thousands=","),
                         Decimal("1056000"))

    def test_european_mirror_is_a_documented_gap(self):
        # NOT a measured failure -- a pinned gap. The mirror of the comma
        # locale repair would read "6,124,68" as 6124.68 under
        # thousands="."; the comma branch runs first and the string dies
        # there. No caller passes the European hint today (eval_ocr hints
        # CJK locales only), so this test exists to make the gap explicit:
        # if European locale support ever lands, this is the assertion to
        # flip, and it must be flipped deliberately.
        self.assertIsNone(parse_amount("6,124,68", thousands="."))
        self.assertIsNone(parse_amount("6,124,68"))
        # What the European hint DOES change today: nothing for comma
        # strings beyond the locale-agnostic branches, and plain decimals
        # keep their own reading.
        self.assertEqual(parse_amount("115,500", thousands="."),
                         Decimal("115500"))
        self.assertEqual(parse_amount("12,50", thousands="."), Decimal("12.50"))
        self.assertEqual(parse_amount("115.50", thousands="."), Decimal("115.50"))
        # Mixed separators are now read by position, whichever comes last
        # being the decimal mark. This changed on 2026-10-18: CORD v2
        # receipt_0067 prints `9.500,00` on an Indonesian receipt, so the
        # "refuse the mixed form" rule was throwing away a real total. The
        # hint no longer matters for a shape whose decimal mark is visible.
        self.assertEqual(parse_amount("1.056,00"), Decimal("1056.00"))
        self.assertEqual(parse_amount("1.056,00", thousands="."),
                         Decimal("1056.00"))
        self.assertEqual(parse_amount("9.500,00", thousands="."),
                         Decimal("9500.00"))
        # A three-digit tail is a thousands group, not cents: no currency
        # prints three decimals, and reading `8.635,000` by position would
        # return 8635.000 instead of 8,635,000. Measured on the Korean photo
        # fixture, where the engine turned a comma into a dot.
        self.assertEqual(parse_amount("8.635,000", thousands=","),
                         Decimal("8635000"))
        self.assertEqual(parse_amount("1,234.567"), Decimal("1234567"))
        self.assertEqual(parse_amount("1,234.567", thousands="."),
                         Decimal("1234567"))

    def test_indonesian_dot_thousands(self):
        # Measured 2026-10-18 on the CORD v2 public fixtures (fixtures/public):
        # Indonesian receipts print `60.000` for 60,000. The hint is what makes
        # it unambiguous; the same string with no hint is a plain 60.000
        # decimal, which is exactly why the hint exists.
        self.assertEqual(parse_amount("60.000", thousands="."),
                         Decimal("60000"))
        self.assertEqual(parse_amount("60.000"), Decimal("60.000"))
        self.assertEqual(parse_amount("1.056.000", thousands="."),
                         Decimal("1056000"))
        # A real decimal survives the dot-locale hint untouched.
        self.assertEqual(parse_amount("115.50", thousands="."), Decimal("115.50"))
        self.assertEqual(parse_amount("12.50", thousands="."), Decimal("12.50"))

    def test_garbage_returns_none(self):
        for raw in ("", "合計", "-", "N/A", "12,,296", "abc"):
            self.assertIsNone(parse_amount(raw), raw)
        # Grouping that does not parse under the hint is dropped, not guessed.
        for raw in ("12..34", "1.2.3", "6.,124"):
            self.assertIsNone(parse_amount(raw, thousands=","), raw)


class TestPageLevel(unittest.TestCase):
    def test_repair_is_idempotent(self):
        once = repair("合計 316_800 円")
        self.assertEqual(repair(once), once)

    def test_grand_total_is_last_amount(self):
        page = "小計 288,000\n消費税 28,800\n合計 316_800"
        self.assertEqual(amount_in_text(page), Decimal("316800"))

    def test_quantity_does_not_swallow_amounts(self):
        # Measured ko page: "1 6,500,000 6,500,000". A token regex that allows
        # inner whitespace glues this into one impossible number and the real
        # totals disappear from the page.
        page = "1 6,500,000 6,500,000\n3\n450,000 1,350,000"
        got = amounts_in_text(page)
        self.assertIn(Decimal("6500000"), got)
        self.assertIn(Decimal("1350000"), got)

    def test_adjacent_lines_stay_separate(self):
        page = "營業税（5%）\n900.00\n18,900.00"
        self.assertIn(Decimal("18900.00"), amounts_in_text(page))

    def test_no_amount(self):
        self.assertIsNone(amount_in_text("請求書 番号のみ"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
