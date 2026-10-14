"""End-to-end CLI test with a stub model. Never loads a real GGUF."""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from invoicelens.cli import build_parser  # noqa: E402
from invoicelens.cli import main  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "fixtures")
RULES = os.path.join(ROOT, "rules", "rules.example.yaml")


class ScriptedLLM:
    """Canned replies keyed by a marker in the prompt.

    Satisfies the same interface as LocalLLM, including load(), because
    cmd_scan() chains LocalLLM(...).load().
    """

    def load(self):
        return self

    def complete(self, system, user, max_tokens=512, temp=0.0):
        if "NORTHWIND STUDIO" in user:
            return json.dumps({
                "vendor": "Northwind Studio",
                "invoice_number": "NS-2026-0917",
                "invoice_date": "2026-09-03",
                "due_date": "2026-10-03",
                "currency": "USD",
                "subtotal": 2400.0,
                "tax": 0.0,
                "total": 2400.0,
                "line_items": [
                    {"description": "Brand identity retainer, Sept",
                     "quantity": 1, "unit_price": 1800.0, "amount": 1800.0},
                    {"description": "Landing page design, phase 2",
                     "quantity": 1, "unit_price": 600.0, "amount": 600.0},
                ],
                "payment_method": "bank transfer",
                "notes": "net 30",
            })
        if "GITHUB, INC." in user:
            return json.dumps({
                "vendor": "GitHub, Inc.",
                "invoice_number": "GH-7714-2026",
                "invoice_date": "2026-10-02",
                "due_date": None,
                "currency": "USD",
                "subtotal": 4.0,
                "tax": 0.0,
                "total": 4.0,
                "line_items": [
                    {"description": "GitHub Team subscription, October 2026",
                     "quantity": 1, "unit_price": 4.0, "amount": 4.0}],
                "payment_method": "Visa ending 4242",
                "notes": "recurring monthly subscription",
            })
        return "not json at all"


def _run_scan(tmp_path, llm):
    out = str(tmp_path / "report")
    argv = ["scan", FIXTURES, "--out", out, "--rules", RULES]
    rc = main(argv)
    return rc, out


def test_scan_end_to_end(tmp_path, monkeypatch, capsys):
    import invoicelens.cli as cli_mod

    llm = ScriptedLLM()
    # inject the stub instead of building the real engine
    monkeypatch.setattr(cli_mod, "LocalLLM", lambda **kw: llm)
    rc, out = _run_scan(tmp_path, llm)
    assert rc == 0

    for name in ("invoices.csv", "invoices.json", "summary.md"):
        assert os.path.isfile(os.path.join(out, name)), name

    data = json.load(io.open(os.path.join(out, "invoices.json"),
                             encoding="utf-8"))
    by_file = {os.path.basename(d["file"]): d for d in data}
    assert by_file["design-retainer-2026-09.txt"]["fields"]["total"] == 2400.0
    assert by_file["design-retainer-2026-09.txt"]["category"] == \
        "professional_services"
    assert by_file["saas-subscription-2026-10.txt"]["category"] == \
        "software_subscriptions"

    captured = capsys.readouterr().out
    assert "InvoiceLens" in captured
    assert "done in" in captured


def test_scan_empty_dir_raises(tmp_path):
    parser = build_parser()
    args = parser.parse_args(["scan", str(tmp_path)])
    import pytest
    with pytest.raises(SystemExit):
        from invoicelens.cli import cmd_scan
        cmd_scan(args)
