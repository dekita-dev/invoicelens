"""Measure the real token cost of one InvoiceLens extraction.

Why this exists: the README and the dev.to article quote an API-cost
comparison, and a comparison built on a guessed token count is worthless.
llama-server returns a standard OpenAI-style `usage` block on every
completion; this script reads it (`engine.LocalLLM.last_usage`) instead of
estimating.

Run it with the runtime and model already in place:

    python tools/measure_tokens.py [fixtures ...]

Prints one line per file (prompt tokens, completion tokens, wall seconds)
and a total. No report files are written; this is a measurement, not a run.
"""

from __future__ import annotations

import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from invoicelens.engine import LocalLLM, _default_threads  # noqa: E402
from invoicelens.extract import extract_fields  # noqa: E402
from invoicelens.textsource import read_text  # noqa: E402


def _usage_totals(llm):
    """(prompt, completion) tokens from the last completion, or (0, 0)."""
    u = llm.last_usage or {}
    return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def main(argv):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if argv:
        files = argv
    else:
        files = sorted(glob.glob(os.path.join(here, "fixtures", "*")))
    if not files:
        print("no fixtures found")
        return 1

    print("threads: %d (physical cores detected)" % _default_threads())
    llm = LocalLLM()
    t0 = time.time()
    llm.load()
    print("model load: %.1fs" % (time.time() - t0))
    print("")
    print("%-34s %8s %8s %8s %9s" % ("file", "prompt", "compl", "total", "seconds"))
    print("-" * 74)

    tot_p = tot_c = 0
    for path in files:
        name = os.path.basename(path)
        try:
            text = read_text(path)
        except Exception as exc:  # noqa: BLE001
            print("%-34s  SKIP (%s)" % (name, exc))
            continue
        t1 = time.time()
        res = extract_fields(llm, text)
        dt = time.time() - t1
        # extract_fields may make more than one call; last_usage holds the
        # final one, so report the attempt count alongside it and let the
        # reader judge.
        p, c = _usage_totals(llm)
        tot_p += p
        tot_c += c
        print("%-34s %8d %8d %8d %9.1f   attempts=%d" % (
            name, p, c, p + c, dt, res.attempts))

    print("-" * 74)
    print("%-34s %8d %8d %8d" % ("TOTAL (server-reported)", tot_p, tot_c,
                                 tot_p + tot_c))
    if llm.last_usage is None:
        print("")
        print("NOTE: server reported no usage block; totals above are zero.")
    llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
