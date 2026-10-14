#!/usr/bin/env python
"""End-to-end smoke test for the local inference runtime.

This is the gate between "code written" and "code works". It does three
things, in order, and fails loudly on the first one that breaks:

  1. load the GGUF through gpt4all (DLL + model compatibility)
  2. one trivial completion (the runtime actually generates tokens)
  3. one real `extract_fields()` call on a fixture invoice (our prompt and
     JSON repair loop produce parseable output)

Usage:
    python tools/smoke_engine.py [path/to/model.gguf]
"""

from __future__ import annotations

import os
import sys
import time

# Make the package importable when run straight from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from invoicelens.engine import DEFAULT_MODEL_PATH, LocalLLM  # noqa: E402

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fixtures", "design-retainer-2026-09.txt",
)


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    print("model : %s" % model_path)
    print("exists: %s" % os.path.isfile(model_path))
    if not os.path.isfile(model_path):
        print("FAIL: model file missing")
        return 1

    t0 = time.time()
    llm = LocalLLM(model_path=model_path, n_ctx=4096)
    llm.load()
    print("load  : %.1fs" % (time.time() - t0))

    t0 = time.time()
    raw = llm.complete(
        "You are a terse assistant. Answer with one short line.",
        "What is 2 + 2? Reply with just the number.",
        max_tokens=16,
        temp=0.0,
    )
    dt = time.time() - t0
    print("gen   : %.1fs -> %r" % (dt, raw))
    if not raw.strip():
        print("FAIL: empty completion")
        return 1

    # Step 3: the real thing.
    from invoicelens.extract import extract_fields
    from invoicelens.textsource import read_text

    if not os.path.isfile(FIXTURE):
        print("SKIP: fixture not found at %s" % FIXTURE)
        return 0

    text = read_text(FIXTURE)
    print("\n--- fixture: %s (%d chars) ---" % (os.path.basename(FIXTURE), len(text)))
    t0 = time.time()
    result = extract_fields(llm, text)
    dt = time.time() - t0
    print("extract: %.1fs, attempts=%d repaired=%s"
          % (dt, result.attempts, result.repaired))
    print("fields : %s" % result.fields)
    if result.errors:
        print("errors : %s" % result.errors)

    ok = bool(result.fields.get("vendor")) and bool(result.fields.get("total"))
    print("\nRESULT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
