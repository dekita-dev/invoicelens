#!/usr/bin/env python
"""Bisect why GPT4All's Kompute backend aborts on this machine.

GGML_ASSERT calls abort(), which kills the whole process. So every
configuration must be tried in its own subprocess. This script is the
driver; the worker is `_worker`.

Each attempt is reported as: config -> outcome. A config "works" if we get
a non-empty completion back.
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(
    os.path.expanduser("~"), ".cache", "invoicelens", "models",
    "gemma-2-2b-it-Q4_K_M.gguf",
)

# (n_ctx, ngl, device, label)
ATTEMPTS = [
    (2048, 100, "cpu", "default ngl=100 (all layers on GPU)"),
    (2048, 0, "cpu", "ngl=0 (zero GPU layers -> CPU compute)"),
    (4096, 0, "cpu", "n_ctx=4096 ngl=0"),
    (8192, 0, "cpu", "n_ctx=8192 ngl=0"),
    (2048, 1, "cpu", "ngl=1 (one GPU layer)"),
    (2048, 100, "kompute", "explicit kompute device"),
]

WORKER = r'''
import sys, os
n_ctx, ngl, device = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
model = sys.argv[4]
try:
    from gpt4all import GPT4All
    m = GPT4All(model_name=os.path.basename(model), model_path=os.path.dirname(model),
                allow_download=False, n_ctx=n_ctx, ngl=ngl, device=device)
    out = m.generate("Reply with just the number: 2+2=", max_tokens=12, temp=0.0)
    print("OK|backend=%s|device=%s|out=%r" % (m.backend, m.device, out))
except BaseException as e:
    print("ERR|%s: %s" % (type(e).__name__, str(e)[:300]))
'''


def main() -> int:
    worker_path = os.path.join(HERE, "_diag_worker.py")
    with open(worker_path, "w", encoding="utf-8") as fh:
        fh.write(WORKER)

    for n_ctx, ngl, device, label in ATTEMPTS:
        print("=" * 70)
        print("TRY n_ctx=%d ngl=%d device=%s  (%s)" % (n_ctx, ngl, device, label))
        sys.stdout.flush()
        proc = subprocess.run(
            [sys.executable, worker_path, str(n_ctx), str(ngl), device, MODEL],
            capture_output=True, text=True, timeout=900,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        tail = "\n".join(err.splitlines()[-4:]) if err else ""
        print("rc=%d" % proc.returncode)
        if out:
            print("stdout: %s" % out[:400])
        if tail:
            print("stderr(tail): %s" % tail[:400])
        sys.stdout.flush()

    try:
        os.remove(worker_path)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
