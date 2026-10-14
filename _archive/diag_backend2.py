#!/usr/bin/env python
"""Retry the backend matrix now that the CUDA 11 runtime DLLs are on PATH.

Each configuration runs in its own subprocess because GGML_ASSERT aborts
the whole process.
"""

from __future__ import annotations

import os
import subprocess
import sys

MODEL = os.path.join(
    os.path.expanduser("~"), ".cache", "invoicelens", "models",
    "gemma-2-2b-it-Q4_K_M.gguf",
)
CUDA_BIN = os.path.join(
    os.path.expanduser("~"), ".cache", "invoicelens", "cuda11", "bin",
)

# (n_ctx, ngl, device, label)
ATTEMPTS = [
    (2048, 0, "cuda", "cuda backend, ngl=0 -> pure CPU compute"),
    (2048, 100, "cuda", "cuda backend, ngl=100 -> all layers on GPU (2GB VRAM)"),
    (2048, 12, "cuda", "cuda backend, ngl=12 -> partial offload"),
    (2048, 0, "cpu", "device=cpu (kompute path) - expect the old assert"),
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
    print("ERR|%s: %s" % (type(e).__name__, str(e)[:400]))
'''


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    worker_path = os.path.join(here, "_diag_worker2.py")
    with open(worker_path, "w", encoding="utf-8") as fh:
        fh.write(WORKER)

    env = dict(os.environ)
    env["PATH"] = CUDA_BIN + os.pathsep + env.get("PATH", "")

    for n_ctx, ngl, device, label in ATTEMPTS:
        print("=" * 72)
        print("TRY n_ctx=%d ngl=%d device=%s  (%s)" % (n_ctx, ngl, device, label))
        sys.stdout.flush()
        try:
            proc = subprocess.run(
                [sys.executable, worker_path, str(n_ctx), str(ngl), device, MODEL],
                capture_output=True, text=True, timeout=900, env=env,
            )
        except subprocess.TimeoutExpired:
            print("TIMEOUT after 900s")
            continue
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        print("rc=%d" % proc.returncode)
        if out:
            print("stdout: %s" % out[:500])
        if err:
            tail = "\n".join(err.splitlines()[-5:])
            print("stderr(tail): %s" % tail[:500])
        sys.stdout.flush()

    try:
        os.remove(worker_path)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
