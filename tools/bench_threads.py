#!/usr/bin/env python
"""Find the fastest thread count for this CPU.

Generation on a small model is memory-bandwidth bound, and hyperthreading
often hurts rather than helps. This asks the model for a fixed amount of
output at several thread counts and reports seconds per run.

Note the unit: the reply is one number per line, so the printed figure is
seconds per *output line*, not seconds per token. Per-token rates (split
into prompt processing and generation) come from tools/bench_backends.py.

Each setting runs in its own server, so a bad setting cannot poison the next.

Usage:
    python tools/bench_threads.py [counts...] [--server PATH] [--model PATH]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from invoicelens.engine import (  # noqa: E402
    DEFAULT_MODEL_PATH, EngineError, find_server_exe,
)

WORKER = r'''
import sys, time, json, urllib.request
port, threads = sys.argv[1], sys.argv[2]
payload = {"model":"local","messages":[
    {"role":"user","content":"Count from 1 to 60, one number per line, nothing else."}],
    "max_tokens":200,"temperature":0.0,"stream":False}
req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port,
    data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
t0=time.time()
with urllib.request.urlopen(req, timeout=900) as r:
    b=json.loads(r.read().decode())
dt=time.time()-t0
txt=b["choices"][0]["message"]["content"]
n=len([l for l in txt.splitlines() if l.strip()])
print("OK|%.1fs|~%d lines|%.2f s/line" % (dt, n, dt/max(n,1)))
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("counts", nargs="*", type=int,
                    help="thread counts to try (default: 2 4 6 8)")
    ap.add_argument("--server", default=None,
                    help="llama-server path (default: located via "
                         "engine.find_server_exe())")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH,
                    help="GGUF model path")
    args = ap.parse_args(argv)

    server = args.server
    if server is None:
        try:
            server = find_server_exe()
        except EngineError as exc:
            print(str(exc))
            return 1
    if not os.path.isfile(args.model):
        print("model not found at %s" % args.model)
        return 1

    worker = os.path.join(HERE, "_bench_worker.py")
    with open(worker, "w", encoding="utf-8") as fh:
        fh.write(WORKER)

    counts = args.counts or [2, 4, 6, 8]
    for n in counts:
        import socket
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        log = open(os.path.join(os.environ.get("TEMP", "/tmp"),
                                "il-bench-%d.log" % port), "w")
        proc = subprocess.Popen(
            [server, "--model", MODEL, "--host", "127.0.0.1",
             "--port", str(port), "--ctx-size", "2048",
             "--threads", str(n), "--log-disable"],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            # wait for health
            deadline = time.time() + 240
            ok = False
            while time.time() < deadline and proc.poll() is None:
                try:
                    import urllib.request
                    with urllib.request.urlopen(
                            "http://127.0.0.1:%d/health" % port, timeout=3) as r:
                        if r.status == 200:
                            ok = True
                            break
                except Exception:
                    time.sleep(0.5)
            if not ok:
                print("threads=%-2d SERVER FAILED" % n)
                continue
            out = subprocess.run([sys.executable, worker, str(port)],
                                 capture_output=True, text=True, timeout=900)
            print("threads=%-2d %s" % (n, (out.stdout or "").strip()[:120]))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
    try:
        os.remove(worker)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
