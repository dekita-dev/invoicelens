# One-off reproduction of the bench_threads.py empty-output failure.
# Starts the same server bench_threads.py starts, runs its exact WORKER
# source, and prints stdout/stderr/returncode instead of swallowing them.
import os
import socket
import subprocess
import sys
import time
import urllib.request

# script lives at products/invoicelens/_archive/bench-2026-10-15/
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # products/invoicelens
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "bt", os.path.join(ROOT, "tools", "bench_threads.py"))
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)  # top-level only: defines WORKER, main; no run

from invoicelens.engine import DEFAULT_MODEL_PATH, find_server_exe

server = find_server_exe()
worker_src = os.path.join(os.environ.get("TEMP", "/tmp"), "_repro_worker.py")
with open(worker_src, "w", encoding="utf-8") as fh:
    fh.write(bt.WORKER)
print("worker written to", worker_src)

s = socket.socket()
s.bind(("127.0.0.1", 0))
port = s.getsockname()[1]
s.close()
log = open(os.path.join(os.environ.get("TEMP", "/tmp"),
                        "_repro-bench-%d.log" % port), "w")
proc = subprocess.Popen(
    [server, "--model", DEFAULT_MODEL_PATH, "--host", "127.0.0.1",
     "--port", str(port), "--ctx-size", "2048",
     "--threads", "2", "--log-disable"],
    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
print("server pid", proc.pid, "port", port)
try:
    deadline = time.time() + 240
    ok = False
    while time.time() < deadline and proc.poll() is None:
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:%d/health" % port, timeout=3) as r:
                if r.status == 200:
                    ok = True
                    break
        except Exception:
            time.sleep(0.5)
    print("health ok:", ok)
    if ok:
        out = subprocess.run([sys.executable, worker_src, str(port), "2"],
                             capture_output=True, text=True, timeout=900)
        print("=== worker rc:", out.returncode)
        print("=== worker stdout:")
        print(out.stdout)
        print("=== worker stderr:")
        print(out.stderr)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
    try:
        os.remove(worker_src)
    except OSError:
        pass
print("repro done")
