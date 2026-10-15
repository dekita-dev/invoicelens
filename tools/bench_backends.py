#!/usr/bin/env python
"""Benchmark llama.cpp backends phase by phase: prompt vs generation.

bench_threads.py answers "how many CPU threads". This one answers a
different question, because a local-first triage tool spends most of its
tokens on the prompt (invoice text is long, the JSON reply is short): what
does a backend cost per phase? One blended tok/s number hides exactly the
split where a GPU helps or does not.

Every rate comes from the server's own `timings` block; wall clock divided
by the usage block is only the fallback, and the output says which one
produced each number.

Usage:
    python tools/bench_backends.py --list-devices
    python tools/bench_backends.py --build cpu --threads 2 4
    python tools/bench_backends.py --build vulkan --device vulkan0 --ngl 0 26
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from invoicelens import engine  # noqa: E402
from invoicelens.engine import (  # noqa: E402
    DEFAULT_MODEL_PATH, EngineError, _build_server_cmd, find_server_exe,
)
from invoicelens.extract import SYSTEM_PROMPT, build_prompt  # noqa: E402
from invoicelens.textsource import UnsupportedSource, read_text  # noqa: E402

GEN_SYSTEM = "You are a terse assistant."
GEN_PROMPT = "Count from 1 to 60, one number per line, nothing else."
GEN_MAX_TOKENS = 200
PROMPT_MAX_TOKENS = 1


def _first_fixture():
    cands = sorted(glob.glob(os.path.join(ROOT, "fixtures", "*.txt")))
    return cands[0] if cands else None


def _invoice_prompt(prompt_file=None):
    """The real extraction prompt for a real invoice (see extract.py)."""
    path = prompt_file or _first_fixture()
    if not path:
        return GEN_PROMPT
    try:
        text = read_text(path)
    except UnsupportedSource as exc:
        print("prompt file unreadable (%s); using the counting prompt" % exc)
        return GEN_PROMPT
    return build_prompt(text)


class BenchServer:
    """One llama-server process for one configuration."""

    def __init__(self, build, model, ctx, threads, ngl, device,
                 server_exe=None):
        self.build = build
        if server_exe is None:
            try:
                server_exe = find_server_exe(gpu=build == "vulkan")
            except EngineError as exc:
                raise SystemExit(str(exc))
        self.server_exe = server_exe
        self.port = engine._free_port()
        self.log_path = os.path.join(
            os.environ.get("TEMP") or "/tmp", "il-bench-%d.log" % self.port)
        self.cmd = _build_server_cmd(
            server_exe, model, self.port, ctx, threads,
            n_gpu_layers=ngl, gpu_device=device)
        self.proc = None
        self._log = None
        self.cold_start = None

    def start(self, timeout=300.0):
        self._log = open(self.log_path, "w", encoding="utf-8",
                         errors="replace")
        t0 = time.time()
        self.proc = subprocess.Popen(
            self.cmd, stdout=self._log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL)
        url = "http://127.0.0.1:%d/health" % self.port
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise SystemExit(
                    "server exited with code %s during startup.\n"
                    "command: %s\nlog: %s\n%s"
                    % (self.proc.returncode, " ".join(self.cmd),
                       self.log_path, self._tail()))
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        self.cold_start = time.time() - t0
                        return self.cold_start
            except Exception:  # noqa: BLE001 - keep polling
                time.sleep(0.5)
        raise SystemExit(
            "server did not become healthy within %.0fs.\nlog: %s\n%s"
            % (timeout, self.log_path, self._tail()))

    def _tail(self, n=25):
        try:
            with open(self.log_path, "r", encoding="utf-8",
                      errors="replace") as fh:
                return "\n".join(fh.read().splitlines()[-n:])
        except OSError:
            return "(log unavailable)"

    def stop(self):
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None

    def complete(self, system, user, max_tokens):
        payload = {
            "model": "local",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        req = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % self.port,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body, time.time() - t0


def _rates(body, dt):
    """(prompt tok/s, generation tok/s, source).

    The server's own timings win; wall clock over the usage block is the
    fallback. The source string travels with the number so a report can
    never quietly mix the two.
    """
    usage = body.get("usage") or {}
    timings = body.get("timings") or {}
    p_tok = usage.get("prompt_tokens")
    c_tok = usage.get("completion_tokens")
    p_rate = g_rate = None
    src = "wall clock / usage"
    p = timings.get("prompt_per_second")
    g = timings.get("predicted_per_second")
    if isinstance(p, (int, float)) and p > 0:
        p_rate = float(p)
    if isinstance(g, (int, float)) and g > 0:
        g_rate = float(g)
    if p_rate is not None or g_rate is not None:
        src = "server timings"
    if p_rate is None and isinstance(p_tok, int) and p_tok > 0 and dt > 0:
        p_rate = p_tok / dt
    if g_rate is None and isinstance(c_tok, int) and c_tok > 0 and dt > 0:
        g_rate = c_tok / dt
    return p_rate, g_rate, src


def _median(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[len(vals) // 2]


def _probe(server, invoice_prompt, repeat):
    """Median prompt tok/s and generation tok/s over *repeat* runs."""
    p_rates, g_rates, srcs = [], [], []
    for _ in range(max(1, repeat)):
        body, dt = server.complete(SYSTEM_PROMPT, invoice_prompt,
                                   PROMPT_MAX_TOKENS)
        p, _g, src = _rates(body, dt)
        p_rates.append(p)
        srcs.append(src)
        body, dt = server.complete(GEN_SYSTEM, GEN_PROMPT, GEN_MAX_TOKENS)
        _p, g, src = _rates(body, dt)
        g_rates.append(g)
        srcs.append(src)
    return _median(p_rates), _median(g_rates), srcs[0] if srcs else "?"


def _list_devices(build, server_override, model):
    try:
        server_exe = server_override or find_server_exe(gpu=build == "vulkan")
    except EngineError as exc:
        print(str(exc))
        return 1

    bench = os.path.join(
        os.path.dirname(server_exe),
        "llama-bench.exe" if sys.platform == "win32" else "llama-bench")
    if os.path.isfile(bench):
        # flush first: the child writes straight to the fd, and without a
        # flush its output can overtake this header in the pipe.
        print("$ %s --list-devices" % bench, flush=True)
        rc = subprocess.call([bench, "--list-devices"])
        if rc == 0:
            return 0
        print("(llama-bench exited %d; falling back to the server log)" % rc)
    else:
        print("(no llama-bench next to %s; falling back to the server log)"
              % server_exe)

    if not os.path.isfile(model):
        print("cannot start a server to enumerate devices: "
              "model not found at %s" % model)
        return 1

    # ngl=1 keeps --log-disable off (see _build_server_cmd), so the log can
    # show which backend and device the build actually picked up.
    srv = BenchServer(build, model, 512, 2, 1, None, server_exe=server_exe)
    try:
        srv.start()
    except SystemExit as exc:
        print(exc)
        return 1
    finally:
        srv.stop()
    log = srv._tail(80)
    device_lines = [ln for ln in log.splitlines()
                    if "vulkan" in ln.lower() or "device" in ln.lower()]
    if device_lines:
        print("device lines from the startup log:")
        for ln in device_lines:
            print("  " + ln.strip())
        return 0
    print("no device lines in the startup log; full tail follows.")
    print(log)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-devices", action="store_true",
                    help="enumerate Vulkan devices and exit")
    ap.add_argument("--build", choices=("cpu", "vulkan"), default="cpu",
                    help="which llama.cpp build to benchmark (default: cpu)")
    ap.add_argument("--device", default=None,
                    help="Vulkan device name, e.g. vulkan0 "
                         "(Vulkan build only)")
    ap.add_argument("--ngl", type=int, nargs="+", default=[0],
                    help="GPU layer counts to sweep (default: 0 = no offload)")
    ap.add_argument("--threads", type=int, nargs="+", default=None,
                    help="CPU thread counts to sweep (default: the engine "
                         "default, physical cores capped at 8)")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH,
                    help="GGUF model path")
    ap.add_argument("--server", default=None, help="llama-server path override")
    ap.add_argument("--ctx", type=int, default=4096,
                    help="context size (default: 4096)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="runs per configuration, median reported "
                         "(default: 1)")
    ap.add_argument("--prompt-file", default=None,
                    help="invoice text for the prompt probe (default: first "
                         "fixtures/*.txt)")
    args = ap.parse_args(argv)

    if args.list_devices:
        return _list_devices(args.build, args.server, args.model)

    if not os.path.isfile(args.model):
        print("model not found at %s" % args.model)
        return 1

    invoice_prompt = _invoice_prompt(args.prompt_file)
    if invoice_prompt == GEN_PROMPT:
        prompt_kind = "counting prompt (no fixture available)"
    elif args.prompt_file:
        prompt_kind = os.path.basename(args.prompt_file)
    else:
        prompt_kind = "first fixtures/*.txt"

    print("model : %s" % args.model)
    print("build : %s%s" % (
        args.build,
        " (device %s)" % args.device if args.device else
        " (no --device: llama.cpp will use device 0)" if args.build == "vulkan"
        else ""))
    print("prompt: %s, max_tokens=%d" % (prompt_kind, PROMPT_MAX_TOKENS))
    print("gen   : counting prompt, max_tokens=%d" % GEN_MAX_TOKENS)
    print("")

    header = ("%-8s %-9s %4s %7s  %13s  %13s  %10s  %s"
              % ("build", "device", "ngl", "thr", "prompt tok/s",
                 "gen tok/s", "cold s", "timing source"))
    print(header)
    print("-" * len(header))

    for ngl in args.ngl:
        for threads in args.threads or [None]:
            thr = threads if threads else engine._default_threads()
            srv = BenchServer(args.build, args.model, args.ctx, thr, ngl,
                              args.device, server_exe=args.server)
            try:
                srv.start()
                p, g, src = _probe(srv, invoice_prompt, args.repeat)
            finally:
                srv.stop()
            print("%-8s %-9s %4d %7d  %13s  %13s  %10s  %s" % (
                args.build, args.device or "-", ngl, thr,
                "%.1f" % p if p else "n/a",
                "%.2f" % g if g else "n/a",
                "%.1f" % srv.cold_start if srv.cold_start else "n/a",
                src))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
