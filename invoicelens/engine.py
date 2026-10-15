"""Local inference runtime, backed by llama.cpp's own HTTP server.

Why this and not gpt4all / llama-cpp-python:

  * llama-cpp-python publishes no Windows wheels for any Python version, and
    this machine has no C++ compiler, so it cannot be built here.
  * gpt4all does ship a Windows wheel, but (measured on this machine, see
    README "Runtime notes") the llama.cpp it bundles is too old for GGUF v3
    files and aborts inside `llama_model_load` on a current Gemma 2 GGUF.
    Its Windows wheel also has no CPU backend at all -- only CUDA and
    Kompute/Vulkan -- so "cpu" is not an option there.

So we drive `llama-server.exe` from the official llama.cpp release as a
subprocess and talk to it over HTTP. That keeps every matmul on this machine,
needs no compiler, and uses a llama.cpp that is current rather than pinned.

The tradeoff is one extra moving part: a subprocess to start, health-check
and stop. That is contained in `load()` and `close()`.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


DEFAULT_MODEL_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "invoicelens", "models",
    "gemma-2-2b-it-Q4_K_M.gguf",
)

# Where the llama.cpp release zip is unpacked, unless overridden.
LLAMA_DIR_CANDIDATES = [
    os.environ.get("INVOICELENS_LLAMA_DIR"),
    os.path.join(os.path.expanduser("~"), ".cache", "invoicelens", "llama.cpp"),
    os.path.join(os.path.expanduser("~"), ".cache", "invoicelens", "llama.cpp",
                 "build", "bin"),
]

# Same idea for the Vulkan build, which is a separate download that unpacks
# to its own directory. A separate list means a GPU-less machine is never
# sent looking for it, and the two builds cannot overwrite each other.
LLAMA_DIR_CANDIDATES_VULKAN = [
    os.environ.get("INVOICELENS_LLAMA_DIR_VULKAN"),
    os.path.join(os.path.expanduser("~"), ".cache", "invoicelens",
                 "llama.cpp-vulkan"),
    os.path.join(os.path.expanduser("~"), ".cache", "invoicelens",
                 "llama.cpp-vulkan", "build", "bin"),
]

SERVER_EXE = "llama-server.exe" if sys.platform == "win32" else "llama-server"

# The Vulkan build also ships ggml-cpu DLLs, so "a CPU backend exists in
# this directory" proves nothing about GPU support. The backend library
# itself is the only honest evidence.
VULKAN_BACKEND_LIBS = ("ggml-vulkan.dll", "libggml-vulkan.so",
                       "libggml-vulkan.dylib")


class EngineError(Exception):
    """Raised when the local model cannot be loaded or run."""


def _has_vulkan_backend(directory):
    """True when *directory* really contains a Vulkan backend library."""
    return any(os.path.isfile(os.path.join(directory, lib))
               for lib in VULKAN_BACKEND_LIBS)


def find_server_exe(gpu=False):
    """Locate llama-server, honouring INVOICELENS_LLAMA_DIR then PATH.

    With gpu=True, only builds that actually contain the Vulkan backend
    count, and the Vulkan candidate list is searched first. The error then
    points at the fetch command, because that is the fix; the CPU-only
    message is left exactly as it was.
    """
    if gpu:
        candidates = LLAMA_DIR_CANDIDATES_VULKAN + list(LLAMA_DIR_CANDIDATES)
    else:
        candidates = list(LLAMA_DIR_CANDIDATES)
    for cand in candidates:
        if not cand:
            continue
        path = os.path.join(cand, SERVER_EXE)
        if os.path.isfile(path) and (not gpu or _has_vulkan_backend(cand)):
            return path
    found = shutil.which(SERVER_EXE)
    if found and (not gpu or _has_vulkan_backend(os.path.dirname(found))):
        return found
    if gpu:
        raise EngineError(
            "no llama-server with a Vulkan backend was found.\n"
            "GPU offload needs the Vulkan build of llama.cpp, which is a\n"
            "separate download from the CPU build:\n"
            "  python tools/fetch_llama_cpp.py --flavor vulkan\n"
            "or point INVOICELENS_LLAMA_DIR_VULKAN at an unpacked one.\n"
            "CPU-only inference keeps working without any of this."
        )
    raise EngineError(
        "%s not found.\n"
        "Download the llama.cpp release zip for your platform from\n"
        "  https://github.com/ggml-org/llama.cpp/releases\n"
        "(pick llama-b*-bin-win-cpu-x64.zip on Windows), unzip it, and either\n"
        "put it on PATH or point INVOICELENS_LLAMA_DIR at the directory.\n"
        "See README, 'Getting the runtime'."
        % SERVER_EXE
    )


def _build_server_cmd(server_exe, model_path, port, n_ctx, n_threads,
                      n_gpu_layers=0, gpu_device=None):
    """argv for llama-server.

    n_gpu_layers=0 reproduces the CPU-only command byte for byte (a test
    pins that, so the default path cannot drift). Above zero, --n-gpu-layers
    and an optional --device are appended and --log-disable is dropped:
    when a GPU is involved the log is the only way to see which backend and
    device were actually picked.
    """
    cmd = [
        server_exe,
        "--model", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", str(n_ctx),
        "--threads", str(n_threads),
    ]
    if n_gpu_layers and n_gpu_layers > 0:
        cmd += ["--n-gpu-layers", str(n_gpu_layers)]
        if gpu_device:
            cmd += ["--device", gpu_device]
    else:
        cmd += ["--log-disable"]
    return cmd


def _env_int(name, fallback):
    """int from the environment, tolerating junk: garbage becomes 0.

    A malformed INVOICELENS_N_GPU_LAYERS must fail safe to CPU-only, never
    to a crash and never to a guess.
    """
    raw = os.environ.get(name)
    if raw is None:
        return fallback
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return 0


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _default_threads():
    """Physical core count, not logical.

    Generation with a small quantized model is memory-bandwidth bound, so
    hyperthreads cost more than they return. Measured on an i5-8350U (4 cores
    / 8 threads) with Gemma 2 2B Q4_K_M, generating 60 tokens:

        2 threads  0.846 s/token
        4 threads  0.893 s/token
        8 threads  1.423 s/token   <- 68% slower than 2

    So: physical cores, and never more than 8 (past that the scheduler is
    the bottleneck, not the model).
    """
    cores = None
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class SYSTEM_LOGICAL_PROCESSOR_INFORMATION(ctypes.Structure):
                _fields_ = [("ProcessorMask", ctypes.c_ulonglong),
                            ("Relationship", ctypes.c_int),
                            ("Reserved", ctypes.c_ulonglong * 2)]

            needed = wintypes.DWORD()
            ctypes.windll.kernel32.GetLogicalProcessorInformation(
                None, ctypes.byref(needed))
            buf = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *
                   ((needed.value + ctypes.sizeof(
                       SYSTEM_LOGICAL_PROCESSOR_INFORMATION) - 1) //
                    ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)))()
            ok = ctypes.windll.kernel32.GetLogicalProcessorInformation(
                buf, ctypes.byref(needed))
            if ok:
                # RelationProcessorCore == 0
                cores = sum(1 for e in buf
                            if e.Relationship == 0 and e.ProcessorMask)
        except Exception:  # noqa: BLE001 - fall through to the generic path
            cores = None
    elif sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"], text=True).strip()
            cores = int(out)
        except Exception:  # noqa: BLE001
            cores = None
    else:
        try:
            seen = set()
            phys = core = None
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("physical id"):
                        phys = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":", 1)[1].strip()
                        if phys is not None:
                            seen.add((phys, core))
            cores = len(seen) or None
        except Exception:  # noqa: BLE001
            cores = None

    if not cores:
        cores = os.cpu_count() or 2
    return max(1, min(cores, 8))


class LocalLLM:
    """A local open-weight chat model. No network, no API key.

    Usage:
        with LocalLLM() as llm:
            llm.complete(system, user)
    or call load()/close() yourself.
    """

    def __init__(self, model_path=None, n_ctx=4096, n_threads=None,
                 startup_timeout=300.0, server_exe=None,
                 n_gpu_layers=None, gpu_device=None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        if not os.path.isfile(self.model_path):
            raise EngineError(
                "model file not found: %s\n"
                "Download it first (see README, 'Getting the model')."
                % self.model_path
            )
        self.n_ctx = n_ctx
        self.n_threads = n_threads or _default_threads()
        self.startup_timeout = startup_timeout
        # GPU offload is opt-in and off by default. A requested layer count
        # also decides which build we look for: a CPU-only llama-server
        # accepts --n-gpu-layers and then quietly runs on CPU anyway, and a
        # silent fallback is the one outcome worth failing loudly over.
        self.n_gpu_layers = _env_int("INVOICELENS_N_GPU_LAYERS", n_gpu_layers)
        self.gpu_device = os.environ.get("INVOICELENS_DEVICE") or gpu_device
        if server_exe is None:
            server_exe = find_server_exe(
                gpu=bool(self.n_gpu_layers and self.n_gpu_layers > 0))
        self._server_exe = server_exe
        self._proc = None
        self._port = None
        self._log = None
        self._log_path = None
        # Token usage of the most recent completion, as reported by the
        # server. Useful for anyone costing a local run against a hosted API;
        # None until the first completion returns.
        self.last_usage = None

    # -- lifecycle ------------------------------------------------------

    def load(self):
        """Start the server and wait until it answers /health."""
        if self._proc is not None:
            return self

        self._port = _free_port()
        self._log_path = os.path.join(
            os.environ.get("TEMP") or "/tmp", "invoicelens-server-%d.log" % self._port
        )
        self._log = open(self._log_path, "w", encoding="utf-8", errors="replace")

        cmd = _build_server_cmd(
            self._server_exe, self.model_path, self._port,
            self.n_ctx, self.n_threads,
            n_gpu_layers=self.n_gpu_layers or 0,
            gpu_device=self.gpu_device,
        )
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=self._log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise EngineError("could not start %s: %s" % (self._server_exe, exc))

        atexit.register(self.close)
        self._wait_healthy()
        return self

    def _wait_healthy(self):
        deadline = time.time() + self.startup_timeout
        url = "http://127.0.0.1:%d/health" % self._port
        last = ""
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise EngineError(
                    "%s exited with code %s during startup.\nServer log: %s\n%s"
                    % (self._server_exe, self._proc.returncode,
                       self._log_path, self._tail_log())
                )
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except urllib.error.URLError as exc:
                last = str(exc)
            except Exception as exc:  # noqa: BLE001 - keep polling
                last = str(exc)
            time.sleep(0.5)
        raise EngineError(
            "%s did not become healthy within %.0fs (%s).\nServer log: %s\n%s"
            % (self._server_exe, self.startup_timeout, last,
               self._log_path, self._tail_log())
        )

    def _tail_log(self, n=20):
        try:
            with open(self._log_path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
            return "\n".join(lines[-n:])
        except OSError:
            return "(log unavailable)"

    def close(self):
        """Stop the server. Safe to call more than once."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
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
        try:
            atexit.unregister(self.close)
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        return self.load()

    def __exit__(self, *exc):
        self.close()
        return False

    # -- inference ------------------------------------------------------

    def complete(self, system, user, max_tokens=512, temp=0.0):
        """One-shot chat completion. Returns the assistant text."""
        if self._proc is None:
            self.load()

        payload = {
            "model": "local",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temp,
            "stream": False,
        }
        req = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % self._port,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise EngineError("server returned HTTP %s: %s" % (exc.code, detail))
        except urllib.error.URLError as exc:
            raise EngineError("could not reach local server: %s" % exc)

        # Real token counts, straight from the server. Kept on the instance so
        # a caller can report them; never used to change model behaviour.
        self.last_usage = body.get("usage")

        try:
            msg = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EngineError("unexpected server response: %s" % str(body)[:400])

        text = (msg.get("content") or "").strip()
        if not text and msg.get("reasoning_content"):
            # Some models split reasoning from the answer; prefer the answer,
            # but never silently return nothing.
            text = (msg["reasoning_content"] or "").strip()
        return text
