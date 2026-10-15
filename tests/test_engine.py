"""Engine internals with stub builds: no model file, no llama-server.

These pin the two behaviours that must never drift:

  * the default (CPU-only) server argv is byte-for-byte what it was before
    GPU offload existed;
  * a requested GPU layer count can only ever select a build that really
    contains the Vulkan backend -- the library itself, not a directory
    name, because the Vulkan build also ships ggml-cpu DLLs.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from invoicelens import engine  # noqa: E402
from invoicelens.engine import (  # noqa: E402
    SERVER_EXE, EngineError, LocalLLM, _build_server_cmd, find_server_exe,
)


def _fake_build(tmp_path, name, vulkan=False):
    """A directory that looks like an unpacked llama.cpp release."""
    d = tmp_path / name
    d.mkdir()
    (d / SERVER_EXE).write_text("stub", encoding="utf-8")
    lib = "ggml-vulkan.dll" if vulkan else "ggml-cpu.dll"
    (d / lib).write_text("stub", encoding="utf-8")
    return str(d)


@pytest.fixture
def stub_candidates(tmp_path, monkeypatch):
    """Two builds on disk: one CPU-only, one with a real Vulkan backend."""
    cpu = _fake_build(tmp_path, "cpu")
    vulkan = _fake_build(tmp_path, "vulkan", vulkan=True)
    monkeypatch.setattr(engine, "LLAMA_DIR_CANDIDATES", [cpu])
    monkeypatch.setattr(engine, "LLAMA_DIR_CANDIDATES_VULKAN", [vulkan])
    monkeypatch.setattr(engine.shutil, "which", lambda name: None)
    return cpu, vulkan


def _model(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_text("stub", encoding="utf-8")
    return str(path)


def test_find_server_exe_cpu_and_gpu(stub_candidates):
    cpu, vulkan = stub_candidates
    assert find_server_exe() == os.path.join(cpu, SERVER_EXE)
    assert find_server_exe(gpu=True) == os.path.join(vulkan, SERVER_EXE)


def test_gpu_lookup_needs_the_backend_library(tmp_path, monkeypatch):
    # A directory *named* like the Vulkan build but without the backend
    # library must not count. This is the whole point of checking the DLL.
    fake = _fake_build(tmp_path, "llama.cpp-vulkan", vulkan=False)
    monkeypatch.setattr(engine, "LLAMA_DIR_CANDIDATES_VULKAN", [fake])
    monkeypatch.setattr(engine, "LLAMA_DIR_CANDIDATES", [fake])
    monkeypatch.setattr(engine.shutil, "which", lambda name: None)
    with pytest.raises(EngineError) as excinfo:
        find_server_exe(gpu=True)
    msg = str(excinfo.value)
    assert "vulkan" in msg.lower()
    assert "--flavor" in msg


def test_build_server_cmd_default_argv_is_pinned():
    # The CPU-only command, byte for byte. If this test needs changing,
    # the default path changed, and that needs a conscious decision.
    cmd = _build_server_cmd("llama-server", "m.gguf", 8080, 4096, 4)
    assert cmd == [
        "llama-server", "--model", "m.gguf", "--host", "127.0.0.1",
        "--port", "8080", "--ctx-size", "4096", "--threads", "4",
        "--log-disable",
    ]


def test_build_server_cmd_gpu_shapes():
    cmd = _build_server_cmd("llama-server", "m.gguf", 8080, 4096, 4,
                            n_gpu_layers=26, gpu_device="vulkan0")
    assert cmd[-4:] == ["--n-gpu-layers", "26", "--device", "vulkan0"]
    # No --log-disable: with a GPU in play the log is the only way to see
    # which backend and device were actually picked.
    assert "--log-disable" not in cmd

    cmd = _build_server_cmd("llama-server", "m.gguf", 8080, 4096, 4,
                            n_gpu_layers=26)
    assert cmd[-2:] == ["--n-gpu-layers", "26"]
    assert "--device" not in cmd


def test_env_fallback_for_gpu_options(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "find_server_exe",
                        lambda gpu=False: "stub-server")
    model = _model(tmp_path)

    monkeypatch.setenv("INVOICELENS_N_GPU_LAYERS", "26")
    assert LocalLLM(model_path=model).n_gpu_layers == 26

    monkeypatch.setenv("INVOICELENS_N_GPU_LAYERS", "not-a-number")
    assert LocalLLM(model_path=model).n_gpu_layers == 0

    monkeypatch.delenv("INVOICELENS_N_GPU_LAYERS")
    assert LocalLLM(model_path=model).n_gpu_layers is None

    monkeypatch.setenv("INVOICELENS_DEVICE", "vulkan1")
    assert LocalLLM(model_path=model).gpu_device == "vulkan1"
    monkeypatch.delenv("INVOICELENS_DEVICE")
    assert LocalLLM(model_path=model,
                    gpu_device="vulkan0").gpu_device == "vulkan0"


def test_gpu_request_selects_the_vulkan_build(tmp_path, stub_candidates,
                                              monkeypatch):
    cpu, vulkan = stub_candidates
    model = _model(tmp_path)

    # CPU default: the CPU build, whatever else is on disk.
    assert LocalLLM(model_path=model)._server_exe == \
        os.path.join(cpu, SERVER_EXE)

    # A layer count switches the lookup, so a CPU-only build can never
    # silently accept --n-gpu-layers and then run on CPU anyway.
    llm = LocalLLM(model_path=model, n_gpu_layers=26)
    assert llm._server_exe == os.path.join(vulkan, SERVER_EXE)
