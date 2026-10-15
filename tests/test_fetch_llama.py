"""Asset-picking regression tests for tools/fetch_llama_cpp.py.

The upstream asset naming is inconsistent (Windows puts the flavor in the
name, Linux only for non-default flavors; Windows ships .zip while Linux
and macOS ship .tar.gz). These tests pin the mapping against a trimmed
copy of the real b11645 asset list, decoys included, so a future rename
upstream fails loudly here instead of in a user's download.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import fetch_llama_cpp as fetch  # noqa: E402

# Names copied from the real b11645 release, with decoys that must NOT
# match (cuda/sycl/openvino/rocm variants share the platform prefix).
ASSETS = [
    {"name": "cudart-llama-b11645-bin-ubuntu-cuda-12.8-x64.tar.gz",
     "browser_download_url": "u1", "size": 1},
    {"name": "llama-b11645-bin-linux-arm64-snapdragon.tar.gz",
     "browser_download_url": "u2", "size": 2},
    {"name": "llama-b11645-bin-macos-arm64.tar.gz",
     "browser_download_url": "u3", "size": 3},
    {"name": "llama-b11645-bin-macos-x64.tar.gz",
     "browser_download_url": "u4", "size": 4},
    {"name": "llama-b11645-bin-ubuntu-arm64.tar.gz",
     "browser_download_url": "u5", "size": 5},
    {"name": "llama-b11645-bin-ubuntu-sycl-fp16-x64.tar.gz",
     "browser_download_url": "u6", "size": 6},
    {"name": "llama-b11645-bin-ubuntu-vulkan-arm64.tar.gz",
     "browser_download_url": "u7", "size": 7},
    {"name": "llama-b11645-bin-ubuntu-vulkan-x64.tar.gz",
     "browser_download_url": "u8", "size": 8},
    {"name": "llama-b11645-bin-ubuntu-x64.tar.gz",
     "browser_download_url": "u9", "size": 9},
    {"name": "llama-b11645-bin-win-cpu-arm64.zip",
     "browser_download_url": "u10", "size": 10},
    {"name": "llama-b11645-bin-win-cpu-x64.zip",
     "browser_download_url": "u11", "size": 11},
    {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip",
     "browser_download_url": "u12", "size": 12},
    {"name": "llama-b11645-bin-win-vulkan-arm64.zip",
     "browser_download_url": "u13", "size": 13},
    {"name": "llama-b11645-bin-win-vulkan-x64.zip",
     "browser_download_url": "u14", "size": 14},
]

RELEASE = {"tag_name": "b11645", "assets": ASSETS}


def _pick(system, machine, flavor):
    import platform
    orig_system, orig_machine = platform.system, platform.machine
    platform.system = lambda: system
    platform.machine = lambda: machine
    try:
        picked = fetch._pick_asset(RELEASE, flavor)
        return picked[0] if picked else None
    finally:
        platform.system, platform.machine = orig_system, orig_machine


@pytest.mark.parametrize("system,machine,flavor,expected", [
    ("Windows", "AMD64", "cpu", "llama-b11645-bin-win-cpu-x64.zip"),
    ("Windows", "ARM64", "cpu", "llama-b11645-bin-win-cpu-arm64.zip"),
    ("Windows", "AMD64", "vulkan", "llama-b11645-bin-win-vulkan-x64.zip"),
    ("Windows", "ARM64", "vulkan", "llama-b11645-bin-win-vulkan-arm64.zip"),
    ("Linux", "x86_64", "cpu", "llama-b11645-bin-ubuntu-x64.tar.gz"),
    ("Linux", "aarch64", "cpu", "llama-b11645-bin-ubuntu-arm64.tar.gz"),
    ("Linux", "x86_64", "vulkan", "llama-b11645-bin-ubuntu-vulkan-x64.tar.gz"),
    ("Linux", "aarch64", "vulkan",
     "llama-b11645-bin-ubuntu-vulkan-arm64.tar.gz"),
    ("Darwin", "arm64", "cpu", "llama-b11645-bin-macos-arm64.tar.gz"),
    ("Darwin", "x86_64", "cpu", "llama-b11645-bin-macos-x64.tar.gz"),
])
def test_picks_the_real_asset(system, machine, flavor, expected):
    assert _pick(system, machine, flavor) == expected


def test_macos_vulkan_refuses_instead_of_guessing():
    import platform
    orig_system, orig_machine = platform.system, platform.machine
    platform.system = lambda: "Darwin"
    platform.machine = lambda: "arm64"
    try:
        with pytest.raises(SystemExit) as excinfo:
            fetch._platform_needle("vulkan")
        assert "Metal" in str(excinfo.value)
    finally:
        platform.system, platform.machine = orig_system, orig_machine


def test_unknown_platform_refuses():
    import platform
    orig_system, orig_machine = platform.system, platform.machine
    platform.system = lambda: "FreeBSD"
    platform.machine = lambda: "amd64"
    try:
        with pytest.raises(SystemExit):
            fetch._platform_needle("cpu")
    finally:
        platform.system, platform.machine = orig_system, orig_machine
