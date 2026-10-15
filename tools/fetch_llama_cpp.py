#!/usr/bin/env python
"""Download and unpack the official llama.cpp release binary.

InvoiceLens needs a llama.cpp runtime but deliberately does not vendor one:
the engine talks to `llama-server` over HTTP, so any recent release works and
you can upgrade independently of this project.

This helper picks the right archive for your platform, downloads it, and
unpacks it to ~/.cache/invoicelens/llama.cpp (override with
INVOICELENS_LLAMA_DIR). The Vulkan flavor is a separate archive and a
separate directory (~/.cache/invoicelens/llama.cpp-vulkan, override with
INVOICELENS_LLAMA_DIR_VULKAN), so the two builds never overwrite each other.

Usage:
    python tools/fetch_llama_cpp.py            # latest CPU release
    python tools/fetch_llama_cpp.py --flavor vulkan
    python tools/fetch_llama_cpp.py --tag b11620
    python tools/fetch_llama_cpp.py --list     # show what is available
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
import zipfile

API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"


def _default_dest(flavor):
    """Unpack directory for a flavor. One env override per flavor, so a
    Vulkan download can never land on top of the CPU build."""
    env = "INVOICELENS_LLAMA_DIR" if flavor == "cpu" else (
        "INVOICELENS_LLAMA_DIR_" + flavor.upper())
    override = os.environ.get(env)
    if override:
        return override
    name = "llama.cpp" if flavor == "cpu" else ("llama.cpp-" + flavor)
    return os.path.join(os.path.expanduser("~"), ".cache", "invoicelens", name)


def _get_json(url, tries=4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/vnd.github+json",
                              "User-Agent": "invoicelens-fetch"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
            time.sleep(2 * (i + 1))
    raise SystemExit("could not reach the GitHub API (%s). "
                     "Download the archive by hand: %s" % (last, url))


def _platform_needle(flavor="cpu"):
    """Filename fragment for this OS/arch and backend flavor.

    The flavor fragments are deliberately non-overlapping ("win-cpu-x64"
    is not a substring of "win-vulkan-x64", and "ubuntu-x64" is not a
    substring of "ubuntu-vulkan-x64"), so a plain `in` match stays safe.
    """
    system = platform.system()
    machine = platform.machine().lower()
    x64 = machine in ("amd64", "x86_64")
    arm = machine in ("arm64", "aarch64")
    if flavor == "vulkan":
        if system == "Darwin":
            raise SystemExit(
                "llama.cpp ships no Vulkan build for macOS; the CPU build "
                "there already uses Metal, which is faster anyway.")
        if system == "Windows":
            if x64:
                return "win-vulkan-x64"
            raise SystemExit("no Windows-on-ARM Vulkan build exists")
        if system == "Linux":
            if x64:
                return "ubuntu-vulkan-x64"
            raise SystemExit("no ARM64 Linux Vulkan build in the release")
        raise SystemExit("unsupported platform: %s %s" % (system, machine))
    if system == "Windows":
        return "win-cpu-x64" if x64 else "win-cpu-arm64"
    if system == "Darwin":
        return "macos-arm64" if arm else "macos-x64"
    if system == "Linux":
        return "ubuntu-x64" if x64 else "ubuntu-arm64"
    raise SystemExit("unsupported platform: %s %s" % (system, machine))


def _pick_asset(release, flavor="cpu"):
    needle = _platform_needle(flavor)
    for asset in release.get("assets", []):
        name = asset.get("name", "")
        if needle in name and name.endswith(".zip"):
            return name, asset["browser_download_url"], asset["size"]
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=None,
                    help="release tag, e.g. b11620 (default: newest with a "
                         "binary for your platform)")
    ap.add_argument("--list", action="store_true",
                    help="list recent releases and exit")
    ap.add_argument("--flavor", choices=("cpu", "vulkan"), default="cpu",
                    help="backend flavor to fetch (default: cpu). The "
                         "Vulkan build is what GPU offload needs; it "
                         "unpacks to its own directory.")
    ap.add_argument("--dest", default=None, help="unpack directory")
    args = ap.parse_args(argv)
    dest = args.dest or _default_dest(args.flavor)

    if args.list:
        releases = _get_json(API + "?per_page=15")
        for r in releases:
            asset = _pick_asset(r, args.flavor)
            print("  %-10s %-10s %s" % (
                r.get("tag_name"), (r.get("published_at") or "")[:10],
                asset[0] if asset else "(no binary for this platform)"))
        return 0

    if args.tag:
        release = _get_json("%s/tags/%s" % (API, args.tag))
    else:
        # Walk back until we find a release that ships our platform's binary.
        release = None
        for page in range(1, 6):
            for r in _get_json("%s?per_page=30&page=%d" % (API, page)):
                if _pick_asset(r, args.flavor):
                    release = r
                    break
            if release:
                break
        if release is None:
            raise SystemExit("no release found with a binary for %s"
                             % _platform_needle(args.flavor))

    picked = _pick_asset(release, args.flavor)
    if not picked:
        raise SystemExit("release %s has no archive for %s"
                         % (release.get("tag_name"),
                            _platform_needle(args.flavor)))
    name, url, size = picked
    tag = release.get("tag_name")

    print("release : %s (%s)" % (tag, (release.get("published_at") or "")[:10]))
    print("archive : %s (%.1f MB)" % (name, size / 1048576.0))
    print("dest    : %s" % dest)

    req = urllib.request.Request(
        url, headers={"User-Agent": "invoicelens-fetch"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        blob = resp.read()
    if len(blob) != size:
        raise SystemExit("size mismatch: got %d, expected %d" % (len(blob), size))
    print("verified: %d bytes" % len(blob))

    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        if zf.testzip() is not None:
            raise SystemExit("downloaded archive is corrupt")
        zf.extractall(dest)

    server = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    path = os.path.join(dest, server)
    if not os.path.isfile(path):
        raise SystemExit("unpacked, but %s is missing from %s" % (server, dest))
    print("ready   : %s" % path)
    print("\nEither add it to PATH or set:")
    print("  INVOICELENS_LLAMA_DIR%s=%s"
          % ("" if args.flavor == "cpu" else "_" + args.flavor.upper(), dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
