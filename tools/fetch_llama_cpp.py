#!/usr/bin/env python
"""Download and unpack the official llama.cpp release binary.

InvoiceLens needs a llama.cpp runtime but deliberately does not vendor one:
the engine talks to `llama-server` over HTTP, so any recent release works and
you can upgrade independently of this project.

This helper picks the right archive for your platform, downloads it, and
unpacks it to ~/.cache/invoicelens/llama.cpp (override with
INVOICELENS_LLAMA_DIR).

Usage:
    python tools/fetch_llama_cpp.py            # latest release
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
DEST = os.environ.get(
    "INVOICELENS_LLAMA_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "invoicelens", "llama.cpp"),
)


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


def _platform_needle():
    """Filename fragment for this OS/arch."""
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows":
        return "win-cpu-x64" if machine in ("amd64", "x86_64") else "win-cpu-arm64"
    if system == "Darwin":
        return "macos-arm64" if machine in ("arm64", "aarch64") else "macos-x64"
    if system == "Linux":
        return "ubuntu-x64" if machine in ("amd64", "x86_64") else "ubuntu-arm64"
    raise SystemExit("unsupported platform: %s %s" % (system, machine))


def _pick_asset(release):
    needle = _platform_needle()
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
    ap.add_argument("--dest", default=DEST, help="unpack directory")
    args = ap.parse_args(argv)

    if args.list:
        releases = _get_json(API + "?per_page=15")
        for r in releases:
            asset = _pick_asset(r)
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
                if _pick_asset(r):
                    release = r
                    break
            if release:
                break
        if release is None:
            raise SystemExit("no release found with a binary for %s"
                             % _platform_needle())

    picked = _pick_asset(release)
    if not picked:
        raise SystemExit("release %s has no archive for %s"
                         % (release.get("tag_name"), _platform_needle()))
    name, url, size = picked
    tag = release.get("tag_name")

    print("release : %s (%s)" % (tag, (release.get("published_at") or "")[:10]))
    print("archive : %s (%.1f MB)" % (name, size / 1048576.0))
    print("dest    : %s" % args.dest)

    req = urllib.request.Request(
        url, headers={"User-Agent": "invoicelens-fetch"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        blob = resp.read()
    if len(blob) != size:
        raise SystemExit("size mismatch: got %d, expected %d" % (len(blob), size))
    print("verified: %d bytes" % len(blob))

    os.makedirs(args.dest, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        if zf.testzip() is not None:
            raise SystemExit("downloaded archive is corrupt")
        zf.extractall(args.dest)

    server = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    path = os.path.join(args.dest, server)
    if not os.path.isfile(path):
        raise SystemExit("unpacked, but %s is missing from %s" % (server, args.dest))
    print("ready   : %s" % path)
    print("\nEither add it to PATH or set:")
    print("  INVOICELENS_LLAMA_DIR=%s" % args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
