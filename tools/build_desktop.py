"""Build a standalone InvoiceLens binary with PyInstaller.

What this is
------------
Desktop attempt #1 is deliberately not a GUI. The thing that makes a desktop
product different from a pip package is that the user does not have to install
Python, and that is what this script proves. A window comes after the binary
exists and the OCR payload is known.

Why onedir, not onefile
-----------------------
onefile unpacks ~200 MB to %TEMP% on every launch. For a tool a freelancer
opens a few times a month that is several seconds of black screen and a
surprise anti-virus hit. onedir starts instantly and is honest about its size.

Why rapidocr only
-----------------
paddleocr pulls in pydantic, pandas, huggingface-hub, modelscope and a model
directory that PyInstaller cannot reliably trace. It is imported lazily inside
invoicelens.ocr, so leaving it out of the bundle is free: the binary just
reports it as unavailable.

Usage
  python tools/build_desktop.py            # onedir
  python tools/build_desktop.py --check    # show what would be bundled
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = os.path.join(ROOT, "packaging", "invoicelens_entry.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--name", default="invoicelens")
    args = ap.parse_args()

    if not os.path.isfile(ENTRY):
        raise SystemExit("missing entry point: %s" % ENTRY)

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        raise SystemExit("pyinstaller not installed: pip install pyinstaller")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--console",
        "--name", args.name,
        "--distpath", os.path.join(ROOT, "dist"),
        "--workpath", os.path.join(ROOT, "build", "pyi"),
        "--specpath", os.path.join(ROOT, "build"),
        # rapidocr ships its ONNX models inside the package; PyInstaller does
        # not follow data files by import graph, so collect the whole tree.
        "--collect-all", "rapidocr_onnxruntime",
        # Excluded on purpose: the bundle must stay one OCR engine wide.
        "--exclude-module", "paddleocr",
        "--exclude-module", "paddle",
        "--exclude-module", "paddlex",
        "--exclude-module", "torch",
        ENTRY,
    ]
    print(" ".join(cmd))
    if args.check:
        return 0

    out = subprocess.run(cmd, cwd=ROOT)
    if out.returncode != 0:
        return out.returncode

    exe = os.path.join(ROOT, "dist", args.name, args.name + ".exe")
    if not os.path.isfile(exe):
        exe = os.path.join(ROOT, "dist", args.name, args.name)
    size = 0
    for dirpath, _dirnames, filenames in os.walk(
            os.path.join(ROOT, "dist", args.name)):
        for f in filenames:
            size += os.path.getsize(os.path.join(dirpath, f))
    print("\nbuilt: %s" % exe)
    print("bundle size: %.1f MB" % (size / 1024 / 1024))
    with open(os.path.join(ROOT, "dist", "SIZE.txt"), "w", encoding="utf-8") as f:
        f.write("%s\n%.1f MB\n" % (exe, size / 1024 / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
