"""Command line interface: python -m invoicelens scan <dir> --out <dir>"""

from __future__ import annotations

import argparse
import os
import sys
import time

from . import __version__
from .classify import classify, load_rules
from .engine import DEFAULT_MODEL_PATH, EngineError, LocalLLM
from .extract import arithmetic_note, extract_fields
from .report import write_csv, write_json, write_summary
from .textsource import UnsupportedSource, read_text

SUPPORTED_EXT = (".txt", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp")


def build_parser():
    p = argparse.ArgumentParser(
        prog="invoicelens",
        description="Local-first invoice and receipt triage. Runs entirely "
                    "on this machine with an open-weight model.",
    )
    p.add_argument("--version", action="version",
                   version="invoicelens %s" % __version__)
    sub = p.add_subparsers(dest="command", required=True)

    # `ocr` exists so the OCR path can be tested without the model, and so a
    # user with an unreadable invoice can see what the machine actually saw.
    ocr_p = sub.add_parser("ocr", help="print the OCR text of one file")
    ocr_p.add_argument("file", help="image or text-based file")
    ocr_p.add_argument("--engine", default=None,
                       help="rapidocr | paddleocr | tesseract (default: first "
                            "installed)")
    ocr_p.add_argument("--lang", default=None,
                       help="language hint for engines that take one "
                            "(zh-Hans, zh-Hant, ja, ko)")
    ocr_p.add_argument("--prep", action=argparse.BooleanOptionalAction,
                       default=False,
                       help="deskew and even out the light before OCR "
                            "(default: off -- on the 60-page corpus it buys "
                            "character accuracy but no field accuracy; --prep "
                            "turns it on for badly skewed photos)")

    scan = sub.add_parser("scan", help="scan a directory of invoices")
    scan.add_argument("input_dir", help="directory containing invoice files")
    scan.add_argument("--out", default="./report",
                      help="output directory (default: ./report)")
    scan.add_argument("--rules", default=None,
                      help="deductibility rules YAML (default: bundled "
                           "rules.example.yaml)")
    scan.add_argument("--model", default=DEFAULT_MODEL_PATH,
                      help="path to the GGUF model file")
    scan.add_argument("--threads", type=int, default=None,
                      help="CPU threads for inference (default: physical "
                           "core count, capped at 8)")
    scan.add_argument("--n-gpu-layers", type=int, default=None,
                      help="layers to offload to the GPU (0 = CPU only, the "
                           "default; needs the Vulkan build of llama.cpp, "
                           "see README)")
    scan.add_argument("--device", default=None,
                      help="Vulkan device for --n-gpu-layers, e.g. vulkan1 "
                           "(default: llama.cpp picks its own, which measured "
                           "half the explicit-device rate on a 3-GPU "
                           "machine; --list-devices shows your options)")
    scan.add_argument("--n-ctx", type=int, default=4096,
                      help="context window (default: 4096)")
    scan.add_argument("--max-attempts", type=int, default=3,
                      help="extraction attempts per file incl. repair "
                           "(default: 3)")
    scan.add_argument("--max-tokens", type=int, default=512,
                      help="max tokens per model reply (default: 512)")
    scan.add_argument("--prep", action=argparse.BooleanOptionalAction,
                      default=False,
                      help="deskew and even out the light on image inputs "
                           "before OCR (default: off -- see `ocr --help`; "
                           "--prep enables it)")
    return p


def _list_inputs(input_dir):
    if not os.path.isdir(input_dir):
        raise SystemExit("not a directory: %s" % input_dir)
    files = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and name.lower().endswith(SUPPORTED_EXT):
            files.append(path)
    return files


def cmd_scan(args, llm=None):
    files = _list_inputs(args.input_dir)
    if not files:
        raise SystemExit(
            "no readable files in %s (looked for %s; images need an OCR "
            "engine -- see README)"
            % (args.input_dir, " ".join(SUPPORTED_EXT))
        )

    try:
        rules = load_rules(args.rules)
    except Exception as exc:
        raise SystemExit("rules error: %s" % exc)

    print("InvoiceLens %s: local-only invoice triage" % __version__)
    print("model : %s (%s bytes)" % (args.model, _size(args.model)))
    print("rules : %s (%d categories)" % (rules.source, len(rules.categories)))
    print("files : %d" % len(files))
    print("")

    if llm is None:
        try:
            llm = LocalLLM(model_path=args.model, n_ctx=args.n_ctx,
                           n_threads=args.threads,
                           n_gpu_layers=args.n_gpu_layers,
                           gpu_device=args.device)
            llm.load()
        except EngineError as exc:
            raise SystemExit("engine error: %s" % exc)
        # Print the engine line after load, from the resolved config, so it
        # reflects env fallbacks too. getattr keeps injected stubs working.
        ngl = getattr(llm, "n_gpu_layers", None) or 0
        device = getattr(llm, "gpu_device", None)
        if ngl > 0:
            print("engine: llama.cpp, %d layers on GPU%s"
                  % (ngl, " (%s)" % device if device else ""))
            if not device:
                print("        note: no --device given; llama.cpp picks its "
                      "own default, which measured half the explicit-device "
                      "rate on this 3-GPU machine -- see "
                      "tools/bench_backends.py --list-devices")
        elif device:
            print("engine: llama.cpp, CPU only "
                  "(--device ignored without --n-gpu-layers)")
        else:
            print("engine: llama.cpp, CPU only")

    records = []
    started = time.time()
    for idx, path in enumerate(files, 1):
        print("[%d/%d] %s" % (idx, len(files), os.path.basename(path)))
        rec = {"file": path, "fields": {}, "category": "uncategorized",
               "deductible": None, "rule_note": "", "attempts": 0,
               "repaired": False, "errors": [], "error": None,
               "arithmetic": None}
        records.append(rec)
        try:
            text = read_text(path, preprocess=args.prep)
        except UnsupportedSource as exc:
            rec["error"] = str(exc)
            print("      skipped: %s" % exc)
            continue

        res = extract_fields(llm, text, max_attempts=args.max_attempts,
                             max_tokens=args.max_tokens)
        rec["attempts"], rec["repaired"], rec["errors"] = (
            res.attempts, res.repaired, res.errors)
        if not res.ok:
            rec["error"] = "extraction failed after %d attempts" % res.attempts
            print("      extraction failed: %s" % "; ".join(res.errors[-1:]))
            continue

        rec["fields"] = res.fields
        rec["arithmetic"] = arithmetic_note(res.fields)
        rec["category"], rec["deductible"], rec["rule_note"] = classify(
            res.fields, rules)
        print("      vendor=%s date=%s total=%s %s"
              % (res.fields.get("vendor") or "?",
                 res.fields.get("invoice_date") or "?",
                 res.fields.get("total"),
                 res.fields.get("currency") or ""))
        print("      category=%s deductible=%s"
              % (rec["category"], rec["deductible"]))
        if rec["arithmetic"]:
            print("      ARITHMETIC: %s" % rec["arithmetic"])
        print("      extraction: %d attempt%s%s"
              % (res.attempts, "" if res.attempts == 1 else "s",
                 ", repaired" if res.repaired else ""))

    elapsed = time.time() - started
    os.makedirs(args.out, exist_ok=True)
    write_csv(records, os.path.join(args.out, "invoices.csv"))
    write_json(records, os.path.join(args.out, "invoices.json"))
    write_summary(records, os.path.join(args.out, "summary.md"),
                  "%s (%s bytes)" % (os.path.basename(args.model),
                                     _size(args.model)),
                  "%s (%d categories)" % (os.path.basename(rules.source),
                                          len(rules.categories)))

    ok = sum(1 for r in records if not r["error"])
    print("")
    print("done in %.1fs: %d/%d extracted, %d repaired, %d failed"
          % (elapsed, ok, len(records),
             sum(1 for r in records if r["repaired"]),
             len(records) - ok))
    print("wrote %s" % os.path.abspath(args.out))

    # Stop the inference server we own. A no-op when a stub was injected.
    closer = getattr(llm, "close", None)
    if callable(closer):
        closer()
    return 0


def _size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return "?"


def cmd_ocr(args):
    from . import ocr as ocrmod

    if ocrmod.available_engines() == []:
        print("no OCR engine installed.", file=sys.stderr)
        return 2
    try:
        res = ocrmod.ocr_image(args.file, engine=args.engine, lang=args.lang,
                               preprocess=args.prep)
    except ocrmod.OCRUnavailable as exc:
        print("OCR failed: %s" % exc, file=sys.stderr)
        return 2
    if not res.text.strip():
        print("no text found in %s" % args.file, file=sys.stderr)
        return 3
    print("engine: %s  lines: %d  avg confidence: %s"
          % (res.engine, len(res.lines),
             "%.3f" % res.confidence if res.confidence is not None else "n/a"))
    print("-" * 60)
    print(res.text)
    return 0


def _ensure_utf8_stdout():
    """Make CJK output legible on Windows.

    A frozen Windows build inherits the OEM code page (cp936 on a Chinese
    install, cp932 on Japanese). Printing Japanese text to it produces
    question marks and half-glyphs, which looks like a broken OCR when the
    OCR was fine. Switching the console to UTF-8 once fixes both a real
    console and a pipe.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - never fail because of cosmetics
        pass


def main(argv=None):
    _ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    if args.command == "ocr":
        return cmd_ocr(args)
    if args.command == "scan":
        return cmd_scan(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
