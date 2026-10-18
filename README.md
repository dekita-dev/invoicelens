# InvoiceLens

**Local-first invoice and receipt triage for sole proprietors and freelancers.**
An open-weight model (Gemma 2 2B) runs on your own machine, reads your
invoices, and turns them into a categorized, deductible-flagged spreadsheet.
No API key. No account. Nothing leaves the laptop.

Built for the [Hacktoberfest 2026 Open-Source AI Challenge, Week 2: Side
Hustle](https://dev.to/devteam/join-the-hacktoberfest-open-source-ai-challenge-side-hustle-2550-in-prizes-across-18-winners-55ba).

---

## Who would pay for this

Anyone who bills by the hour and files their own taxes. A freelancer gets
invoices from a designer in September, a hosting bill in October, and a pile
of mixed receipts in between. At tax time they need one spreadsheet that says
*what came in, from whom, and which of it is a business cost*.

The two things that make this hard with a hosted LLM API are the two things
this project is built around:

- **The data is sensitive.** Client names, project names, amounts, sometimes
  addresses. Sending that to a third-party API to get a spreadsheet is a real
  cost for a one-person business.
- **The cost has to stay below the value.** A few dollars a month in API
  calls to sort a few dozen invoices is a bad trade. Running a 2B model on
  hardware you already own has a marginal cost of zero.

## Status (measured, not claimed)

| Capability | Status |
|---|---|
| Text-based `.pdf` and `.txt` input | Works |
| Field extraction (vendor, dates, amounts, line items) | Works |
| Expense categorization from a YAML rule file | Works |
| Deductibility flags | Works, rules are yours to edit |
| CSV + JSON + Markdown summary output | Works |
| Field extraction with a self-repair loop | Works |
| Images (`.png` `.jpg` `.jpeg` `.webp` `.bmp`) via local OCR | Works, engine optional — see below |
| Phone photos of paper invoices | Works; a perspective and lighting pass exists and is off by default (measurements below) |
| Chinese, Japanese, Korean invoices | Partial — totals read correctly on 58 of 60 synthetic fixtures; text fields depend on the script (table below) |
| Scanned PDFs | **Not in v1** — needs a page renderer |
| A GUI | No. CLI only. |
| Tax advice | **No.** This sorts and flags. It does not decide your return. |

The CJK and image numbers are measured, not claimed. The corpus is 60
synthetic invoices (5 languages x 3 templates x 4 degradation tiers, from
`tools/gen_cjk_fixtures.py`), the engine is RapidOCR, and the full run with
every failure listed is
[`report/P0_CJK基线_v2.md`](report/P0_CJK基线_v2.md), with the raw rows
beside it as JSON.

| Script | Total | Invoice no | Date | Vendor |
|---|---|---|---|---|
| English | 10/12 | 10/12 | 9/12 | 11/12 |
| Japanese | 12/12 | 11/12 | 10/12 | 0/12 |
| Korean | 12/12 | 12/12 | 0/12 | 0/12 |
| Simplified Chinese | 12/12 | 12/12 | 12/12 | 12/12 |
| Traditional Chinese | 12/12 | 12/12 | 12/12 | 6/12 |

The table hides two things worth saying in words. Korean dates are not read
at all by the default engine: it keeps the digits and drops the date
markers, so `2026년 10월 07일` comes back as `20260928`. Japanese and Korean
vendor names are not read either: the kanji and hangul come back mangled
while the numbers beside them are correct, which is the failure shape that
looks fine in a summary and wrong in the invoice. Installing the optional
PaddleOCR engine trades about a minute per page for those fields.

## Install

Two pieces: the Python package, and a llama.cpp runtime.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10 or newer. Dependencies are `requests`, `pypdf` and
`PyYAML` — nothing that needs a compiler.

Reading images needs one more piece, and it is optional: an OCR engine.
InvoiceLens probes for RapidOCR, PaddleOCR and Tesseract and uses whichever
it finds:

```bash
pip install rapidocr-onnxruntime onnxruntime   # smallest, works everywhere
```

Without any engine, text files and text-based PDFs still work exactly as
before; handing the tool an image raises a loud `OCRUnavailable` telling you
which engines it looked for. PaddleOCR is the second engine and measurably
better on Chinese, Japanese and Korean vendor names, at about a minute per
page on CPU; Tesseract is detected if it is already on your `PATH`.

## Getting the runtime

InvoiceLens drives `llama-server` from the official llama.cpp release as a
subprocess and talks to it over HTTP on `127.0.0.1`. Download the build for
your platform and either put it on `PATH` or point an environment variable at
it:

```bash
# Windows (CPU build, ~19 MB)
#   https://github.com/ggml-org/llama.cpp/releases
#   -> llama-b*-bin-win-cpu-x64.zip
# unzip somewhere, then either:
export PATH="/path/to/unzipped:$PATH"
# or:
export INVOICELENS_LLAMA_DIR="/path/to/unzipped"
```

On Linux and macOS the equivalent archive is `llama-b*-bin-ubuntu-x64.zip` or
`llama-b*-bin-macos-*.zip`. There is a helper that does the download and
unzip for you:

```bash
python tools/fetch_llama_cpp.py
```

## Getting the model

InvoiceLens does not download anything on first run. You fetch the weights
once, explicitly:

```bash
mkdir -p ~/.cache/invoicelens/models

# HuggingFace
huggingface-cli download lmstudio-community/gemma-2-2b-it-GGUF \
  gemma-2-2b-it-Q4_K_M.gguf --local-dir ~/.cache/invoicelens/models

# or curl, straight from the repo
curl -L -o ~/.cache/invoicelens/models/gemma-2-2b-it-Q4_K_M.gguf \
  https://huggingface.co/lmstudio-community/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf

# ModelScope mirror, if HuggingFace is unreachable from your network
curl -L -o ~/.cache/invoicelens/models/gemma-2-2b-it-Q4_K_M.gguf \
  https://modelscope.cn/models/lmstudio-community/gemma-2-2b-it-GGUF/resolve/master/gemma-2-2b-it-Q4_K_M.gguf
```

That is about 1.6 GB. It happens once, and after that the tool is fully
offline.

Any Gemma-family GGUF will work — pass the path with `--model`. Smaller
quantizations (`Q3_K_M`) run faster on a laptop; larger ones extract more
reliably.

## Usage

```bash
python -m invoicelens scan ./invoices --out ./report
```

```
InvoiceLens 0.1.0: local-only invoice triage
model : ~/.cache/invoicelens/models/gemma-2-2b-it-Q4_K_M.gguf (1708582752 bytes)
rules : rules.example.yaml (7 categories)
files : 3

engine: llama.cpp, CPU only
[1/3] design-retainer-2026-09.txt
      vendor=NORTHWIND STUDIO date=2026-09-03 total=2400.0 USD
      category=professional_services deductible=True
      extraction: 1 attempt

[2/3] office-mixed-2026-09.txt
      vendor=MERIDIAN OFFICE SUPPLY date=2026-09-27 total=1124.82 USD
      category=office_supplies deductible=True
      extraction: 1 attempt

[3/3] saas-subscription-2026-10.txt
      vendor=GITHUB, INC. date=2026-10-02 total=4.0 USD
      category=software_subscriptions deductible=True
      extraction: 1 attempt

done in 223.3s: 3/3 extracted, 0 repaired, 0 failed
wrote ./report
```

That is the real output of the real run on the bundled fixtures: three
synthetic invoices, one cold model load, 223.3 seconds wall clock on a 4-core
i5-8350U with the Q4_K_M quantization. No file needed a repair prompt. The
only thing normalized above is the path, abbreviated to `~`; the byte count,
the timings and every extracted field are as the tool printed them. Vendor
names come back in whatever case the model found them, which is why the CSV
writer does not pretend to normalize them.

Outputs:

| File | Contents |
|---|---|
| `invoices.csv` | One row per file, ready for a spreadsheet |
| `invoices.json` | Same data plus per-file extraction diagnostics |
| `summary.md` | Grouped by category, with a "needs review" section |

What they actually look like, from the run above:

```
$ cat report/invoices.csv
file,vendor,invoice_number,invoice_date,due_date,currency,subtotal,tax,total,category,deductible,rule_note,extraction_attempts,extraction_repaired,error
design-retainer-2026-09.txt,NORTHWIND STUDIO,NS-2026-0917,2026-09-03,2026-10-03,USD,2400.0,0.0,2400.0,professional_services,True,Subcontractor and agency invoices are normally a business cost.,1,False,
office-mixed-2026-09.txt,MERIDIAN OFFICE SUPPLY,MOS-88921,2026-09-27,2026-10-27,USD,1041.5,83.32,1124.82,office_supplies,True,Office equipment may be deductible or may need to be capitalized. Check locally.,1,False,
saas-subscription-2026-10.txt,"GITHUB, INC.",GH-7714-2026,2026-10-02,,USD,4.0,0.0,4.0,software_subscriptions,True,Business software subscriptions are normally deductible.,1,False,
```

```
$ cat report/summary.md
# InvoiceLens run summary

- model: gemma-2-2b-it-Q4_K_M.gguf (1708582752 bytes)
- rules: rules.example.yaml (7 categories)
- files processed: 3
- extracted: 3
- needed a repair prompt: 0
- failed: 0
- arithmetic mismatches: 0

## By category

### office_supplies (1)

- **MERIDIAN OFFICE SUPPLY** | USD 1124.82 | True | invoice date 2026-09-27

### professional_services (1)

- **NORTHWIND STUDIO** | USD 2400.00 | True | invoice date 2026-09-03

### software_subscriptions (1)

- **GITHUB, INC.** | USD 4.00 | True | invoice date 2026-10-02

## Needs review

_Nothing._
```

`invoices.json` carries the same rows plus the full extracted `line_items`
array, the raw model reply, and the per-file `extraction` block
(`attempts`, `repaired`, `errors`). That is where you look when you want to
know whether the model struggled on a particular file.

Options: `--rules` (your own rule file), `--model` (another GGUF),
`--threads`, `--n-ctx`, `--max-attempts`, `--max-tokens`,
`--n-gpu-layers` and `--device` (opt-in GPU offload, see below), and
`--prep`/`--no-prep` (image preprocessing, off by default — see below).

## Photographing an invoice

Point the camera at the paper and the tool does the rest. There is one
optional pass (`preprocess.py`) that you turn on per run:

1. **Corner detection and perspective correction.** The page outline is
   found, and a tilted photo is warped back to a flat rectangle.
2. **Illumination normalization.** A blurred copy of the image estimates the
   lighting field, and dividing it out removes shadows and gradients.

The pass is all-or-nothing: if any step fails, the original image goes to
OCR untouched. It also skips itself when the page already fills the frame.

**It is off by default, and that is a measured decision.** On the 60-page
corpus it cuts character error on hand-held photos by 15 points
(39.18% to 23.83%) and changes nothing that the tool reports:

| Tier | CER without | CER with | Totals correct without | Totals correct with |
|---|---|---|---|---|
| photo (hand-held simulation) | 39.18% | 23.83% | 15/15 | 15/15 |
| hard (blur + downsample) | 44.94% | 47.09% | 13/15 | 13/15 |

Character accuracy that no reported field benefits from is not worth a
half-second per page and one more thing that can go wrong, so the default
is the raw image. The rule was written down before the data came in
(`report/P0_CJK基线_v2.md` records both), and `--prep` turns the pass on:
use it when a photo is badly skewed or half in shadow.

What actually helps at capture time, since the correction is not magic:

- **Fill the frame with the page.** A tight crop is what lets the corner
  detector see the page edges at all.
- **Even light beats bright light.** A shadow across the page is the hardest
  thing to divide out; two lamps from opposite sides beat one overhead.
- **Do not shoot at an extreme angle.** Perspective correction assumes a flat
  rectangle; a 45-degree shot foreshortens the text itself, which no warp
  can restore.

## What it actually costs (measured, and the answer is not what you expect)

Three invoices, one cold load, measured on an i5-8350U with the Q4_K_M
quantization. Token counts are server-reported, not estimated
(`tools/measure_tokens.py`):

| File | Prompt tokens | Completion tokens | Wall clock |
|---|---|---|---|
| design-retainer-2026-09.txt | 583 | 273 | 57.0 s |
| office-mixed-2026-09.txt | 573 | 400 | 72.9 s |
| saas-subscription-2026-10.txt | 508 | 258 | 50.4 s |
| **Total** | **1664** | **931** | **180.3 s** |

The token counts came out identical on a second, independent run, which is
worth knowing: tokenization of a fixed prompt is deterministic even though
generation is not.

Priced against **Gemini 2.5 Flash-Lite** at $0.10 per million input and $0.40
per million output tokens (Google's published rate, page last updated
2026-10-14 — the cheapest tier on that page), those 2595 tokens cost
**$0.000539**, or about **$0.00018 per invoice**.

Now the same run as electricity. At an assumed 45 W whole-machine draw for
180.3 seconds and $0.15/kWh, that is **$0.000338**. The hosted API is **1.59×**
the electricity. For a freelancer processing 300 invoices a year the two
differ by about **2 cents**.

**So the cost argument for running locally is, on this evidence, a wash, and
this README will not pretend otherwise.** What local inference actually buys
you here is not money:

- **The data does not leave.** Google's own pricing page states that on the
  free tier, inputs and outputs are used to improve Google's products. The
  paid tier does not. Either way the invoice — client names, project names,
  amounts — goes to a third party.
- **No account, no key, no quota, no deprecation.** Nothing to rotate, nothing
  to be rate-limited by, no model version pulled out from under you.
- **It works on a plane.**

And what it costs you is **time**, honestly: one to one and a quarter minutes
per invoice on four cores (180.3 s and 223.3 s on two runs of the same three
invoices), against roughly two seconds for a hosted API. That is the real
trade. If you have a few hundred invoices and an afternoon, local is fine. If
you have five thousand and a deadline, send them to an API you have vetted and
accept the privacy cost knowingly.

One more measured detail worth knowing: the system prompt is 986 characters,
and the measured prompt for the first file was 583 tokens. Roughly 40% of what
you send on every single call is the instructions, not the invoice. That
overhead is identical whether you run locally or pay per token.

## GPU offload (opt-in, and why it is not the default)

On a machine with a discrete GPU the same run gets a lot faster, and the
number is worth having. Everything below was measured on this machine, which
has three GPUs; the raw logs and the decision record are in
[`docs/bench-vulkan.md`](docs/bench-vulkan.md).

First the device matrix, from `llama-bench` (512-token prompt, 16 generated
tokens, 4 threads, all 26 layers offloaded):

| Device | Prompt tok/s | Generation tok/s |
|---|---|---|
| CPU, 4 threads | 23.7 | 6.4 |
| Intel UHD 620 (integrated) | 18.5 | 4.3 |
| AMD RX 6600 | 1142.6 | 42.1 |
| Quadro P500 | cannot allocate | - |

The integrated GPU is **slower than the CPU**, so "offload to the GPU" is not
advice, it is a device-selection problem. With the fastest device selected,
the real three-invoice workload (`tools/bench_backends.py`):

| | CPU (default) | RX 6600, 26 layers |
|---|---|---|
| Prompt tok/s | 23.0 | 604 |
| Generation tok/s | 7.44 | 38.7 |
| Cold start | 5.7 s | 8.3 s |
| Three invoices, wall clock | 223.3 s | 26.1 s |

That is **8.6x end to end**: about 74 seconds per invoice down to about 9.

Usage:

```bash
python tools/bench_backends.py --list-devices        # see what your machine has
python -m invoicelens scan ./invoices --out ./report \
  --n-gpu-layers 26 --device vulkan1
```

Why it stays opt-in:

- **Device 0 is not your friend.** Omitting `--device` does not mean device 0:
  it measured half the explicit-device rate on this machine and matched no
  single GPU. `--list-devices` exists because the fastest device is not
  something the tool can know for you.
- **On a Vulkan build, `--n-gpu-layers 0` does not mean CPU compute.** The
  weights stay in system RAM but the operators still run on the GPU
  (`llama-bench` reports `backend: Vulkan`). InvoiceLens therefore sends
  `--n-gpu-layers 0` explicitly on the CPU path: upstream llama.cpp changed
  that flag's default from 0 to `auto`, so a flag you do not pass is a flag
  you do not control.
- **The quality caveat is real.** Every field the tool depends on came out
  identical between the CPU run and the GPU run: vendor, dates, amounts, all
  nine line items, deductibility, one attempt each, zero repairs. One optional
  free-text field did not: `notes` missed a line it had caught on CPU,
  reproducibly, across two GPU runs. Same model, different arithmetic,
  different answer on a field nobody should be trusting anyway. Review the
  "needs review" list either way.

## How the extraction works

The model is asked for a strict JSON object with an explicit schema, and told
to use `null` rather than guess. Two details do most of the work:

**1. A tolerant parser, not a strict one.** Model replies often arrive with
prose around the JSON. `extract.py` finds the outermost balanced `{...}`
block while respecting braces inside strings, so `{"note": "use {curly}
braces"}` does not break it.

**2. The parse error becomes the next prompt.** If the reply does not parse,
the exception text and the bad reply are fed back as a repair prompt:

```
Your previous reply was not valid JSON.

Parse error: Expecting value: line 1 column 1 (char 0)

Your previous reply was:
---
Vendor: ACME, total 10.5
---

Reply again with only the corrected JSON object matching the schema.
```

This is deliberately *not* a regex fallback. The model still owns the
extraction; the retry only tells it what went wrong. Every run reports how
many attempts each file needed and whether it was repaired, so you can see
the failure rate instead of trusting a silent success.

**3. An arithmetic cross-check, done without the model.** When subtotal,
tax and total all came back as numbers, the tool checks that they add up —
with decimal arithmetic on the string form, because `1041.5 + 83.32` is
`1124.8199999999998` in binary floating point and a check that cries wolf
on every float total is worse than no check. A page whose numbers do not
add up has at least one misread field even when each field looks fine
alone, so it lands in "needs review" with the three numbers spelled out.
The same check runs in the OCR measurement harness
(`tools/eval_ocr.py`, the `arith` column) as an independent signal next to
the per-field rates.

## The rules file

Deductibility is jurisdiction-, entity-, and year-specific, so it is not
hardcoded. `rules/rules.example.yaml` is keyword matching over the vendor
name, line item descriptions, and the model's notes:

```yaml
- name: meals_entertainment
  label: Meals and entertainment
  keywords: [restaurant, cafe, coffee, catering, meal]
  deductible: null          # true / false / null = "no opinion"
  note: "Meals are often only partly deductible and entertainment often not at all. Review."
```

`deductible: null` is a first-class value. When no rule matches, the file
lands in "needs review" rather than being quietly counted as a business cost.

## Limitations (read these before trusting the output)

- **Image-only PDFs are still out.** Photos and standalone image files are
  read through the local OCR path described above, but a PDF whose pages are
  scans has no text layer and v1 has no page renderer to turn it into one.
  The tool says so instead of extracting nothing and reporting a confident
  zero. Rendering pages is the remaining piece of the OCR story, and it is
  the top roadmap item.
- **This is not tax advice.** It sorts and flags. What is deductible is your
  call, or your accountant's.
- **A 2B model makes mistakes.** It will mis-read a date, invent a `null`
  where a number exists, or miss a line item. The diagnostics in
  `invoices.json` exist so you can see where it struggled. Review the
  spreadsheet before you file anything.
- **Keyword rules mis-file mixed receipts.** This is not theoretical — it
  happened on the bundled `office-mixed-2026-09.txt` fixture. The model
  extracted every line item correctly, including `Coffee beans, 1kg`, but the
  rule engine classified the whole invoice as `office_supplies`. The score was
  `office_supplies` 2 hits against `meals_entertainment` 1: `office` from the
  vendor name `MERIDIAN OFFICE SUPPLY`, and `monitor` from the line item
  `Monitor arm, dual`, beat `coffee` on its own. (Note also that the keyword
  list says `supplies` while the vendor says `SUPPLY`, so that keyword did not
  even fire — the match came from two other words entirely.) The coffee landed
  in office supplies instead of `meals_entertainment`, where it would at least
  have been flagged for review. One invoice, one category, is the shape of this
  limitation: a mixed receipt needs either per-line-item classification or a
  model pass over the rules, neither of which v1 does. Check the "needs
  review" list rather than trusting the category.
- **Deterministic-ish, not deterministic.** `--max-attempts 1` and a low
  temperature give more repeatable runs; nothing here is guaranteed stable
  across model versions.
- **It spawns a subprocess.** The inference server is a child process. If
  InvoiceLens is killed hard the server can outlive it; `close()` and an
  `atexit` hook cover the normal paths.

## Runtime notes (why this shape)

The obvious Python routes to local inference did not survive contact with
this machine, so the choices here are evidence-based rather than fashionable:

**`llama-cpp-python` is out.** As of October 2026 it publishes no Windows
wheels for any Python version (checked on the PyPI index: zero
`win_amd64` files), and building it needs a C++ toolchain most Windows
machines do not have.

**`gpt4all` is out too, and the reason is worth writing down.** gpt4all does
ship a `py3-none-win_amd64` wheel with a bundled llama.cpp, so it looks like
the answer. It is not, on Windows:

- The wheel contains only two backends, CUDA and Kompute/Vulkan. There is no
  CPU backend DLL at all, and `device="cpu"` is a no-op — `gpt4all.py:222`
  hard-codes `backend = "kompute"` on every non-macOS platform.
- On this machine the Kompute build aborts inside llama.cpp during model
  load. It does so identically at `llama.cpp:11912` for every combination of
  `n_ctx` (2048/4096/8192) and `n_gpu_layers` (0/1/12/100), which rules out
  context size and GPU memory as the cause.
- The bundled llama.cpp is also old. The architecture table inside the DLL
  contains `gemma2` but not `gemma3`, `llama4` or `qwen3`, which places it
  around early 2025 — before GGUF v3 was fully supported. A current Gemma 2
  GGUF (version 3) aborts in `llama_model_load` as a result.

**So: drive the official llama.cpp release binary.** It is current, it has a
real CPU backend that picks the right instruction-set variant at load time,
and `llama-server` gives an OpenAI-compatible HTTP surface, which means the
engine wrapper is about 40 lines of `urllib` instead of a ctypes binding
against a moving C ABI.

**Threads default to physical cores, not logical ones.** Generation with a
small quantized model is memory-bandwidth bound, so hyperthreads cost more
than they return. Measured on this machine with `tools/bench_backends.py`,
which reads the server's own `timings` block instead of dividing wall clock
by a guessed token count: prompt processing scales from 18.7 tok/s at 2
threads to 33.6 at 8, but generation peaks at 7.44 tok/s on the 4 physical
cores and then *drops* 22% to 5.84 once hyperthreads join in. Because
generation is roughly two thirds of a real invoice's time, the default is
physical core count, detected per platform, capped at 8.

The tradeoff is honest and documented: one subprocess to start, health-check
and stop. That is contained in `load()` and `close()` in `engine.py`.

## Licenses

InvoiceLens is MIT. Its dependencies are not all the same, and the model is
the one that needs a careful read:

| Component | License | Notes |
|---|---|---|
| InvoiceLens (this code) | MIT | Yours to fork and sell |
| llama.cpp | MIT | The inference engine, official release binary |
| requests | Apache-2.0 | HTTP client for the local server |
| pypdf | BSD-3-Clause | PDF text extraction |
| PyYAML | MIT | Rules file parsing |
| **Gemma 2 2B (model weights)** | **Google Gemma Terms of Use** | **Not an OSI open-source license** |

**On the model license, plainly:** Gemma's weights are *open-weight* — you can
download, run, inspect, and fine-tune them — but they are distributed under
[Google's Gemma Terms of Use](https://ai.google.dev/gemma/terms), which is not
an OSI-approved open-source license and carries use restrictions you are
responsible for reading. That distinction is exactly why the Hacktoberfest
prompt says "open-weight models" rather than "open-source models", and it is
the honest reason a 2B model with a use-restricted license can still be the
right core for a tool like this: you get local, private, zero-marginal-cost
inference, and the restriction that matters (Google's acceptable-use policy)
is one a one-person business sorting its own invoices will not hit.

If you need a permissively licensed model instead, swap `--model` for any
Gemma-family or Llama-family GGUF; nothing else in this code changes.

## Roadmap

- **Page rendering for image-only PDFs.** The image path is done (see
  "Photographing an invoice" above); what is left is turning scanned PDF
  pages into images locally, so the same OCR path can read them.
- **Confidence flags.** Have the model return a per-field confidence so the
  "needs review" list is driven by the model rather than by rule misses.
- **Currency normalization** using an offline rate table.
- **Cross-check on public corpora.** Done for the one corpus with a usable
  license: CORD v2, 100 real Indonesian receipts, read through the same
  code. 88% of totals survive the repair layer on the raw image (87% with
  preprocessing), against 96.7% on the synthetic set — real photographs of
  crumpled paper are harder than synthetic ones, which is exactly why the
  number belongs in the report and not in the marketing copy. The corpus
  itself is fetched on demand (`python tools/fetch_public_fixtures.py`,
  about 15 MB, CC BY 4.0 verified in
  [`docs/public-fixtures-license-audit.md`](docs/public-fixtures-license-audit.md)).
  SROIE and WildReceipt are excluded until a license statement exists, and
  the audit file records why.

## Layout

```
invoicelens/
  engine.py       starts and talks to llama-server
  extract.py      prompt, JSON parsing, repair loop
  classify.py     keyword rules -> category + deductibility
  textsource.py   .txt / .pdf -> plain text, images -> OCR, or a loud refusal
  ocr.py          local OCR engine probing: rapidocr / paddleocr / tesseract
  preprocess.py   corner detection, perspective correction, lighting flatten
  textnorm.py     deterministic repair of OCR output before the model sees it
  report.py       CSV / JSON / Markdown writers
  cli.py          argument parsing and the run loop
rules/
  rules.example.yaml
tests/            pytest, no model required
fixtures/
  cjk_synthetic/  synthetic invoices: 5 languages x 3 templates x 4 tiers
  multipage/      two-page PDFs with a text layer, total on page 2
tools/
  fetch_llama_cpp.py   download + unzip the llama.cpp release
  smoke_engine.py      end-to-end check against a real model
  measure_tokens.py    real prompt/completion token counts per file
  bench_threads.py     thread-count benchmark, one server per setting
  bench_backends.py    prompt-vs-generation sweep: backends, devices, ngl, threads
  gen_cjk_fixtures.py  regenerate the synthetic corpus (clean/scan/photo/hard)
  gen_multipage_pdf.py build two-page PDF invoices from scratch, no dependency
  check_multipage.py   read them back: are both pages there, is the total once
  eval_ocr.py          OCR baseline + raw-vs-preprocessed gain table
  fetch_public_fixtures.py  download the CORD v2 test split (100 receipts)
  build_desktop.py     one-file Windows binary via PyInstaller
```

## Development

```bash
pip install pytest
python -m pytest               # 102 tests, all stubbed, no model download
python tools/smoke_engine.py  # needs the runtime + model
python tools/measure_tokens.py  # real token counts, needs the model
```

The OCR layer has its own measurements. The synthetic corpus needs an engine
installed but no model and no network:

```bash
pip install rapidocr-onnxruntime onnxruntime
python tools/gen_cjk_fixtures.py       # regenerate the 60-page corpus
python tools/eval_ocr.py --prep both   # baseline + raw-vs-prepped gain table
python tools/gen_multipage_pdf.py      # 5 two-page PDFs, text layer, no deps
python tools/check_multipage.py        # both pages read, total appears once
```

The public corpus (CORD v2, 100 real receipts) needs the network once, to
download the images it measures:

```bash
python tools/fetch_public_fixtures.py                          # ~15 MB
python tools/eval_ocr.py --manifest fixtures/public/cord-v2/manifest.json --prep both
```

All of them write into `report/`, so a run leaves its evidence behind.
