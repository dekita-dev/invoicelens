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
| Scanned PDFs and images (OCR) | **Not in v1** — see roadmap |
| A GUI | No. CLI only. |
| Tax advice | **No.** This sorts and flags. It does not decide your return. |

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

done in 199.0s: 3/3 extracted, 0 repaired, 0 failed
wrote ./report
```

That is the real output of the real run on the bundled fixtures: three
synthetic invoices, one cold model load, 199 seconds wall clock on a 4-core
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
`--threads`, `--n-ctx`, `--max-attempts`, `--max-tokens`.

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

And what it costs you is **time**, honestly: about 60 seconds per invoice on
four cores, against roughly two seconds for a hosted API. That is the real
trade. If you have a few hundred invoices and an afternoon, local is fine. If
you have five thousand and a deadline, send them to an API you have vetted and
accept the privacy cost knowingly.

One more measured detail worth knowing: the system prompt is 986 characters,
and the measured prompt for the first file was 583 tokens. Roughly 40% of what
you send on every single call is the instructions, not the invoice. That
overhead is identical whether you run locally or pay per token.

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

- **Text-based PDFs only.** A scanned invoice has no text layer, and v1 says
  so instead of extracting nothing and reporting a confident zero. OCR is on
  the roadmap.
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

- **OCR for scans and photos.** The single biggest gap. The plan is to keep it
  optional and local (a bundled Tesseract or an ONNX OCR model), because the
  whole privacy argument collapses the moment the image goes to a cloud API.
- **Confidence flags.** Have the model return a per-field confidence so the
  "needs review" list is driven by the model rather than by rule misses.
- **Currency normalization** using an offline rate table.
- **Receipt photo intake** through the same local OCR path.

## Layout

```
invoicelens/
  engine.py       starts and talks to llama-server
  extract.py      prompt, JSON parsing, repair loop
  classify.py     keyword rules -> category + deductibility
  textsource.py   .txt / .pdf -> plain text, or a loud refusal
  report.py       CSV / JSON / Markdown writers
  cli.py          argument parsing and the run loop
rules/
  rules.example.yaml
tests/            pytest, no model required
fixtures/         synthetic invoices
tools/
  fetch_llama_cpp.py   download + unzip the llama.cpp release
  smoke_engine.py      end-to-end check against a real model
  measure_tokens.py    real prompt/completion token counts per file
  bench_threads.py     thread-count benchmark, one server per setting
```

## Development

```bash
pip install pytest
python -m pytest               # 25 tests, all stubbed, no model download
python tools/smoke_engine.py  # needs the runtime + model
python tools/measure_tokens.py  # real token counts, needs the model
```
