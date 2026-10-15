# GPU offload: measured, decided, shipped

Experiment run 2026-10-16 (Beijing) against the dev.to Hacktoberfest Week 2
deadline. This file is the evidence behind the `--n-gpu-layers` / `--device`
flags in the CLI. Every number below came off this machine on the day named;
raw logs and the S6 report directories are in `_archive/bench-2026-10-15/`
and are committed.

## The question

Generation with Gemma 2 2B Q4_K_M is memory-bandwidth bound. This laptop has
an RX 6600 with dedicated GDDR6 next to an i5-8350U sharing dual-channel
DDR4. Is full Vulkan offload enough faster to justify a flag, and does it
change what the tool extracts?

Decision thresholds, set before measuring:

| Generation s/token vs CPU baseline | Verdict |
|---|---|
| <= 0.70x | GO |
| <= 0.80x | PARTIAL |
| > 0.80x | NO-GO |

## Machine state (measured, not assumed)

- CPU: Intel i5-8350U, 4 cores / 8 threads (Kaby Lake R), AC power confirmed
  (root/wmi `PowerOnline = True`) before any run
- Vulkan device enumeration (build b11645):

| Index | Device | Notes |
|---|---|---|
| vulkan0 | Intel UHD 620 | integrated, shares system RAM |
| vulkan1 | AMD RX 6600 | 8176 MiB total, 7378 MiB free, fp16 + int dot |
| vulkan2 | Quadro P500 | cannot allocate the model's compute buffers |

- Model: `gemma-2-2b-it-Q4_K_M.gguf`, 1,708,582,752 bytes
- Builds, both official `win-x64` releases, both Clang 20.1.8:
  - CPU: build 11620, commit `21a19cd3b` (`~/.cache/invoicelens/llama.cpp/`)
  - Vulkan: build 11645, commit `6e66cb5d4` (`~/.cache/invoicelens/llama.cpp-vulkan/`)

## Results

All runs `threads=4`, `ctx 4096`, median of 2. `tools/bench_backends.py`
reads the server's own `timings` block; `llama-bench` numbers are the
upstream micro-benchmark on a synthetic 512-token prompt.

### CPU baseline (build 11620, CPU backend)

| Tool | Prompt tok/s | Generation tok/s | Cold start |
|---|---|---|---|
| bench_backends | 23.0 | 7.44 | 5.7 s |
| llama-bench pp512/tg16 | 23.72 | 6.38 | - |

### GPU, RX 6600 (build 11645, vulkan1, 26/26 layers)

| Tool | Prompt tok/s | Generation tok/s | Cold start |
|---|---|---|---|
| bench_backends | 604.4 | 38.74 | 8.3 s |
| llama-bench pp512/tg16 | 1142.60 | 42.13 | - |

Generation s/token: 0.134-0.157 on CPU, 0.026 on GPU. That is **0.18x**, well
inside the GO threshold. Prompt processing, which is 64% of the tokens in a
real invoice, goes 26x faster.

### End-to-end, the real workload

Same three fixtures, same rules, back to back
(`_archive/bench-2026-10-15/S6-*.log`):

| Run | Wall clock | Extracted | Repaired | Failed |
|---|---|---|---|---|
| CPU (default) | 223.3 s | 3/3 | 0 | 0 |
| GPU (26 layers, vulkan1) | 26.1 s | 3/3 | 0 | 0 |
| GPU again (reproducibility) | 26.0 s | 3/3 | 0 | 0 |

**8.6x end to end.** Per invoice that is about 74 seconds down to about 9.

### The device matrix, and why the flag matters

`llama-bench`, pp512/tg16, threads 4, 26 layers:

| Device | Prompt tok/s | Generation tok/s |
|---|---|---|
| CPU backend (build 11620) | 23.72 | 6.38 |
| vulkan0, UHD 620 | 18.49 | 4.33 |
| vulkan1, RX 6600 | 1142.60 | 42.13 |
| vulkan2, Quadro P500 | fails to allocate | - |
| no `--device` at all | 409.24 | 19.99 |

Three things fall out of that table:

1. **The integrated GPU is slower than the CPU** (18.49 vs 23.72). Offloading
   to the wrong device is negative optimization, not a smaller win.
2. **The Quadro P500 cannot run the model at all** at this size.
3. **Omitting `--device` is not the same as asking for device 0.** It measured
   409/19.99, which matches no single device on the box, and is about half the
   explicit-RX-6600 rate. The CLI note used to claim device 0; it now says
   llama.cpp picks its own default and points at `--list-devices`.

## Quality gate

Fresh CPU scan vs fresh GPU scan, every structured field diffed
(`_archive/bench-2026-10-15/s6-reports/cpu/` vs `.../gpu/`, with `.../gpu-rerun/`
as the second GPU run):

- vendor, invoice_number, invoice_date, due_date, currency, subtotal, tax,
  total, every line item, payment_method, category, deductible, rule_note:
  **identical, bit for bit**
- extraction attempts = 1, repaired = false, errors = none on both sides
- one difference, reproducible across two GPU runs: the optional free-text
  `notes` field. On `design-retainer-2026-09.txt` the CPU runs extract
  "Net 30. Thank you for your business." (it is on line 29 of the fixture)
  and the GPU runs return null; on `office-mixed-2026-09.txt` the CPU runs
  return null and the GPU runs return `""`. Both GPU runs agree with each
  other, so this is GPU-path behaviour, not sampling noise.

Verdict: the fields the tool depends on do not move; one optional field does,
reproducibly. That caveat ships with the flag and is in the README.

## Four things that were wrong before measurement

Each of these was caught by a number that could not be true, not by reading
code. They are the reason the flags exist in their current shape.

1. **`--n-gpu-layers` defaults to `auto` upstream, not 0.** Verified with
   `--help` on both builds. Our "CPU path" omitted the flag entirely, so on
   any build with a GPU backend it silently offloaded. The first GPU control
   row (supposed to be CPU) reported 20.10 gen tok/s, which the i5-8350U
   cannot produce: that row was GPU data wearing a CPU label. Fix: emit an
   explicit `--n-gpu-layers 0`, which also beats a stray
   `LLAMA_ARG_N_GPU_LAYERS` env var.
2. **On the Vulkan build, `--n-gpu-layers 0` does not mean CPU compute.**
   `llama-bench` reports `backend: Vulkan` at ngl=0 and runs pp512 at
   275.94 tok/s: the weights stay in system RAM but the ops still execute on
   the GPU. The engine never selects the Vulkan build at ngl=0, so there is
   no user-visible exposure, but the bench's "CPU control on the vulkan
   build" row is not a thing that exists.
3. **An apparent 6x gap between the two releases' CPU paths was a
   misattribution, not a kernel regression.** It evaporated once (2) was
   understood: the comparison was CPU backend vs Vulkan backend. Both builds
   load the same CPU variant (`ggml-cpu-haswell.dll`) on this machine, which
   is the correct AVX2 variant for Kaby Lake.
4. **`bench_threads.py` could not produce a number at all.** Its worker
   unpacked two argv entries and the caller passed one, an arg-count
   mismatch present since the initial commit; every run died with an
   IndexError inside the worker and printed an empty row that looked like a
   measurement. The published thread table cannot have come from that script
   as committed. Fixed, and the table in the README and the article is
   re-measured.

## What ships

- `invoicelens/engine.py`: explicit `--n-gpu-layers 0` on the CPU path;
  `n_gpu_layers > 0` selects the Vulkan build automatically and refuses to
  run a GPU flag against a CPU-only build
- `invoicelens/cli.py`: `--n-gpu-layers` and `--device`, plus an engine line
  that prints the resolved configuration after load
- `tools/bench_backends.py`: prompt-vs-generation benchmark with
  `--list-devices`
- `tools/fetch_llama_cpp.py`: `--flavor vulkan` downloads the Vulkan build
  into a separate directory
- README: "GPU offload (opt-in)" section with the numbers above

## Reproduce

```bash
python tools/fetch_llama_cpp.py --flavor vulkan
python tools/bench_backends.py --list-devices
python tools/bench_backends.py --build vulkan --device vulkan1 --ngl 0 26
python -m invoicelens scan fixtures --out report --n-gpu-layers 26 --device vulkan1
```
