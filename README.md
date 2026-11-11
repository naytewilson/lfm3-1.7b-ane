# LFM3-1.7B (pruned, 4-bit) on the Apple Neural Engine — early release

Core ML packages of a width-pruned, 4-bit palettized LFM3-1.7B that decode one token per call
on the Apple Neural Engine (ANE), with the measurements, hardware evidence and hashes that
back every number on this page.

**Status: early release (v0.1.0).** Placement and speed are measured. Output quality is
measurably below the unquantized model (numbers below), the reference runtime is not part of
this release, and several things a production deployment needs have not been measured. Read
[What is not established](#what-is-not-established) before relying on it.

Not affiliated with or endorsed by Liquid AI. LFM3 is Liquid AI's model; this is a derivative
distributed under the [LFM Open License v1.0](LICENSE).

## Headline numbers

All measured on one machine: MacBook Neo (`Mac17,5`), Apple A18 Pro, 8 GB, macOS 27.3
(build `26C5035r`), on 2026-11-10 (evening, US Eastern; timestamps in the evidence are UTC
2026-11-11).

| | 64-position bundle | 288-position bundle |
|---|---|---|
| Operations planned on the ANE | 1582 of 1583 | 1582 of 1583 |
| Decode speed from the median step, tokens per second (3 runs) | 37.8 / 39.7 / 38.6 | 37.3 / 35.0 / 35.2 |
| Decode speed from the mean step, tokens per second (3 runs) | 35.9 / 37.6 / 35.6 | 35.8 / 31.9 / 33.1 |
| Median time per token, ms (3 runs) | 26.45 / 25.19 / 25.89 | 26.82 / 28.60 / 28.42 |
| Stored weights on the ANE | 791 MB in 2 functions | 791 MB in 2 functions |
| Peak resident memory of the runner process (not the system memory holding the model, which was not measured) | 75–82 MB | 97–112 MB |

Each run decodes for about 24 seconds (783–923 steps) and the benchmark reports the median
step; the mean row is total steps divided by total time in the same runs. The speed rows are from one controlled session (03:33Z–03:40Z) with the machine at thermal
state "fair" and a one-minute load average of 5–7. The same bundles measured earlier that evening:

| Session (UTC) | Runtime | 64 positions, tok/s (median step) | 288 positions, tok/s (median step) |
|---|---|---|---|
| 01:17Z–01:42Z | first | 34.5 / 34.0 / 32.3 | 29.7 / 29.4 / 29.6 |
| 02:04Z–02:07Z | current | 43.6 / 42.7 / 43.1 | 37.7 / 36.1 / 37.2 |
| 03:33Z–03:40Z | current | 37.8 / 39.7 / 38.6 | 37.3 / 35.0 / 35.2 |

The "current" runtime differs from the "first" only in how the host carries state between
calls; its logits are bit-identical to the first runtime's over 64 positions (max-abs
difference 0.0). The spread between the two "current" sessions is machine state, not a
change in the artifact: figures are comparable only within a session. No measurement was
taken on a cool, idle machine with the current runtime.

## What "on the ANE" means here

Placement is what Core ML's `MLComputePlan` reports for each compiled function on this OS
build, recorded per operation in `evidence/*/seg0{1,2}_dev.oracle.json`:

| Function | Layers | Stored weights | Ops | ANE-preferred | CPU | GPU |
|---|---|---|---|---|---|---|
| `seg01_dev` | 0–15 | 445 MB | 1035 | 1035 | 0 | 0 |
| `seg02_dev` | 16–23, final norm, output head | 346 MB | 548 | 547 | 1 | 0 |

- The single CPU operation is the last `reshape` that produces `logits`; Core ML reports it
  as CPU-only. Everything before it, including the output head, is ANE-preferred.
- The token-embedding lookup (`gather` over a 128,000 × 2,048 table) is not in either
  function. It runs on the host CPU by design and measures 0.04 ms per token.
- So: **1582 of 1583 operations (99.94%)** in the compiled functions are planned on the ANE.
  It is not 100%.
- This is the planner's assignment. This release contains no hardware-interval trace of
  these bundles, so physical dispatch is inferred from the plan and the timings, not
  separately observed.
- Core ML's built-in model state could not be used: both state forms produced no execution
  plan (error `-14`) on this OS build. The key/value caches and convolution windows are therefore
  explicit inputs and outputs, carried by the host between calls (35 states: 22 attention
  caches updated by row copy, 13 convolution windows). That carry costs about 0.1 ms per
  token.

## The model

| | |
|---|---|
| Upstream | `LiquidAI/LFM3-1.7B`, revision `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd` |
| Architecture | 24 layers: 13 gated short convolutions (width 7), 6 full-attention, 5 sliding-window attention (window 256); hidden size 2,048; 16 heads, 4 key/value heads; vocabulary 128,000; tied input/output embedding |
| Modification 1: pruning | Feed-forward width cut from 7,680 to 7,168 in all 24 layers. 1,728,801,536 → 1,653,304,064 parameters (−4.4%). 72 tensors pruned, 214 copied unchanged. Receipt: `evidence/PRUNING_RECEIPT.json` |
| Modification 2: palettization | 143 weight tensors stored as 4-bit indices into lookup tables, one table per 8 rows, built by deterministic k-means. No calibration data, no fine-tuning |
| Modification 3: conversion | Compiled to Core ML `mlprogram` packages; weights and constants rounded to IEEE binary16; causal mask uses −10000.0 in place of −∞ |
| Pruned checkpoint SHA-256 | `92bc789529bbaf1584efd0e8c10a5d5550648531b06391c90e68f71684e9e2cb` (3,306,640,496 bytes, BF16) |
| Compiler | `anec` 0.2.0 (not public) |

## Output quality

Quality was measured against a teacher: the pruned checkpoint itself, run in 32-bit float
by an independent implementation, over a fixed 270-token passage. The bundle was fed the
passage token by token on the ANE and its logits compared with the teacher's at every
position.

| Measure | 288-position bundle, all 270 positions | 64-position bundle, first 64 positions |
|---|---|---|
| Same top-1 next token as the teacher | 251 / 270 (93.0%) | 45 / 64 (70.3%) |
| Mean KL divergence from the teacher | 0.240 | 0.955 |
| Logit cosine with the teacher | 0.859 | 0.808 |
| Perplexity on the passage (teacher) | 1.983 (1.644) | 17.80 (8.11) |
| Passage's actual next token is top-1 (teacher) | 246 / 269 (250 / 269) | 40 / 63 (44 / 63) |

Read plainly:

- The passage becomes repetitive after its opening, which is why the 270-position scores are
  much better than the first-64 scores. The first 64 positions are the harder and more
  representative part: there the 4-bit model picks a different next token than the teacher
  about 30% of the time.
- The loss comes from 4-bit palettization, not from the conversion. Before palettization the
  compiled program matches the teacher over 24 positions at logit cosine 1.0, max-abs
  difference 0.0018, top-1 24/24, with every layer's output within 0.0002 relative L2.
- A 6-bit build of the same model (not in this release) scored 263/270, KL 0.044 and
  perplexity 1.687 on the same passage, at 24–27 tokens per second and 1,192 MB.

## What is not established

- **Quality of the pruned checkpoint against the original LFM3-1.7B: not measured.** Every
  accuracy number above compares against the pruned checkpoint, not against Liquid AI's
  model. The pruning was not followed by any recovery training.
- **No task or chat evaluation.** Only next-token agreement on one 270-token passage.
- **No long-run or thermal measurement.** Each speed run repeats a 28-step decode (13
  prompt tokens, then generation) from an empty cache for about 24 seconds. The longest
  single decode timed was 270 steps, on the first runtime (33.4 ms per token median, 29.9
  tokens per second). Nothing was run for minutes, and nothing at a full cache.
- **No energy measurement, no CPU or GPU comparison** for this model.
- **One device, one OS build.** ANE placement is decided by the OS at load time and can
  differ on other chips or OS versions.
- **Fixed context.** A bundle holds at most 64 or 288 positions including the prompt; it
  does not slide or extend. The upstream model's 256K context is not available here.
- **One token per call.** The prompt is ingested at the same speed as generation; batched
  prefill was not built.
- **No reference runtime in this release.** See below.

## What is in this release

In this repository:

| Path | Contents |
|---|---|
| `evidence/c64/`, `evidence/c288/` | For each bundle: `fit.json` (state-form and partition search against the hardware), `certificate.json`, `plan.json`, `semantic.json`, per-operation placement evidence (`seg0*_dev.oracle.json`), and three benchmark sessions (`bench-*/results.tsv`, `receipt.json`, `guard.log`, and the raw per-step record of each run, `ane_{1,2,3}.json`) |
| `evidence/logs/` | Job logs of the teacher check, the accuracy runs and the three timing sessions |
| `evidence/PRUNING_RECEIPT.json` | What the pruning changed |
| `tokenizer/` | Tokenizer, chat template and model configuration from the upstream release |
| `SHA256SUMS` | SHA-256 of every release asset and of the files inside the bundles |
| `LICENSE`, `NOTICE.md` | LFM Open License v1.0 and the statement of changes it requires |

As release assets:

| Asset | Contents |
|---|---|
| `lfm3-1.7b-pruned7168-lut4-c64.tar` | `seg01_dev.mlpackage`, `seg02_dev.mlpackage`, `plan.json`, `certificate.json`, `semantic.json` for a 64-position context |
| `lfm3-1.7b-pruned7168-lut4-c288.tar` | The same for a 288-position context |
| `embed_tokens.bf16.bin` | The token-embedding table for the host-side lookup: 128,000 × 2,048, little-endian bfloat16, row-major, 524,288,000 bytes |

The packages published here were rebuilt from the pruned checkpoint on 2026-11-11 and are
byte-identical to the ones measured: for both functions of both bundles, the SHA-256 of
`model.mlmodel` and of `weight.bin` equals the hash recorded in the hardware evidence, and
`certificate.json`, `plan.json` and `semantic.json` are identical files. The two bundles
share the same weight files; only the compiled program differs with the context size.

## Running it

**The runtime used for the measurements (`anec-run`) is not included.** What is published is
enough to load the functions with Core ML and to write a driver, but there is no turnkey
command yet.

One decode step, as recorded in `plan.json`:

1. Look up the token's row in `embed_tokens.bf16.bin` and convert it from bfloat16 to
   float16 → `embed` (shape `[1, 2048]`). No scaling is applied.
2. Call `seg01_dev` with `embed`, the rotary tables and masks for the current position, and
   the caches of layers 0–15. It returns the hidden state `L15_res2` and, per layer, the
   fresh cache rows.
3. Call `seg02_dev` with `L15_res2`, the same position inputs, and the caches of layers
   16–23. It returns `logits` (shape `[1, 128000]`) and the fresh rows.
4. Write each fresh row into its cache at the current position (attention layers), or
   replace the window with the returned one (convolution layers). `plan.json` → `carried`
   lists all 35 states with shapes and the exact update for each; `plan.json` → `step`
   gives the rotary frequencies, mask value, context size and sliding window.

Load both packages with `MLModelConfiguration.computeUnits = .cpuAndNeuralEngine`. The first
load of each function on a machine includes on-device compilation and took 17.6 s and
18.0 s here; later loads took 0.4 s and 0.1 s.

## Reproducing the checks

- `shasum -a 256 -c SHA256SUMS` after unpacking confirms the assets.
- `evidence/*/seg0*_dev.oracle.json` → `artifact.modelSHA256` / `weightSHA256` bind the
  placement evidence to the exact package files; `plan.summary` holds the counts quoted
  above and `plan.ops` the per-operation rows.
- `certificate.json` records the SHA-256 of the pruned checkpoint, of `config.json` and of
  each of the 286 source tensors, the 213 rewrites applied, and for each palettized tensor
  its error against the fp16 source.
- `evidence/*/bench-*/results.tsv` holds the three timing runs of each session with wall
  time, CPU time, major faults and peak resident memory; `receipt.json` holds the verdict;
  `ane_N.json` holds every step time (`stepMS`), the step count and the total seconds.

## License

The model weights, tokenizer and configuration are derivative works of Liquid AI's
LFM3-1.7B and are distributed under the **LFM Open License v1.0** (see `LICENSE`). In
short, and not as a substitute for the text: use, modification and redistribution are
permitted with the license and notices kept; commercial use is licensed only for entities
with annual revenue under US $10 million. `NOTICE.md` states what was changed.
