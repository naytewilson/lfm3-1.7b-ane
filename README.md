# LFM3-1.7B (pruned, 4-bit) on the Apple Neural Engine — early release

Core ML packages of a width-pruned, 4-bit palettized LFM3-1.7B that decode one token per call
on the Apple Neural Engine (ANE), with the measurements, hardware evidence and hashes that
back every number on this page.

**Status: early release (v0.1.0).** Placement, decode and prefill speed, ANE power and
real chat turns are measured. Output quality is measurably below the unquantized model (numbers below), the
runtime is a prebuilt binary without source, and several things a production deployment
needs have not been measured. Read [What is not established](#what-is-not-established)
before relying on it.

Not affiliated with or endorsed by Liquid AI. LFM3 is Liquid AI's model; this is a derivative
distributed under the [LFM Open License v1.0](LICENSE).

## Current build: q1 weights, program revision 2 (added 2026-11-11)

Same weights as the q1 bundle below, byte for byte (both `weight.bin` files have the same
SHA-256). What changed is the program around them, in two places, and the runner:

1. **Range lifts.** At 58 places a value is multiplied by a power of two (4 to 128 on the
   operators' output projections, 2 to 8 on the MLPs' output projections, 8 to 32 on the
   attention q/k projections) before a projection, and the norm that follows takes its
   epsilon times the square of that factor. `RMSNorm(g·y, g²·eps) = RMSNorm(y, eps)`, so the
   function is the same in exact arithmetic. On the device the projection's wide sum and
   the norm's mean of squares then work on values well inside the 16-bit normal range.
2. **SiLU and sigmoid written with `tanh`**: `sigmoid(v) = (1 + tanh(v/2)) / 2` and
   `silu(v) = v · sigmoid(v)`, as ordinary operations instead of the ANE's own kernels.
3. **Runner r3 takes text.** A native tokenizer, the chat layout, sampling and a multi-turn
   chat are in the runner; Node is no longer needed to run a prompt. Runner r4 adds longer
   context (next section).

With those two program changes the ANE gives almost exactly what exact arithmetic gives:

| On the ANE, 270-token teacher passage | Revision 2 | q1, first program | First (k-means) weights |
|---|---|---|---|
| Same top-1 token as the teacher | **253** | 252 | 251 |
| Mean KL from the teacher | **0.117** | 0.188 | 0.240 |
| Logit cosine with the teacher | **0.942** | 0.924 | 0.859 |
| First 64 positions | 47 of 64, KL 0.446 | 46 of 64, KL 0.711 | 45 of 64, KL 0.955 |
| Perplexity on the passage (teacher 1.644) | 1.816 | 1.911 | 1.983 |
| The same weights in exact arithmetic (reference interpreter) | 253 of 270, KL 0.1125 | the same | not run |
| ANE against that exact program: same top-1, mean KL | **270 of 270, 0.0004** | 261 of 270, 0.054 | not run |
| With the prefill hand-off, against the teacher | 253 of 270, KL 0.117 | 251 of 270, KL 0.189 | not run |
| With against without prefill, same top-1 | 270 of 270 | 269 of 270 | not run |
| Operations planned on the ANE (plan level, both functions of the bundle) | 1791 of 1792; prefill 1973 of 1974 | 1582 of 1583; prefill 1764 of 1765 | as q1 |
| Decode, three guarded runs of 63–64 s (mean step) | 43.1 / 42.1 / 42.6 tok/s, one-minute load average 9–10 | 40.2 / 40.9 / 42.0 tok/s, load average 8–17 | 34.0 / 32.4 / 29.7 tok/s, load average 35–56 |
| ANE power in those runs | 1.45 / 1.33 / 1.32 W, 31–34 mJ per token | 1.31 / 1.28 / 1.30 W, 31–33 mJ per token | 1.13 / 1.07 / 0.93 W, 31–33 mJ per token |

What stands behind the revision 2 column (`evidence/q1-p2-20261111/`):

- **Where the device lost accuracy was measured, value by value.** Every intermediate value
  of a layer was made a program output and compared on the ANE and on CPU against a 32-bit
  reference (8 probe tokens, layers 0 to 5; `numerics/base.table.txt` and
  `numerics/lift.table.txt`). Relative error, layer 0:

  | Value | ANE before | ANE with lifts | CPU-only |
  |---|---|---|---|
  | Input of the operator's output projection | 0.1–0.2 % | 0.1–0.2 % | 0.3–0.8 % |
  | Output of that projection | 1.9–7.2 % | 0.2–0.3 % | 0.3–0.6 % |
  | MLP gate, before SiLU | 2.5–5.4 % | 0.1–0.4 % | 0.5–0.7 % |
  | After SiLU | 4.6–7.4 % | 4.0–5.2 % | 0.5–0.8 % |
  | Stream after the layer | 4.3–7.8 % | 2.8–3.6 % | 0.8–1.1 % |

  After six layers the stream was 21–35 % off before, 7–9 % with lifts, 2–3 % on CPU. The
  first jump is a 2048-wide sum of small 16-bit products feeding a norm whose epsilon
  (1e-5) is below the 16-bit normal range. The second is the ANE's SiLU kernel.
- **Each change was then measured on the whole model** (same passage, single runs,
  `numerics/evals.log`, `eval3.log`): mean KL of the ANE from the exact program 0.054 with
  neither, 0.0069 with lifts on the operator outputs only, 0.0063 with all lifts, 0.0004
  with all lifts and the `tanh` form. Writing SiLU as `v · sigmoid(v)` with the ANE's
  sigmoid gave output bit-identical to its SiLU and was dropped (`eval4.log`).
- **The factors come from data, not from this passage.** One 32-bit reference pass over 513
  tokens of chat text (the four chat records already in this repository) recorded each
  value's peak; a factor is the largest power of two that keeps the value the norm sees at
  or below 32 and the lifted value at or below 2048 (`range.json`).
- **Speed is unchanged, measured side by side.** Six guarded 30-second runs each,
  alternating between the two programs in blocks of three: the median step time, averaged
  over the six runs, is 23.66 ms for the first program and 23.74 ms for revision 2, at
  one-minute load average 8–42 (`numerics/bench.log`). That is 42.3 and 42.1 tokens per
  second; the difference is inside the spread between runs (23.2–24.0 and 23.1–24.6 ms).
- **The verifier accepts both functions**: 11 checks pass, 0 fail, 1 skipped, including
  "artifact equals source program bit for bit" and 143 palettes recomputed with zero error.
  Its first run failed that bit-for-bit check: the artifact computes SiLU by the `tanh`
  formula and the source program says SiLU. The verifier was then given its own rule for
  the identity, and the same package files (unchanged hashes) pass.
- **The host interface is unchanged.** `plan.json` of both functions is the same file as
  in the q1 bundle (same SHA-256): same carried states, same step inputs.
- **This is still agreement with a teacher on one passage, not a quality verdict**, and no
  run was on a settled machine, so the speeds are observations.

Chat turns with revision 2, run from the packaged files with runner r3 (single runs,
reasoning block closed, greedy unless noted; `evidence/q1-p2-20261111/packaged-run/`):

| Prompt | Prompt tokens | Time to first new token | Rate | Reply |
|---|---|---|---|---|
| What is the capital of France? | 18 | 1.05 s in the first run of the session, 0.48 s later | 43–47 tok/s | The capital of France is **Paris**. It is the largest city in France and the country's political, economic, and cultural center. |
| The 105-token note to summarize (96 tokens read by the prefill function) | 105 | 1.21 s | 42.6 tok/s | The city council approved a six-month trial of new bike lanes on Main Street, after public comments and a debate between supporters and shop owners, with a decision to review data at the end. |
| Write one sentence about the sea. (temperature 0.8, top-k 40, seed 7) | 18 | 0.56 s | 41.9 tok/s | The sea flows with a rhythm that never stops, whispering secrets to those who listen. |
| Second turn of a chat: And what river runs through it? | 18 more | 0.46 s | 43.2 tok/s | The River **Seine** (or the **Sena**) flows through Paris. It is one of the major rivers in France and historically played a key role in the city's development. |
| Third turn: Name one museum there. | 16 more | 0.42 s | 39.1 tok/s | One museum in Paris is the **Louvre Museum** (Musée du Louvre). It is one of the world's most famous art museums. |

"Explain RoPE in two sentences" is still answered wrongly (it invents an expansion of the
acronym), as with every earlier build. The runner's tokenizer and chat layout produce the
same token ids as the Hugging Face tokenizer used before for all four prompts checked (18,
21, 17 and 105 tokens).

## Longer context: 1024 and 2048 positions (added 2026-11-11)

The same weights and the same program revision, compiled for 1024 and for 2048 positions,
and runner r4, which starts in the 288-position programs and moves into the larger ones
when it runs out of room (`--grow`). A larger context costs speed on every token, so a run
only pays for it once it needs it:

| Context | Median step, three guarded 30 s runs | Tokens per second | Peak memory of the runner process | Planned on the ANE |
|---|---|---|---|---|
| 288 | 22.8 / 23.2 / 23.0 ms (60 s runs) | 43.9 / 43.1 / 43.5 | 69–80 MB | 1791 of 1792 |
| 1024 | 29.3 / 29.1 / 28.8 ms | 34.1 / 34.3 / 34.7 | 135 MB | 1791 of 1792 |
| 2048 | 38.4 / 38.8 / 37.5 ms | 26.0 / 25.8 / 26.7 | 231 MB | 1791 of 1792 |

(One-minute load average 8 when the 1024 runs started and 45 when the 2048 runs started:
observations on a loaded machine.) On the 270-token teacher passage all three contexts give
the same figures as the 288 one: 253 of 270, KL 0.117. The verifier accepts both new
functions (11 pass, 0 fail, 1 skipped), and they use the same two weight files.

What moving between contexts does, and how it was checked (`evidence/q1-p2-context-20261111/`):

- **Nothing is recomputed.** Each attention cache gets a larger array holding the rows
  already written; the convolution windows carry over as they are. The switch took
  0.8–1.1 s here once the larger functions had been loaded on the machine before.
- **Against exact arithmetic, across the boundary.** 513 tokens of chat text were fed
  through (a) 288 growing into 1024, with the prefill function reading the first 256, and
  (b) 1024 from the first token. Same top-1 token as the 32-bit reference at 508 and 511 of
  513 positions (mean KL 0.0006 both); (a) and (b) agree with each other at 508 of 513; in
  (a), positions 288 to 512 alone agree with the reference at 224 of 225. This compares the
  device with the same weights computed exactly, not with the unquantized teacher: beyond
  the 270-token passage there is no teacher comparison.
- **A reply of 1,500 tokens** crossed both boundaries without a break (`jobG.log`): median
  step 24.9 ms below 288, 28.3 ms up to 1024, 40.5 ms beyond, 29.5 tokens per second over
  the whole reply. The text reads fluently, repeats itself and gets facts wrong: this is
  about the mechanism, not the content.
- **From the packaged files** (`packaged-run/`): a 420-token reply that crosses 288 (first
  token after 1.22 s, switch 1.0 s, 28.1 tokens per second overall) and a five-turn chat
  that reaches position 484 (later turns: first token after 0.58–0.73 s, 28–33 tokens per
  second).
- **The first use of a context on a machine takes about 30 s** while the OS compiles it for
  the ANE. In the first packaged run that pause fell in the middle of a reply
  (`jobV-before-preparation.log`); runner r4 now does it once at start and says so.

## q1 weights, first program (2026-11-11)

The bundles named `q1` carry a second set of 4-bit weights for the same pruned model:
palettes made with an error-compensating method (GPTQ) for this project's earlier per-layer
build, moved into this two-function build unchanged. The first bundles on this page use
plain k-means palettes and stay as the baseline. Both sets compile to the same programs, so
placement, size and structure are identical; only the weight values differ. The q1 bundle
is still what you download for the weights; run it with the revision 2 programs above.

**Correction (2026-11-11).** This table first gave "47 of 64, KL 0.315" for q1's first 64
positions. That figure came from a comparison run with the wrong row width and was wrong.
The value below is from a rerun of the same bundle (`evidence/q1-p2-20261111/numerics/evals.log`,
variant `v0-none`, which reproduces the 270-position figures exactly; the first 64 positions
are its first two position ranges together).

| | q1 weights | First (k-means) weights |
|---|---|---|
| Same top-1 token as the teacher over 270 positions, on the ANE | 252 | 251 |
| Mean KL from the teacher, on the ANE | 0.188 | 0.240 |
| Logit cosine with the teacher, on the ANE | 0.924 | 0.859 |
| First 64 positions, on the ANE | 46 of 64, KL 0.711 | 45 of 64, KL 0.955 |
| Perplexity on the passage (teacher 1.644) | 1.911 | 1.983 |
| The same program in exact arithmetic (reference interpreter) | 253 of 270, KL 0.1125 | not run |
| Operations planned on the ANE | 1582 of 1583; prefill 1764 of 1765 | the same |
| Decode, three runs of 60–66 s | 40.2 / 40.9 / 42.0 tok/s (mean step), load average 8–17 | 34.0 / 32.4 / 29.7 tok/s in an earlier session, load average 35–56 |
| ANE power in those runs | 1.31 / 1.28 / 1.30 W, 31–33 mJ per token | 1.13 / 1.07 / 0.93 W, 31–33 mJ per token |

What stands behind the q1 column (`evidence/q1-combined-20261111/`):

- **The move is exact.** The verifier recomputed all 143 palettes of the q1 bundle from the
  converted checkpoint with zero error and found the bundle equal to its source program bit
  for bit (12 checks pass, 0 fail, 1 skipped).
- **It is the same model as in the earlier build.** In exact arithmetic the q1 program
  scores 253 of 270 and KL 0.1125; the earlier build's own record for q1 is 253 of 270 and
  KL 0.113.
- **The ANE costs q1 accuracy with this first program:** KL 0.1125 in exact arithmetic,
  0.188 on the device. The earlier build holds 0.113 on the device by scaling weights so
  that 16-bit products stay out of the subnormal range. Program revision 2 (above) closes
  the gap in this build a different way, without touching the weights: 0.117 on the device.
  A first attempt that kept the earlier build's weight scales without adjusting the norms'
  epsilon made the model worse (KL 0.207 in exact arithmetic) and was dropped.
- **With the prefill hand-off:** 251 of 270 and KL 0.189 against the teacher; 269 of 270
  against the run without prefill.
- **The two speed cells are not a comparison.** They come from different sessions under
  different load. The programs are the same, so no difference is expected, and none was
  measured head to head.
- **This is agreement with a teacher on one passage, not a quality verdict.** The earlier
  build's frozen evaluation set has not been run on this bundle.

Four chat turns with q1, run from the packaged files with the published runner (single
runs; reasoning block closed, greedy):

| Prompt | Time to first new token | Reply |
|---|---|---|
| What is the capital of France? | 1.03 s (first run of the process; the first call alone took 0.57 s) | The capital of France is **Paris**. It is the country's largest city and its political, economic, and cultural center. |
| Explain in one sentence why the sky is blue. | 0.64 s | The sky is blue because it scatters incoming sunlight, with shorter wavelengths (like blue) being scattered more efficiently by the molecules in the atmosphere, making the sky appear blue. |
| The 105-token note to summarize, with prefill | 0.74 s | The city council approved a six-month trial of new bike lanes on Main Street, after public comments and a two-hour discussion, with a decision to proceed despite concerns about parking for shop owners. |
| Explain RoPE in two sentences | 0.50 s | Wrong again: it expands RoPE as "Rotating Predicted" and had not finished at 128 tokens. |

Three of the four are sound; the RoPE answer is wrong with both sets of weights.

## Headline numbers

By this project's evidence standard a speed counts as a measurement only on a settled
machine (one-minute load below 4). No run on this page met that, so every speed here is an
observation on a loaded machine, with the load stated beside it.

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

## Chat turns, sustained decode and ANE power (2026-11-11)

The published runner and the files downloaded from this release were run on the same
machine on 2026-11-11, 19:06Z–19:13Z. The machine was busy with unrelated work the whole
time (load average 35–56, CPU drawing about 3.9 W before the test started), so read the
rates as what a loaded machine gave, not as the best this hardware does. Everything is in
`evidence/chat-20261111/`.

**Runner check.** Fed the first 64 tokens of the test passage, the published runner with the
downloaded 288-position bundle picks the same next token as the 2026-11-10 run at 64 of 64
positions.

**Sustained decode with ANE power.** 288-position bundle, three runs of about 64 seconds
each through the benchmark guard (all valid), each run repeating full 270-token passes.
Power is the `ane_power` reading of macmon 0.10.0 (IOReport, no root), sampled once a
second and averaged over the run. It read 0.00 W before and after.

| Run | Steps | Seconds | Tokens/s (mean) | Tokens/s (median step) | ANE power | ANE energy per token |
|---|---|---|---|---|---|---|
| 1 | 2,159 | 63.5 | 34.0 | 34.5 | 1.13 W | 33.3 mJ |
| 2 | 2,159 | 66.6 | 32.4 | 33.1 | 1.07 W | 32.9 mJ |
| 3 | 1,889 | 63.7 | 29.7 | 32.3 | 0.93 W | 31.4 mJ |

This is the ANE's own reading, not wall power. The runner's CPU and memory power could not
be separated from the other work on the machine and are not reported. Peak resident memory
of the runner in these runs was about 309 MB, against 97–112 MB in the short runs above;
the likely cause is that the runner keeps the logits of all 270 positions of a pass.

**Chat turns.** Chat template with the reasoning block closed, greedy decoding, up to 128
new tokens, 288-position bundle. These are single runs outside the benchmark guard, so
treat the timings as indicative.

| Prompt | Prompt tokens | Time to first new token | New tokens | Generation rate | Whole turn |
|---|---|---|---|---|---|
| What is the capital of France? | 18 | 0.53 s | 36, then stopped | 34.7 tok/s | 1.54 s |
| Explain in one sentence why the sky is blue. | 21 | 0.67 s | 47, then stopped | 35.0 tok/s | 1.99 s |
| Explain RoPE in two sentences | 17 | 0.54 s | 128, cut off | 34.2 tok/s | 4.25 s |

Times start at the first prompt token and exclude loading the model. Loading took 0.1–0.7 s
with the compiled model cached and 32 s the first time on this machine (14.1 s + 17.9 s for
the two functions).

The replies, verbatim:

> The capital of France is **Paris**. It is located on the river Seine in the northern part
> of the country and is one of the most well-known cities in the world.

> The sky is blue because sunlight is reflected by the molecules in the atmosphere, and the
> specific wavelengths of light that are scattered by these molecules (mainly nitrogen and
> oxygen) to the eye are in the blue part of the spectrum.

> **RoPE** stands for **Realientation of Weights** (or **Parameter-wise **) and refers to a
> technique used in **Transformer** models, particularly in **Large Language Models
> (LLMs)**, to **normalize the weights** of the model's parameters. […]

The first two are sound. The third is wrong: RoPE is rotary position embedding, and the
reply invents an expansion and had not finished at 128 tokens. Whether the original or the
6-bit model gets it right was not tested.

**Prompt speed in these turns.** The prompt went through the same one-token-per-call path
as generation, at 31–34 tokens per second (the first call of a turn took 67–83 ms). These
prompts are shorter than 32 tokens, so the prefill function below would not change them.

## Prefill function (2026-11-11)

A second compile of the same 4-bit model that reads 32 tokens per call, at the 288-position
context, for getting through a prompt quickly. Built and measured on the same machine on
2026-11-11, 19:21Z–19:29Z (load average 6–9). Everything is in `evidence/prefill-20261111/`.

| | |
|---|---|
| Operations planned on the ANE | 1764 of 1765. The exception is again the final reshape that produces the logits |
| Weights | The same two weight files as the one-token 288-position bundle (identical SHA-256). The package is 1.7 MB and links to them |
| Agreement with the one-token function, both on the ANE, same 256 tokens | Same top-1 token at 255 of 256 positions; logit cosine 0.99945; mean KL 0.00045 |
| Agreement with the teacher over those 256 positions | 236 of 256, mean KL 0.253. The one-token function scores 237 of 256 and 0.253 |
| Independent verifier (`verify-20261111.log`) | 12 checks pass, 0 fail, 1 skipped (the same "device executed every segment" check). 8,192,000 output elements and all 35 carried states bit-identical to the source program over 2 chained calls |

Speed and ANE power: three runs of about 60 seconds through the benchmark guard (all
valid), each repeating 8 calls of 32 tokens over the first 256 tokens of the test passage.
Power was sampled as in the section above and read 0.00 W before and after.

| Run | Calls | Tokens | Seconds | Tokens/s (mean) | Tokens/s (median call) | ANE power | ANE energy per token |
|---|---|---|---|---|---|---|---|
| 1 | 503 | 16,096 | 60.0 | 268.3 | 266.6 | 1.89 W | 7.0 mJ |
| 2 | 487 | 15,584 | 60.2 | 259.0 | 260.3 | 1.75 W | 6.8 mJ |
| 3 | 503 | 16,096 | 60.8 | 264.8 | 263.6 | 1.86 W | 7.0 mJ |

A call takes about 121 ms once warm; the first call after loading took 254–270 ms. Peak
resident memory of the runner was 546–592 MB in these runs.

### Prefill joined to generation (runner r2)

The second runner build (`lfm3-ane-runner-r2-macos-arm64.tar`) takes `--prefill`: the
prefill function reads the prompt in whole calls of 32 tokens, then the one-token function
continues from the same cache for the rest of the prompt and for the reply. Measured on the
same machine on 2026-11-11, 19:47Z–19:51Z; everything is in
`evidence/prefill-chain-20261111/`.

| Check | Result |
|---|---|
| r2 without `--prefill` against the first runner, 270 tokens of the test passage | Same next-token pick at 270 of 270 positions |
| r2 with `--prefill` against without, same 270 tokens (8 prefill calls, then 14 one-token calls) | Same pick at 269 of 270 positions; logit cosine 0.99946; mean KL 0.00043. The one difference is at position 13, inside the prefill part. All 14 positions after the hand-off are identical |

A chat turn with a 105-token prompt (a short news note to summarize in one sentence), 64
new tokens, three runs each way. These are single runs outside the benchmark guard, so
indicative:

| | Without `--prefill` | With `--prefill` |
|---|---|---|
| How the prompt is read | 105 one-token calls | 3 calls of 32 tokens, then 9 one-token calls |
| Time to the first new token | 3.38 / 3.35 / 3.44 s | 1.36 / 1.01 / 1.07 s |
| Generation rate after that | 31.0 / 33.7 / 32.6 tok/s | 28.1 / 29.1 / 33.9 tok/s |

The first prefill call of a process is slow (0.24–0.65 s against about 0.15 s for the later
ones), which is the spread in the right-hand column.

**The reply is not the same with and without prefill.** Both start with the same 9 tokens
and then choose different words:

> The city council approved a six-month trial of proposed bike lanes on Main Street, after a
> public comment period and a vote approving a trial starting in March, with a review of
> data on accidents and sales. *(without prefill)*

> The city council approved a six-month trial of new bike lanes on Main Street, after a
> public comment period, with a review of data on safety and sales. *(with prefill)*

Both are sound summaries. The prefill calls compute the same cache rows in a different
batch shape, the ANE's 16-bit arithmetic rounds them slightly differently, and greedy
decoding turns a small difference into a different word once two candidates are close.

The remaining limits of the prefill function:

- **Prompts of 32 tokens or fewer gain nothing.** At least one prompt token always goes
  through the one-token function, and the rest of the prompt is read in whole calls of 32.
- **It computes more than a prompt needs.** It produces logits for all 32 positions, though
  only the last matters when reading a prompt. Of the 121 ms per call, about 99 ms is the
  second function, which holds the output head. A variant without the head was not built.
- **No guarded benchmark of a chained turn.** The runner refuses `--prefill` together with
  its sustained-timing mode, so the turn timings above are single runs.

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

## Agreement with the teacher (first weights)

This is parity evidence, not a quality verdict: every figure compares with a teacher on one
passage. The numbers in this section are for the first (k-means) weights; the q1 numbers
are in the table at the top. Agreement was measured against a teacher: the pruned checkpoint itself, run in 32-bit float
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
- **No per-position data from the test passage in this repository.** The passage's token
  ids and the per-position picks were removed from the current files on 2026-11-11 (they
  remain in the git history), so that the passage can still be used blind elsewhere. Only
  aggregates are kept.
- **Revision 2's device accuracy rests on 59 fitted factors.** They were fitted on 513
  tokens of chat text. Text whose values peak higher than in that sample runs closer to the
  16-bit limit; the bound leaves about a factor of 8 in amplitude before a square
  overflows, and what happens beyond it was not tested. Larger factors were not tried on
  hardware either. With the first program (no factors) the device costs q1 KL 0.188
  against 0.1125 in exact arithmetic.
- **No task or chat evaluation.** Next-token agreement on one 270-token passage, and a
  handful of chat prompts, one of which every build answers wrongly.
- **No long-run or thermal measurement.** The 2026-11-10 speed runs repeat a 28-step decode
  (13 prompt tokens, then generation) from an empty cache for about 24 seconds; the
  2026-11-11 runs repeat 270-token passes for about 64 seconds. Nothing was run for
  minutes, and nothing with the cache completely full.
- **ANE power only.** No wall or whole-system power, and no CPU or GPU comparison for this
  model.
- **One device, one OS build.** ANE placement is decided by the OS at load time and can
  differ on other chips or OS versions.
- **Context up to 2048 positions, not beyond, and not tested full.** With `--grow` a run
  can move from 288 to 1024 to 2048 positions; it does not slide past that, and the
  upstream model's 256K context is not available here. The longest run made reached
  position 1542; nothing was run with 2048 positions completely used. Accuracy past
  position 270 is checked against exact arithmetic on one 513-token sequence only.
- **Prefill can change the reply with the first programs.** There, with `--prefill` the
  reply can differ in wording from the one produced without it (shown in the prefill
  section). With revision 2 the two paths picked the same top-1 token at all 270 positions
  of the test passage; a chat reply with and without it was not compared.
- **The runtime is a binary only.** `anec-run` is published without source. Runner r3
  takes text, prints the reply as it is produced, can sample with a seed and keeps a
  multi-turn chat, and r4 adds `--grow`; there is no server, and a chat ends when the
  largest context given is used up. It was run only on the machine above. r3 and r4 were
  built from changes that were still under review in their private repository when they
  were published.

## What is in this release

In this repository:

| Path | Contents |
|---|---|
| `evidence/c64/`, `evidence/c288/` | For each bundle: `fit.json` (state-form and partition search against the hardware), `certificate.json`, `plan.json`, `semantic.json`, per-operation placement evidence (`seg0*_dev.oracle.json`), and three benchmark sessions (`bench-*/results.tsv`, `receipt.json`, `guard.log`, and the raw per-step record of each run, `ane_{1,2,3}.json`) |
| `evidence/logs/` | Job logs of the teacher check, the accuracy runs and the three timing sessions |
| `evidence/chat-20261111/` | The 2026-11-11 runs: the three chat records and their token ids, the runner check, the 64-second benchmark runs, the raw power samples (`macmon.jsonl`), the job script and log, and the two small scripts that compute the tables above |
| `evidence/prefill-20261111/` | The prefill function: its fit record, certificate, plan and per-operation placement evidence, the two comparison runs, the 60-second benchmark runs, the raw power samples, the verifier log, the job script and log, and the script that computes the table |
| `evidence/prefill-chain-20261111/` | Prefill joined to generation: the three 270-token comparison records, the six chat records and their token ids, the job script and log, and the script that computes the tables |
| `evidence/q1-combined-20261111/` | The q1 bundle: fit records, certificates, plans and placement evidence for both functions, the verifier log, the carry report (palette hashes and the scale found for every tensor), the exact-arithmetic comparison by position range, the benchmark runs with power samples, four chat records, and the job scripts and logs |
| `evidence/q1-p2-20261111/` | Program revision 2: fit records, certificates, plans and placement evidence for both functions, the two verifier logs, the fitted factors (`range.json`), the comparison with the teacher (`jobA.log`), the 60-second benchmark runs with power samples, the chat records made with the text front end, the run from the packaged files (`packaged-run/`), and under `numerics/` the value-by-value device comparison, the per-variant results, the side-by-side timing and the scripts that produced them |
| `evidence/q1-p2-context-20261111/` | Longer context: fit records, certificates, plans, placement evidence and verifier logs for the 1024- and 2048-position functions, their benchmark runs, the comparison across the boundary against exact arithmetic (`jobG.log`, `grow_a.json`, `grow_b.json`), the 1,500-token reply, and the runs from the packaged files before and after the start-time preparation |
| `evidence/PRUNING_RECEIPT.json` | What the pruning changed |
| `tokenizer/` | Tokenizer, chat template and model configuration from the upstream release |
| `tools/tok.mjs` | Turns a chat message into token ids and token ids back into text |
| `SHA256SUMS` | SHA-256 of every release asset and of the files inside the bundles |
| `LICENSE`, `NOTICE.md` | LFM Open License v1.0 and the statement of changes it requires |

As release assets:

| Asset | Contents |
|---|---|
| `lfm3-1.7b-pruned7168-q1-p2-c288.tar` | Program revision 2, programs only (3.3 MB): the one-token function (`lfm3_q1p2_C288`) and the 32-token prefill function (`lfm3_q1p2_C288_prefill32`), each with `plan.json`, `certificate.json`, `semantic.json`; `link-weights.sh` links in the q1 bundle's weight files and checks their SHA-256; `range.json` holds the fitted factors |
| `lfm3-1.7b-pruned7168-q1-p2-c1024-c2048.tar` | Program revision 2 for 1024 and 2048 positions, programs only (3.1 MB): `lfm3_q1p2_C1024` and `lfm3_q1p2_C2048`, each with `plan.json`, `certificate.json`, `semantic.json`; `link-weights.sh` links in the q1 bundle's weight files |
| `lfm3-ane-runner-r4-macos-arm64.tar` | The current runner: r3 plus `--grow`. Use this one |
| `lfm3-ane-runner-r3-macos-arm64.tar` | The third runner: the same package as r2 with an `anec-run` that takes text (`--prompt`, `--tokenizer`, `--chat`, sampling options). Kept because the revision 2 runs used it |
| `lfm3-1.7b-pruned7168-q1-c288.tar` | The q1 weights with the first program, 288-position context, same contents as the bundles below. Unpacks to `lfm3_q1_C288`. Revision 2 needs it for the weight files |
| `lfm3-1.7b-pruned7168-q1-c288-prefill32.tar` | The 32-token prefill function for the q1 bundle (1.7 MB; `link-weights.sh` links in the q1 bundle's weight files). Unpacks to `lfm3_q1_C288_prefill32` |
| `lfm3-1.7b-pruned7168-lut4-c64.tar` | First (k-means) weights. `seg01_dev.mlpackage`, `seg02_dev.mlpackage`, `plan.json`, `certificate.json`, `semantic.json` for a 64-position context |
| `lfm3-1.7b-pruned7168-lut4-c288.tar` | The same for a 288-position context |
| `embed_tokens.bf16.bin` | The token-embedding table for the host-side lookup: 128,000 × 2,048, little-endian bfloat16, row-major, 524,288,000 bytes |
| `lfm3-1.7b-pruned7168-lut4-c288-prefill32.tar` | The 32-token prefill function for the 288-position context: the two compiled programs, `plan.json`, `certificate.json`, `semantic.json`, and `link-weights.sh`, which links in the weight files from the unpacked 288-position bundle and checks their SHA-256 |
| `lfm3-ane-runner-r2-macos-arm64.tar` | The second runner: the same package as below with an `anec-run` that accepts `--prefill` (token ids in, token ids out). Kept because the prefill hand-off and q1 runs used it |
| `lfm3-ane-runner-macos-arm64.tar` | The first runner, kept because the 2026-11-11 chat and power runs used it. The `anec-run` binary (arm64, built for macOS 15 or later, ad-hoc signed, not notarized), `model/config.json`, and `setup.sh` with the 112-byte header that turns the embedding table into the one-tensor `model.safetensors` the runner reads |

The packages published here were rebuilt from the pruned checkpoint on 2026-11-11 and are
byte-identical to the ones measured: for both functions of both bundles, the SHA-256 of
`model.mlmodel` and of `weight.bin` equals the hash recorded in the hardware evidence, and
`certificate.json`, `plan.json` and `semantic.json` are identical files. The two bundles
share the same weight files; only the compiled program differs with the context size.

The compiler's independent verifier was run on the published 64-position bundle on
2026-11-11 against the pruned checkpoint and the hardware evidence
(`evidence/c64/verify-20261111.log`): **12 checks pass, 0 fail, 1 skipped.** It recomputed
all 143 palettes from the source tensors (worst relative L2 error of a palettized tensor
against its fp16 source: 0.130), found the packages equal to the source program bit for bit
on inputs it chose (256,000 output elements and all 35 carried states over 2 chained
calls), and recomputed the placement as 1582/1583 from the per-operation rows. The skipped
check is "the device executed every segment": no ANE-versus-CPU output comparison was
recorded for it. The verifier was not run on the 288-position bundle today.

## Running it

You need an Apple-silicon Mac, this repository cloned, and five assets downloaded into it:
`lfm3-1.7b-pruned7168-q1-c288.tar` (the weights), `lfm3-1.7b-pruned7168-q1-p2-c288.tar`
(the current programs), `lfm3-1.7b-pruned7168-q1-p2-c1024-c2048.tar` (the same programs for
longer contexts; optional), `embed_tokens.bf16.bin` and `lfm3-ane-runner-r4-macos-arm64.tar`.

```sh
# 1. unpack, link the programs to the weights, assemble the model folder
tar -xf lfm3-1.7b-pruned7168-q1-c288.tar
tar -xf lfm3-1.7b-pruned7168-q1-p2-c288.tar
tar -xf lfm3-1.7b-pruned7168-q1-p2-c1024-c2048.tar
tar -xf lfm3-ane-runner-r4-macos-arm64.tar
./lfm3_q1p2_C288/link-weights.sh lfm3_q1_C288
./lfm3_q1p2_C1024/link-weights.sh lfm3_q1_C288
./lfm3-ane-runner/setup.sh embed_tokens.bf16.bin
xattr -d com.apple.quarantine lfm3-ane-runner/anec-run 2>/dev/null || true   # needed if a browser downloaded it

# 2. one prompt, on the Neural Engine
./lfm3-ane-runner/anec-run lfm3_q1p2_C288 --model lfm3-ane-runner/model \
    --prefill lfm3_q1p2_C288_prefill32 --tokenizer tokenizer/tokenizer.json \
    --prompt "What is the capital of France?"

# 3. a chat: the first turn on the command line, further turns typed in (/exit or Ctrl-D ends it).
#    --grow lets it continue past 288 positions, into 1024 and then 2048.
./lfm3-ane-runner/anec-run lfm3_q1p2_C288 --model lfm3-ane-runner/model \
    --prefill lfm3_q1p2_C288_prefill32 --tokenizer tokenizer/tokenizer.json \
    --grow lfm3_q1p2_C1024,lfm3_q1p2_C2048 \
    --prompt "What is the capital of France?" --chat 1
```

What to expect:

- The first run on a machine compiles the four programs for the ANE and took about 55 s
  here; later runs load in under a second.
- The reply is printed as it is produced and stops at the end-of-turn token or after
  `--max-tokens` (default 256). A line on stderr then gives the prompt tokens, the time to
  the first new token, the rate and the position reached in the context.
- Options: `--system "text"` adds a system message; `--temperature T --top-k K --seed S`
  sample instead of taking the most likely token; `--repetition-penalty P`; `--think 1`
  leaves the model's reasoning block open; `--raw 1` feeds the text as given, without the
  chat layout; `--stats 1` also prints the step tables; `--out reply.json` writes the full
  record, including the token ids and the reply text.
- Prompt, replies and all chat turns together must fit the largest context given: 288
  positions, or 2048 with `--grow` as above. The runner says so and ends the chat when the
  next turn would not fit. With `--grow`, the first start on a machine prepares each larger
  context once (about 30 s each here); moving into one later takes about a second, and
  every token after that is slower (see the longer-context table).
- `--prefill` is used for whole blocks of 32 prompt tokens of the first turn; shorter
  prompts and later turns are read one token at a time. It can be left out.
- `--units cpu` runs the same functions without the ANE.
- Token ids still work: `--tokens "1,2,3" --generate 128` instead of `--prompt`, as with
  the earlier runners, and `tools/tok.mjs` still converts between text and ids.
- Steps 1 to 3 were run on the machine above from the published archives and the downloaded
  embedding table (`evidence/q1-p2-20261111/packaged-run/` with runner r3, and
  `evidence/q1-p2-context-20261111/packaged-run/` with runner r4 and `--grow`), with two
  differences: the weight files were linked from a local copy of the q1 packages rather
  than from the unpacked q1 archive (`link-weights.sh` checked them against the published
  SHA-256), and the chat turns were piped in rather than typed. Nothing was run on another
  machine.

For the earlier builds use their assets and folder names (`lfm3_q1_C288` with
`lfm3_q1_C288_prefill32` from `…-q1-c288-prefill32.tar`, or the `lut4` ones); runner r2 or
r3 runs them.

To time the prefill function on its own:

```sh
IDS32=$(node tools/tok.mjs encode --raw "any text long enough to give at least 32 tokens ..." | cut -d, -f1-32)
./lfm3-ane-runner/anec-run lfm3_q1_C288_prefill32 --model lfm3-ane-runner/model \
    --tokens "$IDS32" --generate 0 --units ane
```

`--tokens` must then hold a multiple of 32 ids. The same command was run on the machine
above with the first weights and the first runner, on 32 tokens of the test passage, and
gave the same 32 next-token picks as the measured run.

If you would rather write your own driver, one decode step, as recorded in `plan.json`:

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

- `shasum -a 256 -c --ignore-missing SHA256SUMS` after unpacking confirms the assets you
  downloaded; files from assets you did not download are skipped. The unlabelled
  `lfm3-ane-runner/anec-run` line is runner r4. Every runner unpacks to that same path, so
  the earlier ones are listed as `(r1)`, `(r2)` and `(r3)` for reference, and with one of
  those unpacked the unlabelled line reports a mismatch.
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

The `anec-run` binary is not a derivative of LFM3 and is not under that license. It is
published without source, as is and without warranty, for running these bundles.
`tools/tok.mjs` and the files under `evidence/` are original to this repository.
