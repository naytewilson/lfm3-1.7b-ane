# Notice of changes

This repository and its release assets contain derivative works of **LFM3-1.7B** by
Liquid AI, Inc. (`LiquidAI/LFM3-1.7B`, revision
`40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`), distributed under the LFM Open License v1.0
(see `LICENSE`). They are not produced, reviewed or endorsed by Liquid AI.

The following files are modified from the upstream work:

| File | Change |
|---|---|
| `seg01_dev.mlpackage`, `seg02_dev.mlpackage` (both bundles) | Model weights with the feed-forward width pruned from 7,680 to 7,168 in all 24 layers; 143 tensors palettized to 4-bit lookup tables (one table per 8 rows, deterministic k-means); all weights and constants rounded to IEEE binary16; converted to Core ML `mlprogram` format and split into two functions; causal mask value −10000.0 in place of −∞ |
| `tokenizer/config.json` | `intermediate_size` changed from 7680 to 7168 |

The following files are unmodified copies of upstream files:

| File | |
|---|---|
| `embed_tokens.bf16.bin` | The tensor `model.embed_tokens.weight`, extracted byte for byte from the pruned checkpoint (the pruning receipt reports 72 feed-forward tensors pruned and the other 214 copied unchanged) (SHA-256 `5f1df3bbc564554f48a7fb542f96e745c92e2315c92279944d9e65d234bec4bc`) |
| `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`, `tokenizer/special_tokens_map.json`, `tokenizer/chat_template.jinja`, `tokenizer/generation_config.json` | |
| `LICENSE` | |

Everything under `evidence/` and this repository's `README.md` are original to this
release.
