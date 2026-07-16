# `core/parts/` — the sub-word part vocabulary

Words are decomposed into **parts** — sub-word units (morpheme-like fragments) that form the shared alphabet
of every signature bit. This folder builds the part vocabulary and decomposes text against it.

## Motivation

A dense per-word vocabulary is huge and can't generalize to unseen or misspelled words. Instead, a compact
**part dictionary** is learned once (the iterative *peel* algorithm), and any word is greedily decomposed into
a small sorted set of part IDs. Those IDs are the bits of the signature — so morphologically related words
share bits, and out-of-vocabulary words still decompose. This is the layer that turns raw text into the
integer bit-sets everything else consumes.

## Contents

| file | role |
|---|---|
| `tokenize.js` | stream tokenizer — raw text → Word / Delimiter tokens |
| `extractor.js` | builds a `PartDictionary` from observed words via the iterative **peel** algorithm |
| `dictionary.js` | the trained part vocabulary + text/binary serialization |
| `decomposer.js` | greedy decomposition of a word/delimiter into dictionary part IDs (the shared "peel") |
| `kind.js` | a part's positional role + the length bounds shared by extractor and decomposer |
| `byteString.js` | UTF-8 ⇄ latin1 byte-string conversion for byte-exact string ops |

## Sub-stages (each has its own README)

[`encoding/`](./encoding) · [`generation/`](./generation) · [`normalization/`](./normalization) ·
[`signature/`](./signature) — part encoding, candidate generation, normalization (nearest-neighbor / typo
tolerance), and signature construction.
