# Benchmarks & analysis

These are **not** unit tests — they are exploratory/measurement tools that run
the pipeline on real corpora and print numbers to compare approaches and check
that the design behaves as expected. All logic lives in `src/`; these scripts
only orchestrate + report. Run them from the repo root.

**➡ Captured numbers with interpretation live in [`RESULTS.md`](./RESULTS.md).**

**Layout** (mirrors `src/core/`): `signature/` (signature collisions & uniqueness) ·
`parts/` (part extraction, encoding, normalization) · `phase1/` (the deployed StQP+AdaBoost
extractor CV) · `pipeline/` (end-to-end).

| tool | measures |
|---|---|
| `comparePartMethods.js` | combo generator vs. frequency-peeler (track complementarity) |
| `partsByLayerType.js` | part counts per **layer × type** across corpora (`*_en`, `wiki`, `words_en`) |
| `letterFilling.js` | single-letter filler rate by length; greedy vs optimal DP; bigram backstop on/off |
| `nnTypoSweep.js` | NN recovery under typos — rate sweep (0.5–25 %) and by edit type |
| `chunkSignatures.js` | ★ the **chunk-contextual [L,C,R]** — collision rates (incl. `[L,R]`), next-word/cloze **prediction consistency**, OR-pooling effects, **context-cap (window) sweep** |
| `signatureUniqueness.js` | robustness of a word's *internal* structure (peel) under injected error |
| `collisions.js` | non-unique *word-internal* signatures — colliding word groups in real sentences |
| `pipelineBench.js` | end-to-end demo + basic self-match / typo-recovery / coverage |
| `partsStats.js` | part distribution + short-substring space coverage for one dict |

Corpora used below: **English** = `data/corpora/alice_en.txt` +
`pride_en.txt` (~156 k tokens, ~8 k word types); **Wiki** =
`data/corpora/wiki.test.txt` (~204 k tokens, ~20 k types — a tractable sample;
`wiki.train.txt` is 535 MB and out of scope for these tools); **words_en** =
`data/corpora/words_en.txt` (~466 k word list).

---

## `comparePartMethods.js` — combo generator vs. frequency-peeler

```
node benchmark/parts/comparePartMethods.js data/corpora/alice_en.txt data/corpora/pride_en.txt
```

Builds the four-track **combo** dictionary (`generateParts`) and the **peeler**
(`PartExtractor`) on the same corpus and reports part counts by (kind × length),
vocabulary overlap, and — the key result — the **complementarity of the two
learned tracks**.

**Expected (English):** wholes ≈ 438, generative ≈ 1 986, discriminative ≈ 3 474,
with **gen ∩ disc only ≈ 524 (~11 %)** — i.e. the generative and discriminative
tracks contribute mostly *distinct* parts. The combo learned-vocab (~5.3 k) is
leaner than the peeler (~7.3 k) but recovers the common interior stems
(`form`, `graph`, mid-`tion`) the pure discriminative tree drops. No learned
parts at length 2 (bigrams are the fixed backstop).

**Why it matters:** validates that generative (coverage) + discriminative
(separation) are worth running together, not redundant.

---

## `pipelineBench.js` — end-to-end signature + normalizer benchmark

```
node benchmark/pipeline/pipelineBench.js data/corpora/wiki.test.txt
```

Builds the full pipeline (generate → bigram backstop → fuzzy index) and reports:

1. **Encoder demo** — `[L, C, R]` for a handful of words (`running → ru·nning`,
   `the → WHOLE`, `1st → ·1##·st`).
2. **Fuzzy typo-robustness** — Jaccard(F, F′) for word/typo pairs. **Expected:**
   ~0.56–0.74 for one-edit typos, ~0.26 for unrelated words — typos stay close,
   unrelated stay far.
3. **Normalizer** — **self-match ≈ 99.7 %** (known words return themselves at
   J≈1, so in-vocab words are never corrupted) and **typo-recovery ≈ 85 %**
   (typos resolve to the original; misses land on a close sibling).
4. **OOV gate** — the Jaccard distributions of *typo→correct* and
   *word→neighbor* **overlap** (both ~0.5–0.7). Takeaway: the gate is a
   "snap if close enough (θ≈0.5), else keep" threshold, **not** a typo detector.
   In-vocab protection comes from self-match, not the gate.
5. **BPE coverage** (English) — ~1.4 parts/word; ~9 % of tokens touch a lone
   single-char fragment (odd-parity leftovers), ~98 % of characters carried by
   real ≥2-length parts.

---

## Stage READMEs

Each pipeline stage documents its own design under `src/core/parts/<stage>/README.md`:
[generation](../src/core/parts/generation/README.md) ·
[encoding](../src/core/parts/encoding/README.md) ·
[normalization](../src/core/parts/normalization/README.md) ·
[signature](../src/core/parts/signature/README.md).
