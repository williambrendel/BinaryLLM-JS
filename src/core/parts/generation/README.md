# Generation stage

Builds the part **dictionary** — the four-track combo — from a corpus word
frequency map. `generateParts(freqMap, opts)` returns `{ dict, tracks, stats }`.

| Track | Function | What it contributes |
|---|---|---|
| 1. Wholes | `prefilterWholes` | 1–2 letter words (all of them) + the frequency head of longer lengths — atomic tokens the affix tracks can't or shouldn't decompose. |
| 2. Generative | `generativeAffixes` | The **obvious** common morphemes: frequency head per (length, position), elbow-gated. Fills the interior stems the tree skips (`form`, `graph`, mid-`tion`). |
| 3. Discriminative | `discriminativeTree` | The balanced-split, **word-separating** parts (decorrelated). L = 7→3. |
| 4. Atoms | `augmentWithAtoms` | Letters / connectors / delimiters, regenerated at load. |

The length-2 **bigram backstop** is deliberately *not* here — it's a fixed,
corpus-independent layer added by the [encoder](../encoding/README.md).

## Why two learned tracks

Generative and discriminative are **complementary**, not redundant (~11–13 %
overlap measured). Generative guarantees the obvious common parts; discriminative
adds the parts that actually separate words. The `tracks` map tags each learned
part `w`/`d`/`g` so the encoder can prefer discriminative parts on the boundaries.

See `benchmark/` for the head-to-head against the frequency-peeler.
