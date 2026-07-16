# Encoding stage

Turns a word into its two representations over a part dictionary.

| Function | Output | Purpose |
|---|---|---|
| `bpeEncode(dict, word, tracks?)` | `{ whole, L, C, R, parts }` | **Structural** code — a strict, boundaries-first, discriminative-before-generative greedy peel. Non-redundant; feeds the `[L, C, R]` signature. |
| `fuzzyEncode(dict, word)` | sorted `Uint32Array` (F) | **Fuzzy** code — the full containment signature. Redundant and typo-robust; the key indexed by the normalizer. |
| `addBigramBackstop(dict)` | `dict` | Seeds the fixed 676-bigram positional backstop (length-2 coverage layer). |

Single characters are **not** a separate kind: they are length-1 `Start`/`Mid`/`End`
fragments (bare value; position is the kind), or a `Whole` when the char is the
word. There is no `letterValue`/`Kind.Letter` — see [`positionalKind`](../kind.js).

## Two codes, two jobs

- The **peel** (`bpeEncode`) is non-redundant so the pooled `L`/`R` components stay clean and discriminative. It runs on *canonical* words (post-normalization), so it does not need to be typo-robust. Priority: boundaries before mid → discriminative before generative → longest → leftmost, with the bigram backstop filling gaps and a length-1 Start/Mid/End fragment for odd parity.
- The **containment signature** `F` (`fuzzyEncode`) is deliberately redundant: a one-character edit only perturbs the ids overlapping it, so Jaccard(F, F′) degrades gracefully. It is the fuzzy retrieval key for canonicalization.

## Track tags

`bpeEncode` takes an optional `tracks` map (`"kind|value" → "d"|"g"`) emitted by
the [generation stage](../generation/README.md) so the peel can prefer
**discriminative** parts over **generative** ones on the boundaries. Untagged
parts (the bigram backstop) sort last.
