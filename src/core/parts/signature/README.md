# Signature stage

The **chunk-contextual** `[L, C, R]` signature. A chunk is a word sequence (a
sentence or a paragraph). `buildChunkSignatures(words, ctx)` returns one entry
per word:

```
C_i     = F(word_i)              the word's own containment signature
L_i     = OR_{j<i}  C_j          pooled signatures of the PRECEDING words
R_i     = OR_{j>i}  C_j          pooled signatures of the FOLLOWING words
[L∪C]_i = OR_{j<=i} C_j          the causal prefix-union (the GPT-side input)
```

Each is a sorted `Uint32Array` of part ids, pooled by the sparse set-union
([`or`](../../math/sparse/or.js)). The per-word `F` comes from
[`fuzzyEncode`](../encoding/fuzzyEncode.js); pass an `index` in `ctx` to
canonicalize each word first (typo/OOV → nearest known word) so `C` is the clean
signature.

## Two properties of OR-pooled context

- **Order-invariant** — `[L∪C]` is a *bag* of signatures, so `you know i` and
  `i know you` collide (same prefix word-set). This is not a defect: raw
  self-attention is itself permutation-equivariant and recovers order from a
  separate positional encoding.
- **Saturation** — in long chunks the *unbounded* union eventually covers ~all
  parts, so late positions become indistinguishable. This is the real divergence
  from softmax attention (a normalized, bounded pool).

## Phases (`ctx.phase`)

`[L,R]` and `[L∪C]` are **not the same object** — they are different stages of the
stack with different constraints, so they are pooled differently. Windowing removes
the saturation above, and the *direction* it may pool is what distinguishes the
phases:

| phase | who | produces | window | shift-pad |
|---|---|---|---|---|
| `'bidirectional'` | BERT-style encoder | `L`, `C`, `R` | radius `r` (default 5) | **on** (R is real) |
| `'causal'` | GPT-style generator | `L`, `C`, `LC` (`R` = ∅) | `r` preceding (default 5) | off (R = future) |
| `'raw'` *(default)* | analysis / back-compat | `L`, `C`, `R`, `LC` | unbounded, or `ctx.window` | — |

```js
buildChunkSignatures(words, { dict, phase: "causal", radius: 5 });        // [L∪C], last 6 words
buildChunkSignatures(words, { dict, phase: "bidirectional", radius: 5 }); // ±5 cloze frame
```

**Shift-pad is a per-phase setting, not a global property.** In the bidirectional
encoder the right context is legitimately available, so shift-pad is on: near a
chunk boundary the window slides inward to keep the full `2r` context instead of
letting the short side atrophy — recovering the boundary uniqueness a plain clamp
loses. In the causal generator the right context is the *forbidden future*, so
shift-pad is off (borrowing rightward would leak the target). Because the two
phases are different objects there is **no cross-phase invariant** to hold.

Empirically: causal next-word determinism ~67 % (unbounded) → **~89 %** at radius
5, flat past there; bidirectional cloze → **~99.9 %** once neither side saturates,
with shift-pad recovering the last of the whole-position `[L,C,R]` uniqueness.

`benchmark/signature/chunkSignatures.js` measures all of it: per-signature collision rates
(`C` / `L` / `R` / `[L,R]` / `[L∪C]` / `[L,C,R]`), next-word/cloze prediction
consistency, the window sweep + collision-by-prefix-length, and truncated-vs-shift-pad.

> The **word-internal** decomposition (Start/Mid/End) lives in the
> [encoding stage](../encoding/README.md) (`bpeEncode`). It is *not* `[L,C,R]` —
> it is how a single word is peeled; the chunk signature here pools whole-word
> `F` across the chunk.
