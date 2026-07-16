# Benchmark results

Captured runs of the `benchmark/` tools. Reproduce with the command under each
heading. Corpora: **english** = `alice_en`+`pride_en`, **wiki** = `wiki.test`
(a tractable sample), **words_en** = the ~466 k-word list.

> These are the peeler dictionaries for the distribution sweep and the four-track
> **combo** dictionary (+ bigram backstop) for everything else. Numbers are from
> a deterministic run; small drift is possible if the corpora change.

---

## 1. Parts by layer & type — `partsByLayerType.js`

`node benchmark/partsByLayerType.js` (frequency peeler; it scales to words_en).

Combined **affix (Start+Mid+End) counts by length** — affixes skew long, with
the 6/7 buckets the largest, and grow **strongly sub-linearly** with corpus size:

| length | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| english (340 k tok) | 609 | 538 | 907 | 1298 | 1932 | 1837 |
| wiki (473 k tok) | 1222 | 1281 | 1635 | 2269 | 3393 | 3701 |
| words_en (945 k tok) | 1517 | 2284 | 3393 | 5239 | 8357 | 10763 |

**Whole atoms by length** — for prose (english/wiki) they hump around 4–5 and
thin out (the "common words as shortcuts" behaviour); for the **word-list**
`words_en` (every word frequency 1, no prose signal) wholes spread monotonically
across all lengths — a useful reminder that a dictionary is not a corpus:

| length | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|
| english | 26 | 36 | 58 | 27 | 17 | 12 | 0 | 0 | 1 | 0 |
| wiki | 17 | 23 | 33 | 25 | 10 | 8 | 4 | 0 | 0 | 0 |
| words_en | 228 | 597 | 813 | 685 | 661 | 621 | 559 | 341 | 110 | 27 |

**Mid** is the dominant kind everywhere (interior substrings are most numerous);
total trained parts: english 7 300, wiki 13 623, words_en 36 954.

---

## 2. Single-letter filling — `letterFilling.js`

`node benchmark/letterFilling.js` (optimal min-letter DP, strict positions).

The **bigram backstop is decisive**. Overall filler rate:

| corpus | metric | no bigrams | +bigrams |
|---|---|---|---|
| english | by-token | 20.7 % | **5.6 %** |
| english | by-type | 62.1 % | **8.4 %** |
| english | chars as letters | 9.1 % | **1.3 %** |
| wiki | by-token | 22.5 % | **6.3 %** |
| wiki | by-type | 60.2 % | **15.1 %** |
| wiki | chars as letters | 9.1 % | **1.9 %** |

By **word length** (+bigrams), the residual is a **U-shape** — the structural
len-3 spike (a 3-char word can't tile with ≥2-parts) plus a slow rise in the long
tail (rarer words, max part length 7):

| length | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 15 |
|---|---|---|---|---|---|---|---|---|---|
| english % need ≥1 letter | 95 | 1 | 18 | 0 | 5 | 1 | 5 | 9 | 28 |
| wiki % need ≥1 letter | 98 | 29 | 27 | 4 | 7 | 2 | 4 | 10 | 31 |

Takeaway: with the fixed 676-bigram backstop only **~1–2 % of characters** are
lone letters; a "filler" is almost always a single atomic character (the `1` in
`1st`), not a coverage failure.

---

## 3. NN recovery under typos — `nnTypoSweep.js`

`node benchmark/nnTypoSweep.js` (wiki; deterministic corruption; recovery counts
only *genuinely corrupted* words).

**Substitution rate sweep** — recovery degrades gracefully:

| error % | recovery % | mean J |
|---|---|---|
| 0.5 | 77.7 | 0.642 |
| 5 | 75.2 | 0.605 |
| 10 | 68.9 | 0.565 |
| 15 | 59.8 | 0.532 |
| 20 | 52.1 | 0.496 |
| 25 | 43.0 | 0.465 |

**By edit type** (≈1 edit): insertion is easiest (adds signal), substitution and
transposition hardest (they destroy a bigram *and* create a wrong one).

| edit | recovery % | mean J |
|---|---|---|
| insertion | 90.0 | 0.694 |
| deletion | 65.4 | 0.664 |
| transposition | 61.2 | 0.573 |
| substitution | 60.6 | 0.541 |

---

## 4. Signature uniqueness & robustness — `signatureUniqueness.js` ★

`node benchmark/signatureUniqueness.js` (wiki). **The design-validating result.**

**(a) Uniqueness at rest** (collision-free % of distinct words):

| signature | unique % |
|---|---|
| **F** (containment) | **99.7 %** |
| [L, R] (boundaries) | 93.5 % |
| [L ∪ C] (GPT input) | 40.2 % |

`[L ∪ C]` is the *least* unique because it deliberately **excludes R** (the end
is the predicted/right side), dropping the discriminative suffix. `F` is unchanged
from before the 4-type refactor (single-char containment is position-partitioned,
same as the old positional letters). The peel's `[L,R]`/`[L∪C]` did move, because a
leftover boundary char now flows into `L`/`R` (a length-1 Start/End) rather than an
interior fill — so `[L,R]` gained uniqueness and `[L∪C]`, which drops R, lost it.

**Metric note.** All three signatures are id-sets, so they use one consistent
lens — **mean Jaccard** — measured two ways. F additionally reports **recovery %**
(its operational role as the retrieval key; F is *never* "unchanged" under an
edit, so for F the meaningful question is "does it still retrieve the right
word," not "is it identical").

**(b) RAW — intrinsic robustness** (encode the corrupted word directly; mean Jaccard):

| error % | F | [L,R] | [L∪C] |
|---|---|---|---|
| 0.5 | 0.985 | 0.974 | 0.972 |
| 5 | 0.858 | 0.768 | 0.752 |
| 10 | 0.749 | 0.609 | 0.594 |
| 15 | 0.650 | 0.485 | 0.458 |
| 20 | 0.558 | 0.370 | 0.350 |
| 25 | 0.479 | 0.287 | 0.275 |

The divergence **F > [L,R] > [L∪C]** is the design premise: the redundant
containment degrades slowest, the non-redundant peel fastest ([L∪C] worst, since
dropping R exposes it to more of the word). This is *why* F is the fuzzy key and
the peel must not carry robustness itself.

**(c) via-NN — the real pipeline** (θ=0.5; F = recovery %, peel = mean Jaccard on the canonicalized word):

| error % | F recovery % | [L,R] J | [L∪C] J |
|---|---|---|---|
| 0.5 | 98.8 | 0.990 | 0.990 |
| 5 | 87.5 | 0.898 | 0.900 |
| 10 | 75.1 | 0.794 | 0.794 |
| 15 | 60.9 | 0.668 | 0.660 |
| 20 | 47.1 | 0.544 | 0.539 |
| 25 | 33.9 | 0.424 | 0.420 |

Once canonicalized, the peel runs on the *same* corrected word, so `[L,R]` and
`[L∪C]` **converge on the recovery rate** — the NN lifts the brittle raw peel
(0.29 / 0.26 at 25 %) back up (0.41 / 0.40). Robustness is handled once, upstream.

**Collisions** (`node benchmark/collisions.js`, English novels for legibility):
the non-unique signatures are visible and intuitive.
- `[L,R]` groups words sharing **both boundaries** — `L=see R=∅ → {seen, seem, seek, sees}`, `L=hea R=∅ → {hear, heap, head, heal}`.
- `[L∪C]` collides **massively** because it drops R — `C={d##} → {dream, deal, done, dogs, death, dying, …}`, every word whose only structural content is an initial `d`. This is the concrete reason `[L∪C]` is just 57.6 % unique.

The tool prints each colliding word inside a real sentence (`⟦word⟧`) so the
differences are visible in context.

---

## 5. Chunk-contextual [L, C, R] — `chunkSignatures.js` ★

`node benchmark/chunkSignatures.js` (English novels). This is the **correct**
`[L,C,R]` model (§4 above measures a word's *internal* Start/Mid/End structure —
a different thing). For word `i` in a chunk:
`C_i = sparseF(w_i)`, `L_i = OR_{j<i} C_j`, `R_i = OR_{j>i} C_j`, `[L∪C]_i = OR_{j≤i} C_j`.

**`C` is the SPARSE peel, not the fuzzy F.** The two-stage encoding is: the
redundant fuzzy F is the *layer-0 NN recovery key* (typo → canonical word); the
signature content `C` is the **`bpeEncode` peel** of that canonical word — a tiny
id-set (`|C|` ≈ 1–2: `running → {ru, nning}`) versus the fuzzy F's ~17. Switching
`C` from fuzzy to sparse **barely moves the chunk numbers** (below), because chunk
collisions are driven by the *word-sequence* structure (order-invariance,
saturation), not by the per-word representation — and both encodings are ~injective
per word.

**Collision rates** (paragraph level, 155.5k positions):

- **C** is **100.0 % unique per distinct word** (of 7 978 words) — the sparse peel
  of distinct words is still essentially distinct. *Per token position* C is only
  2.1 % unique, but that is just common words repeating (`the`, `of`, …), **not** a
  signature collision.

| context signature | distinct | unique % (per position) |
|---|---|---|
| L | 106 127 | 50.2 |
| R | 108 962 | 52.3 |
| [L, R] | 144 818 | 88.0 |
| [L∪C] | 108 865 | 51.8 |
| **[L,C,R]** | 155 223 | **99.8** |

The full `[L,C,R]` triple is almost always unique; the one-sided pooled contexts
(`L`, `R`, `[L∪C]`) collide ~half the time. **Joining both boundaries — `[L,R]`
— jumps to 87.9 %**: a word's two-sided context is far more distinctive than
either side alone (it is `[L,C,R]` minus the saturating `C`). Sentence level is
higher across the board (shorter chunks → less saturation): L 63.8 %, R 66.9 %,
**[L,R] 97.1 %**, [L∪C] 67.7 %, [L,C,R] 99.8 %.

**Prediction consistency — the signatures that predict a word are judged by
whether the context *determines* its target, not by bare uniqueness.** `[L∪C]`
(the GPT input) predicts the **next** word; `[L,R]` (the cloze context) predicts
the **current** word. A collision only hurts when the same context maps to a
*different* target.

| context → target | determines target % | shares ctx % | …of which target agrees % |
|---|---|---|---|
| **[L∪C] → next word** (paragraph) | 51.7 | 48.9 | 1.2 |
| **[L∪C] → next word** (sentence) | 67.0 | 33.3 | 1.1 |
| **[L, R] → current word** (paragraph) | 88.1 | 12.1 | 1.2 |
| **[L, R] → current word** (sentence) | **97.2** | 2.9 | 3.5 |

Two things fall out:

1. **`[L,R]` is a near-deterministic cloze key** — the two-sided boundary context
   fixes the current word **88 % / 97 %** of the time. Where it collides it is
   genuine fill-in ambiguity (`…peeped into ⟦the⟧ book…` vs `…into the ⟦book⟧…`).
2. **`[L∪C]` alone doesn't determine the next word — but the ~1 % overstates
   it.** It fixes the next word 52 % / 67 % of the time; shared contexts agree
   ~1 %. Three effects are conflated, and only one is a pooling flaw. **(a)
   Intrinsic** — next-word is a *distribution*, not a function (`…gutenberg
   ⟦ebook⟧` precedes `11` **or** `1342`; no signature splits those). **(b)
   Missing position** — many shared-context collisions are pure order-invariance
   (`you know ⟦i⟧` ≡ `i know ⟦you⟧`). Self-attention is *itself*
   permutation-equivariant; a transformer restores order with a separate
   positional encoding. This probe omits position, so it **undercounts** `[L∪C]`
   exactly where a real model would separate the cases — order-invariance is not
   a mark against it. **(c) Saturation** — the one genuine pooling defect (see
   below). And no transformer predicts from the pool *alone*: the current token
   stays in the residual stream and attention is query-weighted, not a flat
   idempotent union. So read 52 % / 67 % as a floor for a *positionless,
   weightless, single-target* probe — not `[L∪C]`'s ceiling in a real model.

**Two properties of OR-pooled context — and only one is a real problem.**

1. **Order-invariance is NOT a defect.** `[L∪C]` is a bag, so `you know ⟦i⟧` ≡
   `i know ⟦you⟧` (71 paragraph / 160 sentence cross-chunk groups). But raw
   self-attention is *itself* permutation-equivariant — the attention op is a
   set reduction over value vectors; a transformer recovers word order from a
   *separate* positional encoding added to each token, not from the aggregation.
   So a bag-pool is **feature-parity with attention**, and the fix is the one
   transformers already use: carry position alongside the pool (a causal index or
   a positional part) and the order collisions vanish. On its own this should not
   be counted against `[L∪C]`.
2. **Saturation IS the real divergence.** Boolean union is monotone and
   *unnormalized*: over a long prefix it climbs toward "all parts" and can
   neither shrink nor re-weight (biggest group: 28 positions collapsed in one
   paragraph; ~27 k groups — the dominant collision cause). Softmax attention is
   a *normalized convex combination* — it stays bounded and can remain
   selective/peaked at any context length. This is where OR departs from
   attention. A bounded/decaying/normalized pool — or query-weighting, i.e.
   actual attention — is the lever; plain OR is not it.

**Is the saturation actually costing predictions? Collisions by prefix length.**
Break every next-word prediction out by how much prefix the predictor has
(sentence level; "harmful" = colliding positions whose next word differs):

| prefix before target | positions | collide % | harmful % of collisions |
|---|---|---|---|
| 1 word | 6 629 | 96.1 | 98.9 |
| 2 words | 6 499 | 65.6 | 94.8 |
| 3 words | 6 383 | 25.4 | 90.7 |
| **4–5 words** | 12 219 | **7.8** | 89.8 |
| 6–8 words | 16 397 | 10.0 | 99.1 |
| 9+ words | 100 620 | **34.5** | 99.9 |

The collide rate is a **U**: high at 1–2 words (too little context — *nothing*
predicts there, so the collision is free), bottoms out at **7.8 % by 4–5 words**,
then climbs back to 34.5 % at 9+ words — the saturation regime. But that tail is
almost all *within-chunk*: demanding a genuinely damning case — prefix ≥ 5 words,
union **not** saturated, collision spanning **different chunks** — yields **12
groups out of 117 275 contexts (~0.01 %)**, and every one is *also* intrinsically
unpredictable (a number after `ebook`, a chapter-heading artifact, a passage
quoted twice). So unbounded `[L∪C]` never collapses a context a competent
predictor would have split; its low determinism is intrinsic entropy + sentence
starts + saturation, **not** discarded signal.

**Fix — cap the context to a sliding window** (the OR-pool analogue of local
attention; `buildChunkSignatures(words, { dict, window })`). Since positions with
< `w` words of prefix already span their whole prefix, the window only touches the
saturation regime:

| variant | determines next word % |
|---|---|
| unbounded `[L∪C]` | 67.0 |
| window = 3 words | 68.1 |
| **window = 5 words** | **88.7** |
| window = 7 words | 88.8 |

**`window = 5` is the knee** — it lifts next-word determinism **67 → 88.7 %**
(landing on the `[L,R]` cloze figure, 88.1 %), and 7 adds nothing. Collisions by
prefix length confirm it is free: the < 5-word buckets are **byte-identical** to
unbounded, 6–8 words drops 10.0 → 4.6 %, and the 9+ tail collapses **34.5 → 3.9 %**.
This is the "suck it up below five words, cap above" design: nothing lost where no
predictor could win, saturation removed everywhere else.

**Clamping both sides (symmetric `window = 5`).** The sweep above caps only the
causal `[L∪C]`. Applying the same `window = 5` to `L` and `R` too improves *every*
predictive metric and *every* one-sided uniqueness. Note the reach: `window = 5`
gives the causal `[L∪C]` a 5-word span (current + 4 preceding), so symmetrically
`L` and `R` each reach **`w − 1 = 4`** words — the current word is the 5th slot —
i.e. a 4 + ⟦word⟧ + 4 = 9-word cloze frame.

| metric (paragraph) | unbounded | window = 5 both sides |
|---|---|---|
| L uniqueness | 50.1 | 86.1 |
| R uniqueness | 52.2 | 87.7 |
| [L,R] uniqueness | 87.9 | 99.4 |
| [L∪C] uniqueness | 51.7 | 92.2 |
| [L,C,R] uniqueness | 99.8 | 99.5 ↓ |
| **[L∪C] → next word** | 51.7 | **93.3** |
| **[L,R] → current (cloze)** | 88.1 | **99.9** |

(Sentence level: `[L∪C] → next` 67.0 → 88.7, `[L,R] → cloze` 97.2 → 99.7,
`[L,C,R]` uniqueness 99.8 → 99.7.)

The feared context-loss never materialises — cloze `[L,R]` *rises* to **99.9 %**,
because unbounded `L`/`R` were themselves saturating; a tight 9-word frame pins the
centre word almost perfectly (the residual 0.1 % is genuine fill-in ambiguity).
Both one-sided halves stop saturating (`L`/`R` ~50 → 86 %). The **only** regression
is `[L,C,R]` as a whole-position fingerprint (99.8 → 99.5): the far context that
separated distant look-alike positions is gone. But that is *position identity*,
not prediction — a metric the model never uses — so it is a ~0.3-pt loss on the
wrong axis for a ~40-pt gain on the two right ones.

**±5 frame and boundary compensation (`window = 6`, shift-pad).** Bumping to
`window = 6` gives the true symmetric **±5** frame (`[L∪C]` spans 6 words; `L`/`R`
reach 5 each). Two refinements measured at radius r=5 (paragraph):

| metric | unbounded | truncated (window=6) | shift-pad r=5 |
|---|---|---|---|
| [L,R] uniqueness | 87.9 | 99.6 | 99.7 |
| [L,C,R] uniqueness | 99.8 | 99.7 ↓ | **99.8** ✔ |
| [L,R] → cloze | 88.1 | 99.9 | 99.9 |
| [L∪C] → next word | 51.7 | 93.8 | n/a (causal) |

- **±5 barely beats ±4 on next-word** — 93.3 → 93.8 (paragraph), 88.7 → 89.1
  (sentence). The causal knee is flat across 5–6 words; 5 already captures it.
- **Shift-pad** keeps the *total* context at `2r` words by extending the far side
  when one side is short near a chunk boundary (instead of truncating). It
  **recovers the `[L,C,R]` regression** (99.7 → 99.8, back to unbounded): the
  boundary positions that had lost distinguishing context get it back from the
  opposite side. Cloze is already at ceiling, so no further gain. Strictly ≥
  truncation but small here — only chunks longer than `2r+1 = 11` words have
  boundary positions to compensate.
- **Shift-pad is a per-phase setting, not a global one.** The bidirectional
  encoder (`phase: 'bidirectional'`) has the right context for real, so shift-pad
  is on there. The causal generator (`phase: 'causal'`) cannot borrow the future,
  so it stays truncated. These are *different objects at different stages of the
  stack*, so there is **no `[L∪C] = L ∪ C` invariant to keep** — `[L,R]` and
  `[L∪C]` were never the same signature. The library ships this as the default:
  radius-5 shift-pad for the bidirectional phase, radius-5 causal window for the
  next-word phase (`src/core/parts/signature/`).

> **Bug fixed here:** the sparse `or`/`xor` `&&`/`||` chain double-processed a
> **0** value — a branch whose comma-sequence ended on `out[k++]=v` returns `v`,
> and `v===0` is falsy, so it fell through to the next `||` and re-ran. Part id 0
> is the first dictionary entry, so real signatures hit it; the old tests (all
> ids ≥ 1) never did. Fixed by ordering the `++i`/`++j` increment **last** in
> each branch so the branch value is always truthy — the terse chain stays,
> with 0-value regression tests added.
