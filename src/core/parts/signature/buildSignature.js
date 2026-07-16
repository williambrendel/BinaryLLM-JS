"use strict";

/**
 * @file buildSignature.js
 * @brief Stage 4 — the chunk-contextual [L, C, R] signature.
 *
 * A chunk is a word sequence (a sentence or paragraph). Each word gets a per-word
 * signature `C`, pooled by set-union (OR) over the chunk:
 *
 *   C_i     = encode(word_i)         the word's own signature (see below)
 *   L_i     = OR of the PRECEDING words' C
 *   R_i     = OR of the FOLLOWING words' C
 *   [L∪C]_i = OR of C over words <= i (the causal prefix-union; GPT-side input)
 *
 * **The encoder is the configuration** (a first-input choice to evaluate
 * end-to-end — the whole `[L,C,R]` uses ONE, never a mix). NN recovery is *not* a
 * separate knob; it is intrinsic to the sparse pipeline:
 *
 *   - `ctx.encoder = 'fuzzy'` — `C` is the redundant containment F of the *raw*
 *     word (~19 ids). Self-robust: a typo only perturbs the few parts overlapping
 *     it, so it needs NO recovery. Blurry under loose matching, but more overlap
 *     for an attention layer to work over.
 *   - `ctx.encoder = 'sparse'` (default) — `C` is the non-redundant peel (~2 ids,
 *     precise). Peeling a raw/typo'd word is garbage, so sparse INTRINSICALLY runs
 *     the recovery first: fuzzy F → NN nearest canonical word (`ctx.index`) →
 *     peel. NN is how you reach a clean word to peel, not an independent toggle.
 *
 * **Phases.** `[L,R]` and `[L∪C]` are not one representation — they belong to
 * different stages of the stack with different constraints, so they are pooled
 * differently. `ctx.phase` selects which:
 *
 *   - `'bidirectional'` (BERT-style encoder) — the right context is legitimately
 *     available (no leak). Produces `L`, `C`, `R` over a radius-`r` window
 *     (default `r = 5`). **Shift-pad is on by default**: near a chunk boundary,
 *     instead of letting the short side atrophy, the window slides inward to keep
 *     the full `2r` context — recovering the boundary uniqueness a plain clamp
 *     loses. `LC` is not defined here. Set `shiftPad: false` for a plain clamp.
 *   - `'causal'` (GPT-style generator) — the right context is the *forbidden
 *     future*, so it cannot be pooled and cannot be borrowed by shift-pad.
 *     Produces `L`, `C`, `LC` over a causal window of `r` preceding words
 *     (default `r = 5`, i.e. `[L∪C]` spans 6 words). `R` is empty.
 *   - `'raw'` (default; analysis/back-compat) — the whole-chunk unbounded
 *     prefix/suffix union (`L`, `R`, `LC` all populated). Pass a finite
 *     `ctx.window` to clamp symmetrically instead. This mode saturates in long
 *     chunks; the two production phases above do not.
 *
 * Shift-pad is therefore a per-phase setting, not a global property: on for the
 * bidirectional encoder (R is real), off for the causal generator (R would leak).
 * There is no cross-phase invariant to keep — they are different objects.
 *
 * Note: OR-pooled context is **order-invariant** (a word-*set* maps to one
 * signature) — this matches raw self-attention, which is itself
 * permutation-equivariant and recovers order from a separate positional encoding.
 * See `benchmark/signature/chunkSignatures.js` and `RESULTS.md` §5.
 */

import { bpeEncode } from "../encoding/bpeEncode.js";
import { fuzzyEncode } from "../encoding/fuzzyEncode.js";
import { canonicalize } from "../normalization/canonicalize.js";
import { or } from "../../math/sparse/or.js";

const union = (a, b) => Uint32Array.from(or(a, b));
const EMPTY = new Uint32Array(0);

/**
 * @function sparseF
 * @description The sparse (peel) signature of a word: the sorted, de-duplicated
 * set of part ids produced by {@link module:core/parts/encoding/bpeEncode}.
 * @param {import("../dictionary.js").PartDictionary} dict - Part dictionary.
 * @param {string} word - The (canonical) word to peel.
 * @returns {Uint32Array} Sorted unique peel part ids.
 */
export const sparseF = (dict, word) => Uint32Array.from(
  [...new Set(bpeEncode(dict, word).parts.map((p) => p.id))].sort((a, b) => a - b)
);

/**
 * @constant ENCODERS
 * @description The two per-word encoders `C` can be derived from — pick ONE for
 * the whole `[L,C,R]` (never mix). `sparse` is the non-redundant peel id-set (the
 * token identity); `fuzzy` is the redundant containment F (more overlap for an
 * attention layer to work over). The choice is a first-input configuration to
 * evaluate end-to-end. `fuzzyEncode` already returns a sorted id array.
 */
const ENCODERS = {
  sparse: (dict, w) => sparseF(dict, w),
  fuzzy: (dict, w) => fuzzyEncode(dict, w),
};

/**
 * @function rangeOr
 * @description OR of `C[lo..hi]` (inclusive), clamped to the array bounds;
 * returns the shared empty set when the range is degenerate (`lo > hi`).
 * @param {Uint32Array[]} C - Per-word signatures.
 * @param {number} lo - Range start (may be negative; clamped to 0).
 * @param {number} hi - Range end (may exceed `C.length-1`; clamped).
 * @returns {Uint32Array} Sorted union over the range.
 */
const rangeOr = (C, lo, hi) => {
  let u = EMPTY;
  for (let j = Math.max(0, lo); j <= Math.min(C.length - 1, hi); j++) u = union(u, C[j]);
  return u;
};

/**
 * @function shiftWin
 * @description Fixed-size window of radius `r` (i.e. `2r+1` words) centred at `i`,
 * shifted *inward* so it stays full when it would overrun a chunk edge — the
 * boundary-compensation used by the bidirectional phase. When one side is short
 * near a boundary the opposite side is extended to preserve the total `2r`
 * context. Chunks no longer than `2r+1` words return the whole chunk.
 * @param {number} i - Centre position.
 * @param {number} k - Chunk length.
 * @param {number} r - Radius in words.
 * @returns {[number, number]} Inclusive `[lo, hi]` window bounds.
 */
const shiftWin = (i, k, r) => {
  if (k <= 2 * r + 1) return [0, k - 1];
  let lo = i - r, hi = i + r;
  if (lo < 0) { hi -= lo; lo = 0; } else if (hi > k - 1) { lo -= hi - (k - 1); hi = k - 1; }
  return [lo, hi];
};

/**
 * @typedef {Object} WordSignature
 * @property {string} word - The (possibly canonicalized) word at this position.
 * @property {string} input - The original word.
 * @property {Uint32Array} C - The word's per-word signature under `ctx.encoder`
 *   (sparse peel by default, or fuzzy F).
 * @property {Uint32Array} L - OR of preceding words' `C` (left context). Empty at
 *   position 0.
 * @property {Uint32Array} R - OR of following words' `C` (right context). Empty at
 *   the last position, and **empty for the whole `'causal'` phase** (no future).
 * @property {Uint32Array} LC - `[L∪C]`, the causal prefix-union. **Empty for the
 *   `'bidirectional'` phase** (it belongs to the causal phase).
 */

/**
 * @function buildChunkSignatures
 * @description Computes the chunk-contextual signature for every word in a chunk,
 * pooled according to `ctx.phase` (see the file header).
 *
 * @param {string[]} words - The chunk as a sequence of word byte-strings.
 * @param {Object} ctx
 * @param {import("../dictionary.js").PartDictionary} ctx.dict - Part dictionary
 *   (with the bigram backstop added).
 * @param {'sparse'|'fuzzy'} [ctx.encoder='sparse'] - Which per-word encoder the
 *   ENTIRE `[L,C,R]` is derived from (never mixed). `fuzzy` = the raw, self-robust
 *   containment F (no NN). `sparse` = the peel, reached via the recovery pipeline
 *   (fuzzy F → NN → peel).
 * @param {import("../normalization/invertedIndex.js").InvertedIndex} [ctx.index]
 *   - Vocabulary index for the sparse encoder's NN recovery (fuzzy F → nearest
 *   canonical word) — needed for sparse to be typo-robust. **Ignored by `fuzzy`**
 *   (self-robust) and not an independent knob. Without it, sparse peels the raw
 *   word (valid only for clean input).
 * @param {number} [ctx.theta=0.5] - Canonicalization gate (sparse recovery).
 * @param {'raw'|'causal'|'bidirectional'} [ctx.phase='raw'] - Pooling phase.
 * @param {number} [ctx.radius=5] - Context radius in words for the `'causal'`
 *   and `'bidirectional'` phases.
 * @param {boolean} [ctx.shiftPad=true] - `'bidirectional'` only: keep the full
 *   `2r` context at boundaries (`true`) or clamp plainly (`false`).
 * @param {number} [ctx.window=Infinity] - `'raw'` only: `Infinity` for the
 *   unbounded prefix/suffix union, or a finite `w` for a symmetric clamp of `w`
 *   words (`L`/`R` reach `w-1` each side).
 * @returns {WordSignature[]} One entry per word, in position order.
 *
 * @example
 * // Causal (GPT-side): predict the next word from the last 5 words. R is empty.
 * const s = buildChunkSignatures(words, { dict, phase: "causal", radius: 5 });
 * // s[i].LC = OR of C[max(0,i-5) .. i]   (never saturates past 6 words)
 *
 * @example
 * // Bidirectional (BERT-side): ±5 cloze frame, shift-padded at boundaries.
 * const s = buildChunkSignatures(words, { dict, phase: "bidirectional", radius: 5 });
 * // s[0].R spans 2·5 words (left side is empty, so the window slides right)
 *
 * @example
 * // Raw (analysis): unbounded running prefix/suffix union.
 * const s = buildChunkSignatures(["the", "cat", "sat"], { dict });
 * s[1].L;   // === s[0].LC  (left context = previous prefix-union)
 */
export const buildChunkSignatures = (words, ctx) => {
  const { dict, index, theta = 0.5, phase = "raw", radius = 5, shiftPad = true, encoder = "sparse" } = ctx;
  const k = words.length;
  const encode = ENCODERS[encoder] || ENCODERS.sparse;
  // NN recovery is INTRINSIC to sparse, not a separate knob: peeling a raw (maybe
  // typo'd) word is garbage, so sparse runs fuzzy F → NN → peel as one pipeline
  // (needs `index`). Fuzzy is the raw self-robust key and never recovers.
  //
  // When the index carries a peel cache, an NN hit returns its precomputed peel
  // by `wi` — no re-encoding per token. Only OOV/below-θ (`wi < 0`) re-peels.
  const useNN = encoder === "sparse" && index;
  const canon = new Array(k), C = new Array(k);
  for (let i = 0; i < k; i++) {
    if (useNN) {
      const { canonical, wi } = canonicalize(words[i], dict, index, { theta });
      canon[i] = canonical;
      C[i] = (wi >= 0 && index.peels) ? index.peels[wi] : encode(dict, canonical);
    } else {
      canon[i] = words[i];
      C[i] = encode(dict, words[i]);
    }
  }
  const L = new Array(k), R = new Array(k), LC = new Array(k);

  if (phase === "causal") {
    // GPT-style: causal window of `r` preceding words; the right side is the
    // forbidden future, so R is empty and shift-pad cannot apply.
    const r = Math.max(1, radius | 0);
    for (let i = 0; i < k; i++) {
      L[i] = rangeOr(C, i - r, i - 1);
      LC[i] = union(L[i], C[i]);
      R[i] = EMPTY;
    }
  } else if (phase === "bidirectional") {
    // BERT-style: R is available. shift-pad (default) keeps the full 2r context
    // at chunk boundaries; a plain clamp lets the short side atrophy. [L∪C] is
    // the causal phase's object and is not defined here.
    const r = Math.max(1, radius | 0);
    for (let i = 0; i < k; i++) {
      const [lo, hi] = shiftPad ? shiftWin(i, k, r) : [i - r, i + r];
      L[i] = rangeOr(C, lo, i - 1);
      R[i] = rangeOr(C, i + 1, hi);
      LC[i] = EMPTY;
    }
  } else {
    // raw (analysis/back-compat): unbounded, or a symmetric clamp when window is finite.
    const window = ctx.window ?? Infinity;
    if (window === Infinity) {
      let run = EMPTY;
      for (let i = 0; i < k; i++) { L[i] = run; LC[i] = run = union(run, C[i]); }
      run = EMPTY;
      for (let i = k - 1; i >= 0; i--) { R[i] = run; run = union(run, C[i]); }
    } else {
      const w = Math.max(1, window | 0);
      for (let i = 0; i < k; i++) {
        L[i] = rangeOr(C, i - (w - 1), i - 1);
        LC[i] = union(L[i], C[i]);
        R[i] = rangeOr(C, i + 1, i + (w - 1));
      }
    }
  }
  return words.map((word, i) => ({ input: word, word: canon[i], C: C[i], L: L[i], R: R[i], LC: LC[i] }));
};

export default buildChunkSignatures;
