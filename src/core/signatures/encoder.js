"use strict";

// ============================================================================
// core/signatures/encoder.js
//
// Port of core/signatures/encoder.cpp — encode the Word tokens of a token
// stream into 3F signatures.
//
// A signature is a LIST OF BANDS: `TypedArray[]`, each band a sparse sorted set
// of local part ids in [0, F). The three bands of a full signature are:
//
//   L (before)  — OR of the current bags of the left context words
//   C (current) — this word's Option B bag (see wordEncoder.js)
//   R (after)   — OR of the current bags of the right context words
//
// Bands are always local ([0,F)), so the element type keys on F via
// `sparseArrayType(F)` (Uint16Array while F <= 65536, else Uint32Array). This
// holds uniformly for 1-band, 2-band, and 3-band signatures — a small or a
// large dictionary keeps 2 bytes/id.
//
//   encode          — L / R pool over the whole [start,end) scope
//   encodeWindowed  — L / R pool over a bounded window of `radius` words each
//                     side (spec v3.1 Pass-1 §8). radius 0 => empty L / R.
//
// Band selectors return reduced banded signatures; `flatten` is the ONE place
// bands are concatenated into a single global-indexed vector, and the ONE
// place the total dimension (nBands*F) — not F — drives the element type:
//   lcr (3F [L,C,R]), lr (2F [L|R], BERT), lc (2F [L|C] control),
//   lUnionC (F [L∪C], GPT); flatten(bands, F) -> one TypedArray.
// ============================================================================

import { StreamTokenType, asciiLowercase } from "../parts/tokenize.js";
import { sparseArrayType } from "../../utilities/sparseArrayType.js";
import { encodeWord } from "./wordEncoder.js";

/**
 * A signature is an array of bands, each a sparse sorted TypedArray of local
 * ids in [0, F). A full signature is [L, C, R]; reduced views have 1 or 2 bands.
 * @typedef {Array<Uint16Array|Uint32Array>} Signature
 */

// Union of two ascending, de-duplicated id sequences (two-pointer merge).
// Accepts arrays or TypedArrays; returns a plain Array.
const mergeUnion = (a, b) => {
  const out = [];
  let i = 0;
  let j = 0;
  const al = a.length;
  const bl = b.length;
  while (i < al && j < bl) {
    const ai = a[i];
    const bj = b[j];
    if (ai < bj) out.push(ai), i++;
    else if (ai > bj) out.push(bj), j++;
    else out.push(ai), i++, j++;
  }
  while (i < al) out.push(a[i++]);
  while (j < bl) out.push(b[j++]);
  return out;
};

// Union of current[lo..hi) — used for bounded-window pooling.
const unionRange = (bags, lo, hi) => {
  let acc = [];
  for (let j = lo; j < hi; j++) acc = mergeUnion(acc, bags[j]);
  return acc;
};

// Shared setup: the per-word bags (local id arrays) for the Word tokens in
// [start,end). `bagOf` selects the word encoding (sparse `encodeWord` default,
// or `fuzzyEncodeWord` for the dense all-candidates bag).
const prepare = (dict, tokens, start, end, bagOf) => {
  const n = tokens.length;
  if (end > n) end = n;
  const bags = [];
  for (let i = start; i < end; i++) {
    if (tokens[i].type === StreamTokenType.Word) {
      bags.push(bagOf(dict, asciiLowercase(tokens[i].value)));
    }
  }
  return bags;
};

/**
 * Encode Word tokens in [start,end) into 3F signatures, pooling L/R over the
 * whole scope.
 * @param {import("../parts/dictionary.js").PartDictionary} dict
 * @param {Array<{type: number, value: string}>} tokens
 * @param {number} [start=0]
 * @param {number} [end=Infinity] clamped to tokens.length.
 * @returns {Signature[]} one [L, C, R] signature per Word token.
 */
export const encode = (dict, tokens, start = 0, end = Infinity, bagOf = encodeWord) => {
  const Arr = sparseArrayType(dict.size());
  const current = prepare(dict, tokens, start, end, bagOf);
  const N = current.length;
  const out = new Array(N);
  if (N === 0) return out;

  // Suffix unions: after[i] = union(current[i+1..N)).
  const after = new Array(N);
  let accR = [];
  for (let i = N - 1; i >= 0; i--) {
    after[i] = accR;
    accR = mergeUnion(accR, current[i]);
  }
  // Forward: before[i] = union(current[0..i)).
  let accL = [];
  for (let i = 0; i < N; i++) {
    out[i] = [Arr.from(accL), Arr.from(current[i]), Arr.from(after[i])];
    accL = mergeUnion(accL, current[i]);
  }
  return out;
};

/**
 * Encode Word tokens in [start,end) into 3F signatures, pooling L/R over a
 * bounded window of `radius` Word tokens each side.
 * @param {import("../parts/dictionary.js").PartDictionary} dict
 * @param {Array<{type: number, value: string}>} tokens
 * @param {number} radius window reach each side (0 => empty L/R bands).
 * @param {number} [start=0]
 * @param {number} [end=Infinity] clamped to tokens.length.
 * @returns {Signature[]} one [L, C, R] signature per Word token.
 */
export const encodeWindowed = (dict, tokens, radius, start = 0, end = Infinity, bagOf = encodeWord) => {
  const Arr = sparseArrayType(dict.size());
  const current = prepare(dict, tokens, start, end, bagOf);
  const N = current.length;
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    const lo = i > radius ? i - radius : 0;
    const hi = Math.min(N, i + radius + 1);
    out[i] = [
      Arr.from(unionRange(current, lo, i)),
      Arr.from(current[i]),
      Arr.from(unionRange(current, i + 1, hi)),
    ];
  }
  return out;
};

/**
 * Encode Word tokens into POSITION-PRESERVING signatures: instead of OR-pooling a
 * window into one F-wide bag (which makes every bit fire diffusely), each offset
 * keeps its own subspace — the word at −o contributes bits shifted by `(o−1)·F`.
 * So a bit stays sparse and position-specific ("piece X at offset −2") across a
 * wide window, keeping peels expressible where pooling would only allow bisects.
 *
 *   L: word@−o → bits + (o−1)·F      (o = 1…radius)   → [0, r·F)
 *   C: current word → bits + r·F                       → [r·F, (r+1)·F)
 *   R: word@+o → bits + (r+o)·F      (o = 1…radius)   → [(r+1)·F, (2r+1)·F)
 *
 * Bands land in disjoint id ranges, so `L∪C` / `[L|R]` compose without collision.
 * @returns {Signature[]} one [L, C, R] positional signature per Word token.
 */
export const encodePositional = (dict, tokens, radius, start = 0, end = Infinity, bagOf = encodeWord) => {
  const F = dict.size();
  const Arr = sparseArrayType((2 * radius + 1) * F);
  const current = prepare(dict, tokens, start, end, bagOf);
  const N = current.length;
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    const L = [];
    for (let o = 1; o <= radius && i - o >= 0; o++) { const off = (o - 1) * F; for (const b of current[i - o]) L.push(b + off); }
    const C = []; { const off = radius * F; for (const b of current[i]) C.push(b + off); }
    const R = [];
    for (let o = 1; o <= radius && i + o < N; o++) { const off = (radius + o) * F; for (const b of current[i + o]) R.push(b + off); }
    L.sort((a, b) => a - b); R.sort((a, b) => a - b);
    out[i] = [Arr.from(L), Arr.from(C), Arr.from(R)];
  }
  return out;
};

// ---- Band selectors (return reduced banded signatures) -------------------
// Each returns a Signature (TypedArray[]); bands stay local and F-typed.

/** 3F [L, C, R] — the full signature (returns the same three bands). */
export const lcr = (sig) => [sig[0], sig[1], sig[2]];

/** 2F [L | R] — drop the current band (the BERT input). */
export const lr = (sig) => [sig[0], sig[2]];

/** 2F [L | C] — drop the after band (the band-C control). */
export const lc = (sig) => [sig[0], sig[1]];

/**
 * F [L ∪ C] — union the L and C bands into one F-wide band (the GPT input).
 * Creates a new band, so it types it via sparseArrayType(F) like `encode`.
 * @param {Signature} sig
 * @param {number} F dictionary size (band width).
 * @returns {Signature} a single-band signature.
 */
export const lUnionC = (sig, F) => [sparseArrayType(F).from(mergeUnion(sig[0], sig[1]))];

/**
 * Concatenate a banded signature into one global-indexed sparse vector: band k
 * is shifted by k*F. This is the single place the TOTAL dimension (nBands*F),
 * not F, drives the element type — so a 2F/3F flat may widen to Uint32 even
 * when the bands are Uint16.
 * @param {Signature} bands one or more F-wide bands (e.g. from lcr/lr/lc/lUnionC).
 * @param {number} F dictionary size (band width).
 * @returns {Uint16Array|Uint32Array} sorted ids in [0, bands.length*F).
 */
export const flatten = (bands, F) => {
  let total = 0;
  for (const b of bands) total += b.length;
  const out = new (sparseArrayType(bands.length * F))(total);
  let k = 0;
  for (let band = 0; band < bands.length; band++) {
    const off = band * F;
    const b = bands[band];
    for (let i = 0; i < b.length; i++) out[k++] = off + b[i];
  }
  return out;
};

export default encode;
