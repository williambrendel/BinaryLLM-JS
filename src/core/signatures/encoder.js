"use strict";

// ============================================================================
// core/signatures/encoder.js
//
// Port of core/signatures/encoder.cpp — encode the Word tokens of a token
// stream into 3F signatures.
//
// A 3F signature is a sparse sorted set of "on" bit indices over dimension
// 3*F, laid out as three F-wide bands:
//
//   [0,  F)  before  (L) — OR of the current bags of the left context words
//   [F, 2F)  current (C) — this word's Option B bag (see wordEncoder.js)
//   [2F,3F)  after   (R) — OR of the current bags of the right context words
//
//   encode          — L / R pool over the whole [start,end) scope
//   encodeWindowed  — L / R pool over a bounded window of `radius` words each
//                     side (spec v3.1 Pass-1 §8). radius 0 => empty L/R.
//
// Representation: each signature is a TypedArray of ascending global bit
// indices (sparse). Element type is chosen from the dimension by
// `sparseArrayType`: Uint16Array when the largest index fits (dim <= 65536),
// else Uint32Array — so a small dictionary stays 2 bytes/id and a large one
// (3F > 65536) is safely widened.
//
// Band views (`toLCR`, `toLR`, `toLunionC`, `toLC`) remap a 3F signature into
// the reduced inputs used by the models; they mirror apps/tree_parts_common's
// tp_data band selection, not a separate encoder.
// ============================================================================

import { StreamTokenType, asciiLowercase } from "../parts/tokenize.js";
import { encodeWord } from "./wordEncoder.js";

/**
 * Sparse element type for a signature of the given dimension: the largest bit
 * index is dim-1, so Uint16 suffices iff dim <= 65536.
 * @param {number} dim number of bit positions (e.g. 3F, 2F, or F).
 * @returns {Uint16ArrayConstructor|Uint32ArrayConstructor}
 */
export const sparseArrayType = (dim) => (dim <= 0x10000 ? Uint16Array : Uint32Array);

// Union of two ascending, de-duplicated id arrays (two-pointer merge).
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

// Assemble one 3F signature from its three (already-sorted, disjoint-range)
// bands. Concatenation is globally sorted: before < F <= current < 2F <= after.
const assemble = (Arr, F, before, current, after) => {
  const sig = new Arr(before.length + current.length + after.length);
  let k = 0;
  for (let t = 0; t < before.length; t++) sig[k++] = before[t]; // [0,F)
  for (let t = 0; t < current.length; t++) sig[k++] = F + current[t]; // [F,2F)
  for (let t = 0; t < after.length; t++) sig[k++] = 2 * F + after[t]; // [2F,3F)
  return sig;
};

// Shared setup: the Word-token indices in [start,end) and their Option B bags.
const prepare = (dict, tokens, start, end) => {
  const n = tokens.length;
  if (end > n) end = n;
  const wordIdx = [];
  for (let i = start; i < end; i++) {
    if (tokens[i].type === StreamTokenType.Word) wordIdx.push(i);
  }
  const current = wordIdx.map((wi) => encodeWord(dict, asciiLowercase(tokens[wi].value)));
  return current;
};

/**
 * Encode Word tokens in [start,end) into 3F signatures, pooling L/R over the
 * whole scope.
 * @param {import("../parts/dictionary.js").PartDictionary} dict
 * @param {Array<{type: number, value: string}>} tokens
 * @param {number} [start=0]
 * @param {number} [end=Infinity] clamped to tokens.length.
 * @returns {Array<Uint16Array|Uint32Array>} one signature per Word token.
 */
export const encode = (dict, tokens, start = 0, end = Infinity) => {
  const F = dict.size();
  const Arr = sparseArrayType(3 * F);
  const current = prepare(dict, tokens, start, end);
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
    out[i] = assemble(Arr, F, accL, current[i], after[i]);
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
 * @returns {Array<Uint16Array|Uint32Array>} one signature per Word token.
 */
export const encodeWindowed = (dict, tokens, radius, start = 0, end = Infinity) => {
  const F = dict.size();
  const Arr = sparseArrayType(3 * F);
  const current = prepare(dict, tokens, start, end);
  const N = current.length;
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    const lo = i > radius ? i - radius : 0;
    const hi = Math.min(N, i + radius + 1);
    out[i] = assemble(Arr, F, unionRange(current, lo, i), current[i], unionRange(current, i + 1, hi));
  }
  return out;
};

// ---- Band views over a 3F signature -------------------------------------

/**
 * Identity view: a 3F signature is already [L, C, R]. Returns a copy.
 * @param {Uint16Array|Uint32Array} sig
 * @returns {Uint16Array|Uint32Array}
 */
export const toLCR = (sig) => sig.slice();

/**
 * F-dim [L ∪ C]: union the before band and the current band into one F-wide
 * bag (the GPT input). Drops R.
 * @param {Uint16Array|Uint32Array} sig 3F signature.
 * @param {number} F dictionary size.
 * @returns {Uint16Array|Uint32Array} sorted ids in [0,F).
 */
export const toLunionC = (sig, F) => {
  const before = [];
  const curr = [];
  for (let i = 0; i < sig.length; i++) {
    const b = sig[i];
    if (b < F) before.push(b);
    else if (b < 2 * F) curr.push(b - F);
    else break; // after band — sorted, so we're done
  }
  return sparseArrayType(F).from(mergeUnion(before, curr));
};

/**
 * 2F-dim [L | R]: before band in [0,F) and after band remapped to [F,2F) (the
 * BERT input). Drops C.
 * @param {Uint16Array|Uint32Array} sig 3F signature.
 * @param {number} F dictionary size.
 * @returns {Uint16Array|Uint32Array} sorted ids in [0,2F).
 */
export const toLR = (sig, F) => {
  const res = [];
  for (let i = 0; i < sig.length; i++) if (sig[i] < F) res.push(sig[i]); // L -> [0,F)
  for (let i = 0; i < sig.length; i++) if (sig[i] >= 2 * F) res.push(sig[i] - F); // R -> [F,2F)
  return sparseArrayType(2 * F).from(res);
};

/**
 * 2F-dim [L | C]: before band and current band kept in place (the GPT band-C
 * control). Drops R.
 * @param {Uint16Array|Uint32Array} sig 3F signature.
 * @param {number} F dictionary size.
 * @returns {Uint16Array|Uint32Array} sorted ids in [0,2F).
 */
export const toLC = (sig, F) => {
  const res = [];
  for (let i = 0; i < sig.length; i++) {
    if (sig[i] < 2 * F) res.push(sig[i]);
    else break;
  }
  return sparseArrayType(2 * F).from(res);
};

export default encode;
