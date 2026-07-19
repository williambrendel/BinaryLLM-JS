"use strict";

/**
 * @file affinity.js
 * @brief §1.1–1.2 — the `w`-weighted unary log-odds `u` and the PMI-difference affinity `M`, REBUILT each round.
 *
 *   u(b)  = log( (p⁺_w(b)+α_r) / (p⁻(b)+α_r) )              rate-smoothed unary log-odds
 *   M_ab  = PMI⁺_w(a,b) − PMI⁻(a,b)                          PMI difference
 *   PMI(a,b) = log( p(a,b) / (p(a)·p(b)) )                   marginal-normalized association
 *
 * p⁺_w is the AdaBoost-weighted positive rate (weight w_i per context); p⁻ is frozen. An edge exists iff
 * `M_ab ≠ 0`. Bits index into `pPlus` for the matvec. (The marginal normalization inside PMI is load-bearing —
 * it cancels common bits; a plain rate difference fails. The old max(0,·) flooring was CV-confirmed unnecessary.)
 */

import { pairKey, PAIRK } from "./negSet.js";

/**
 * @function buildAffinity
 * @param {Array<number[]>} A - positive contexts.
 * @param {number[]|Float64Array} w - AdaBoost weights over A (any positive scale).
 * @param {{pPlus:number[], pset:Set<number>, negN:number, nMinus:Map, jMinus:Map}} neg
 * @param {object} [opts] @param {number} [opts.alphaR] - rate-smoothing (default 1/negN).
 * @returns {{u:Float64Array, edges:{i:Int32Array,j:Int32Array,m:Float64Array}, idx:Map<number,number>, n:number}}
 */
export const buildAffinity = (A, w, neg, opts = {}) => {
  const { pPlus, pset, negN, nMinus, jMinus } = neg;
  const alphaR = opts.alphaR ?? 1 / Math.max(1, negN);
  const n = pPlus.length;
  const idx = new Map(); pPlus.forEach((b, i) => idx.set(b, i));

  // weighted positive marginals + joints over P⁺ (full rebuild each round — exact).
  let W = 0; const mPos = new Float64Array(n), jPos = new Map();
  for (let ci = 0; ci < A.length; ci++) {
    const wi = w[ci]; if (wi <= 0) continue; W += wi;
    const pos = []; for (const b of A[ci]) if (pset.has(b)) pos.push(idx.get(b));
    pos.sort((x, z) => x - z);
    for (const p of pos) mPos[p] += wi;
    for (let i = 0; i < pos.length; i++) for (let j = i + 1; j < pos.length; j++) { const k = pos[i] * pPlus.length + pos[j]; jPos.set(k, (jPos.get(k) || 0) + wi); }
  }

  // unary log-odds
  const u = new Float64Array(n);
  for (let p = 0; p < n; p++) { const pp = mPos[p] / (W || 1), pm = (nMinus.get(pPlus[p]) || 0) / (negN || 1); u[p] = Math.log((pp + alphaR) / (pm + alphaR)); }

  // pairwise PMI difference (no flooring). Candidate pairs = observed in positives OR negatives.
  const pairM = (jw, mi, mj, jn, ni, nj) => {
    const pp = jw > 0 ? Math.log((jw / W) / ((mi / W) * (mj / W))) : 0;
    const pn = jn > 0 ? Math.log((jn / negN) / ((ni / negN) * (nj / negN))) : 0;
    return pp - pn;
  };
  const seen = new Set(), Iarr = [], Jarr = [], Marr = [];
  const emit = (pi, pj, mval) => { if (mval !== 0) { Iarr.push(pi); Jarr.push(pj); Marr.push(mval); } };
  for (const [k, jw] of jPos) {                                          // positive-observed pairs
    const pi = Math.floor(k / pPlus.length), pj = k % pPlus.length; seen.add(k);
    const nk = pairKey(pPlus[pi], pPlus[pj]); const jn = jMinus.get(nk) || 0;
    emit(pi, pj, pairM(jw, mPos[pi], mPos[pj], jn, nMinus.get(pPlus[pi]) || 0, nMinus.get(pPlus[pj]) || 0));
  }
  for (const [nk, jn] of jMinus) {                                       // negative-only pairs → repulsive
    const a = Math.floor(nk / PAIRK), b = nk % PAIRK;
    const pi = idx.get(a), pj = idx.get(b); if (pi === undefined || pj === undefined) continue;
    const key = pi < pj ? pi * pPlus.length + pj : pj * pPlus.length + pi;
    if (seen.has(key)) continue;
    emit(Math.min(pi, pj), Math.max(pi, pj), pairM(0, mPos[pi], mPos[pj], jn, nMinus.get(a) || 0, nMinus.get(b) || 0));
  }

  return { u, edges: { i: Int32Array.from(Iarr), j: Int32Array.from(Jarr), m: Float64Array.from(Marr) }, idx, n };
};

export default buildAffinity;
