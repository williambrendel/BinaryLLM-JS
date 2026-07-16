"use strict";

/**
 * @file affinity.js
 * @brief §1.1–1.2 — the `w`-weighted unary log-odds `u` and the PMI-difference
 * affinity `M`, REBUILT each round (§10). No C⁺/C⁻ state, no downdate.
 *
 *   u(b) = log( (p⁺_w(b)+α_r) / (p⁻(b)+α_r) )                      rate-smoothed log-odds
 *   M_ab = max(0, PMI⁺_w(a,b)) − max(0, PMI⁻(a,b))                 PMI-difference (each floored ≥0)
 *
 * p⁺_w is the AdaBoost-weighted positive rate (weight w_i per context); p⁻ is
 * frozen. An edge exists iff `M_ab ≠ 0` (pair bound in positives → +, bound in
 * negatives → −, in neither → no edge). Bits index into `pPlus` for the matvec.
 */

import { pairKey, PAIRK } from "./negSet.js";

/**
 * @function buildAffinity
 * @param {Array<number[]>} A - positive contexts.
 * @param {number[]|Float64Array} w - AdaBoost weights over A (any positive scale).
 * @param {{pPlus:number[], pset:Set<number>, negN:number, nMinus:Map, jMinus:Map}} neg
 * @param {object} [opts] @param {number} [opts.alphaR] - rate-smoothing (default 1/negN).
 * @returns {{u:Float64Array, edges:{i:Int32Array,j:Int32Array,m:Float64Array}, idx:Map<number,number>, n:number}}
 *   `u` indexed by P⁺ position; `edges` = COO of the symmetric M over P⁺ positions.
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

  // pairwise PMI difference. Candidate pairs = observed in positives OR negatives.
  const pmiPos = (jw, pi, pj) => { const v = Math.log((jw / W) / ((mPos[pi] / W) * (mPos[pj] / W))); return v > 0 ? v : 0; };
  const seen = new Set(), Iarr = [], Jarr = [], Marr = [];
  const emit = (pi, pj, mval) => { if (mval !== 0) { Iarr.push(pi); Jarr.push(pj); Marr.push(mval); } };
  // positive-observed pairs
  for (const [k, jw] of jPos) {
    const pi = Math.floor(k / pPlus.length), pj = k % pPlus.length; seen.add(k);
    const pp = pmiPos(jw, pi, pj);
    const nk = pairKey(pPlus[pi], pPlus[pj]); const jn = jMinus.get(nk) || 0;
    let pn = 0; if (jn > 0) { const v = Math.log((jn / negN) / (((nMinus.get(pPlus[pi]) || 0) / negN) * ((nMinus.get(pPlus[pj]) || 0) / negN))); pn = v > 0 ? v : 0; }
    emit(pi, pj, pp - pn);
  }
  // negative-only pairs (both bits in P⁺, not co-observed in positives) → repulsive −PMI⁻
  for (const [nk, jn] of jMinus) {
    const a = Math.floor(nk / PAIRK), b = nk % PAIRK;
    const pi = idx.get(a), pj = idx.get(b); if (pi === undefined || pj === undefined) continue;
    const key = pi < pj ? pi * pPlus.length + pj : pj * pPlus.length + pi;
    if (seen.has(key)) continue;
    const v = Math.log((jn / negN) / (((nMinus.get(a) || 0) / negN) * ((nMinus.get(b) || 0) / negN)));
    const pn = v > 0 ? v : 0; emit(Math.min(pi, pj), Math.max(pi, pj), -pn);
  }

  // §1.3 — soft precondition (instead of hard ban): scale common bits' u and incident M toward 0 by
  // coefficient c∈[0,1]. c=1 keep, c=0 ≡ ban (π=−ρx → decays). A common bit survives only if its genuine
  // pairwise signal clears the reduced bar — signal-proportional admission, no frequency cutoff.
  const ccU = opts.commonCoefU ?? opts.commonCoef ?? 1, ccM = opts.commonCoefM ?? opts.commonCoef ?? 1, CS = opts.commonSet;
  if (CS && CS.size && (ccU !== 1 || ccM !== 1)) {                              // decoupled: unary vs edge channel
    if (ccU !== 1) for (let p = 0; p < n; p++) if (CS.has(pPlus[p])) u[p] *= ccU;
    if (ccM !== 1) for (let e = 0; e < Marr.length; e++) { let f = 1; if (CS.has(pPlus[Iarr[e]])) f *= ccM; if (CS.has(pPlus[Jarr[e]])) f *= ccM; if (f !== 1) Marr[e] *= f; }
  }

  // peel/downdate: bits claimed by accepted parts are discouraged so the replicator finds a DIFFERENT clique
  // next round. excludeCoef=0 → HARD peel (u→−∞, edges→0); 0<c<1 → SOFT (scale u & incident M ×c toward 0,
  // a nudge not a ban — the bit can return if strongly re-implicated).
  const excl = opts.exclude, ec = opts.excludeCoef ?? 0;
  if (excl && excl.size) {
    if (ec > 0) {
      for (let p = 0; p < n; p++) if (excl.has(pPlus[p])) u[p] *= ec;
      for (let e = 0; e < Marr.length; e++) { let f = 1; if (excl.has(pPlus[Iarr[e]])) f *= ec; if (excl.has(pPlus[Jarr[e]])) f *= ec; if (f !== 1) Marr[e] *= f; }
    } else {
      for (let p = 0; p < n; p++) if (excl.has(pPlus[p])) u[p] = -1e9;
      for (let e = 0; e < Marr.length; e++) if (excl.has(pPlus[Iarr[e]]) || excl.has(pPlus[Jarr[e]])) Marr[e] = 0;
    }
  }

  // edge normalization (changes the objective — NON-uniform transforms only; a global rescale is absorbed by the
  //  replicator step size and is a no-op on the parts). "tanh": squash outlier PMI edges to (−1,1) at temp T;
  //  "sym": degree-normalize M_ab/√(D_a·D_b) (spectral-clustering style, down-weights hub bits).
  const nrm = opts.normEdges;
  if (nrm === "tanh") { const T = opts.normT ?? 2; for (let e = 0; e < Marr.length; e++) Marr[e] = Math.tanh(Marr[e] / T); }
  else if (nrm === "sym") { const D = new Float64Array(n); for (let e = 0; e < Marr.length; e++) { const a = Math.abs(Marr[e]); D[Iarr[e]] += a; D[Jarr[e]] += a; } for (let e = 0; e < Marr.length; e++) { const d = Math.sqrt((D[Iarr[e]] || 1) * (D[Jarr[e]] || 1)); Marr[e] /= d || 1; } }

  return { u, edges: { i: Int32Array.from(Iarr), j: Int32Array.from(Jarr), m: Float64Array.from(Marr) }, idx, n };
};

export default buildAffinity;
