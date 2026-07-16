"use strict";

/**
 * @file peelNegative.js
 * @brief PEEL_NEG (§5.3) — build the negative channel `p⁻_k` for a clique `Q`:
 * the confusable look-alike vocabulary sense `k` must exclude.
 *
 * `p⁻_k` = bits frequent among `Q`'s false positives (residual negatives that
 * carry ≥ m of `Q`'s bits) that are (a) not in `Q`, (b) not in the global drop
 * set `D_τ`, and (c) DEPLETED in the positives (`d(b) < κ⁻¹`). Sorted ascending.
 *
 * `∉ D_τ` is largely implied by ∉Q + depleted-in-A, but asserted explicitly.
 */

import { buildHitLists, bitLift } from "./synergyGain.js";
import andCount from "../math/sparse/andCount.js";

/** Membership test against a sorted array via a Set built once. */
const setOf = (arr) => new Set(arr);

/**
 * @function peelNegative
 * @param {number[]} Q - the clique (sorted bit ids).
 * @param {Array<number[]>} Nctx - residual negative contexts.
 * @param {Array<number[]>} Pctx - the part's positive slice (for depletion d(b)).
 * @param {object} opts
 * @param {number[]} opts.Dtau - global drop set (sorted).
 * @param {number} [opts.kappa=2] - depletion cutoff: keep b iff d(b) < 1/κ.
 * @param {number} [opts.m=2] - m-of-n support that marks a negative a false positive.
 * @param {number} [opts.maxNeg=Infinity] - cap on |p⁻| (top by FP frequency).
 * @param {number} [opts.minFP=1] - min FP occurrences a bit needs to qualify.
 * @returns {{pMinus:number[], nFP:number, AQ:number[]}} the negative channel
 *   (sorted), the false-positive count, and the carriers `A_Q` (indices into Pctx).
 */
export const peelNegative = (Q, Nctx, Pctx, opts) => {
  const { Dtau, kappa = 2, m = 2, maxNeg = Infinity, minFP = 1 } = opts;
  const Qset = setOf(Q), Dset = setOf(Dtau);
  const nPos = Pctx.length, nNeg = Nctx.length;

  // False positives: residual negatives carrying ≥ m of Q's bits. Tally bit freq in them.
  const freq = new Map();
  let nFP = 0;
  for (const y of Nctx) {
    if (andCount(y, Q) < m) continue;
    nFP++;
    for (let t = 0; t < y.length; t++) { const b = y[t]; if (!Qset.has(b) && !Dset.has(b)) freq.set(b, (freq.get(b) || 0) + 1); }
  }

  // Depletion filter: keep bits under-represented in the positive slice (d < 1/κ).
  const hitP = buildHitLists(Pctx), hitN = buildHitLists(Nctx), thr = 1 / kappa;
  const cand = [];
  for (const [b, f] of freq) {
    if (f < minFP) continue;
    if (Dset.has(b)) continue;                                  // asserted, not just implied
    if (bitLift(b, hitP, nPos, hitN, nNeg) < thr) cand.push([b, f]);
  }
  cand.sort((a, b) => b[1] - a[1]);                             // by FP frequency, desc
  const kept = cand.slice(0, maxNeg === Infinity ? cand.length : maxNeg).map((e) => e[0]);
  kept.sort((a, b) => a - b);                                    // sorted for sparse ops

  const AQ = [];
  for (let i = 0; i < Pctx.length; i++) if (andCount(Pctx[i], Q) >= m) AQ.push(i);

  return { pMinus: kept, nFP, AQ };
};

export default peelNegative;
