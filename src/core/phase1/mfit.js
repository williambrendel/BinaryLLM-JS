"use strict";

/**
 * @file mfit.js
 * @brief §6 — m-of-n gate fit. Given a part's bit-set, pick the support `m` that
 * maximizes precision on the confusable negatives, subject to the weighted-recall
 * floor 0.5:
 *   m* = argmax_m precision(Q,m)  s.t.  recall_w(Q,m) ≥ 0.5
 *
 * Precision is base-rate-correct so it is comparable to `p_min = |A|/(|A|+|Neg|)`:
 * the weighted recall is converted to an effective covered-positive count
 * `tp = recall_w·|A|`, and `fp` is the raw count of firing confusables.
 */

import andCount from "../math/sparse/andCount.js";

/**
 * @function mfit
 * @param {number[]} Qbits - part bits (sorted).
 * @param {Array<number[]>} A - positive contexts.
 * @param {number[]|Float64Array} w - AdaBoost weights over A.
 * @param {Array<number[]>} Neg - confusable negatives.
 * @param {object} [opts] @param {number} [opts.recallFloor=0.5]
 * @returns {{m:number, recall:number, precision:number, recallAt1:number, valid:boolean}}
 *   best gate; `valid=false` (no `m` clears the recall floor) is a ρ-too-small signal (§6). `recallAt1`
 *   is the support ceiling (m=1, OR over Q) used by the §6 well-posedness diagnostic.
 */
export const mfit = (Qbits, A, w, Neg, opts = {}) => {
  const { recallFloor = 0.5 } = opts;
  const maxm = Qbits.length; if (maxm === 0) return { valid: false, recallAt1: 0 };
  const nA = A.length;
  const cA = new Int32Array(nA); for (let i = 0; i < nA; i++) cA[i] = andCount(A[i], Qbits);
  const cN = new Int32Array(Neg.length); for (let j = 0; j < Neg.length; j++) cN[j] = andCount(Neg[j], Qbits);
  let W = 0; for (let i = 0; i < nA; i++) W += w[i];

  let best = null, recallAt1 = 0;
  for (let m = 1; m <= maxm; m++) {
    let tw = 0; for (let i = 0; i < nA; i++) if (cA[i] >= m) tw += w[i];
    const recall = tw / (W || 1);
    if (m === 1) recallAt1 = recall;                                     // support ceiling (well-posedness, §6)
    if (recall < recallFloor) continue;                                  // recall is monotone-decreasing in m
    let fp = 0; for (let j = 0; j < cN.length; j++) if (cN[j] >= m) fp++;
    // §6: base-rate-correct precision — tp on the COUNT scale (recall_w·|A|), commensurate with the count fp
    const tp = recall * nA, precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    if (!best || precision > best.precision) best = { m, recall, precision };
  }
  return best ? { ...best, recallAt1, valid: true } : { valid: false, recallAt1 };
};

export default mfit;
