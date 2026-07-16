"use strict";

/**
 * @file thresholdFit.js
 * @brief Fit the per-gate decision threshold `t_k = argmax_t GAIN(score > t)`
 * separating a positive slice from residual negatives (§4/§6). Returns the
 * threshold value; the predicate is strict `score(x) > t`.
 */

import { splitGain } from "./synergyGain.js";

/**
 * @function fitThreshold
 * @param {Function} scoreFn - `x → score(x)`.
 * @param {Array<number[]>} Pctx - positive contexts.
 * @param {Array<number[]>} Nctx - negative contexts.
 * @returns {{t:number, gain:number, recall:number, fp:number}} best threshold,
 *   its info gain, and the (recall, FP-rate) at that operating point.
 */
export const fitThreshold = (scoreFn, Pctx, Nctx) => {
  const nPos = Pctx.length, nNeg = Nctx.length;
  const sP = Pctx.map(scoreFn), sN = Nctx.map(scoreFn);

  // Candidate thresholds: each distinct score value u (predicate ">u"), plus
  // −Infinity (fire everything). Fire counts computed by sorting once.
  const vals = [...new Set([...sP, ...sN])].sort((a, b) => a - b);
  let bestT = -Infinity, bestGain = -Infinity, bestFirP = 0, bestFirN = 0;

  const consider = (t) => {
    let firP = 0, firN = 0;
    for (const v of sP) if (v > t) firP++;
    for (const v of sN) if (v > t) firN++;
    const g = splitGain(firP, firN, nPos, nNeg);
    if (g > bestGain) { bestGain = g; bestT = t; bestFirP = firP; bestFirN = firN; }
  };

  consider(-Infinity);
  for (const u of vals) consider(u);

  return { t: bestT, gain: bestGain, recall: nPos ? bestFirP / nPos : 0, fp: nNeg ? bestFirN / nNeg : 0 };
};

export default fitThreshold;
