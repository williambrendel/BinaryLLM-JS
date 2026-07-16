"use strict";

/**
 * @file metrics.js
 * @brief §9.1 primary metrics, computed on the §7 max-pool `fires(x)` over a
 * positive and a negative pool. precision / recall / accuracy / FP% — the
 * numbers the whole exercise targets, never scored on a single part.
 */

/**
 * @function poolMetrics
 * @param {Function} fires - `x → boolean` (the §7 max-pool readout).
 * @param {Array<number[]>} pos - positive contexts.
 * @param {Array<number[]>} neg - negative contexts.
 * @returns {{tp:number, fp:number, fn:number, tn:number, precision:number,
 *   recall:number, accuracy:number, fpRate:number}}
 */
export const poolMetrics = (fires, pos, neg) => {
  let tp = 0, fp = 0;
  for (const y of pos) if (fires(y)) tp++;
  for (const y of neg) if (fires(y)) fp++;
  const fn = pos.length - tp, tn = neg.length - fp;
  return {
    tp, fp, fn, tn,
    precision: tp + fp > 0 ? tp / (tp + fp) : 0,
    recall: pos.length > 0 ? tp / pos.length : 0,
    accuracy: (tp + tn) / (pos.length + neg.length),
    fpRate: neg.length > 0 ? fp / neg.length : 0,
  };
};

export default poolMetrics;
