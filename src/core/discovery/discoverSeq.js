"use strict";

/**
 * @file discoverSeq.js
 * @brief 6a — sequential, λ-driven discovery (K emergent). The fallback /
 * cross-check for 6b. Grows parts on the residual `R`, hard-peels the positives
 * a part claims, and residualizes the negatives (`Ā_fired`) so each part is
 * charged only for the NEW ensemble false positives it introduces (§5.0/§6a).
 *
 * Known bias (why it is the fallback, not the default): part 1 grows on all of
 * A, skimming the highest-mass sense first; later parts inherit a part-1-shaped
 * residual. `K` is emergent, tuned via `λ`.
 */

import growClique from "./growClique.js";
import peelNegative from "./peelNegative.js";
import fitThreshold from "./thresholdFit.js";
import { adaptedGateScore } from "../gate/adaptedGate.js";

/**
 * @function discoverSeq
 * @param {Array<number[]>} A - positive contexts.
 * @param {Array<number[]>} AbarH - hard negatives (§3).
 * @param {number[]} Dtau - shared global drop set (§3).
 * @param {object} [opts]
 * @param {number} [opts.lambda=0.02] - gain floor; sets emergent K.
 * @param {number} [opts.m=2] - clique-support for A_Q / FP.
 * @param {number} [opts.kappa=2] - enrichment cutoff.
 * @param {number} [opts.sMin=3] - residual size floor.
 * @param {number} [opts.maxK=64] - safety cap.
 * @param {number} [opts.minCover=2] - a part must peel ≥ this many positives; when
 *   the best remaining clique explains fewer, the residual is exhausted of real
 *   parts and the loop stops. Prevents the low-λ peel cascade that spawns
 *   hundreds of tiny single-context gates (the seq high-K runaway).
 * @param {Array<number[]>} [opts.thrNeg] - broad negative pool for fitting `t_k`.
 *   `Ā_h` drives GROW/peel (the discrimination signal), but the ensemble FP is
 *   measured on the FULL negative surface — gates false-fire on EASY negatives
 *   outside `Ā_h` that `Ā_fired` never sees, so |⋃FP| grows with K. Fitting `t_k`
 *   against this broader pool controls the actual FP surface (§5.0).
 * @returns {{Dtau:number[], gates:Array<{Q:number[],pMinus:number[],t:number,support:number,gain:number,nFP:number}>, pMinusGlobalSize:number}}
 */
export const discoverSeq = (A, AbarH, Dtau, opts = {}) => {
  const { lambda = 0.02, m = 2, kappa = 2, sMin = 3, maxK = 64, thrNeg = null, minCover = 2 } = opts;
  const gates = [];
  let R = A.slice();
  const fired = new Array(AbarH.length).fill(false);        // Ā_fired membership

  // global negative-channel size, re-based to Ā_h (§9.3 comparison baseline)
  const pGlob = peelNegative([...new Set(A.flat())].sort((a, b) => a - b), AbarH, A, { Dtau, kappa, m });
  const pMinusGlobalSize = pGlob.pMinus.length;

  while (R.length >= sMin && gates.length < maxK) {
    const AbarRes = AbarH.filter((_, j) => !fired[j]);       // residual negs: NEW FPs only
    if (AbarRes.length === 0) break;

    const { Q, gain, support } = growClique(R, AbarRes, { kappa, minSupport: m, dropSet: Dtau });
    if (Q.length === 0 || gain < lambda) break;              // λ sets K

    const { pMinus, nFP } = peelNegative(Q, AbarRes, R, { Dtau, kappa, m });
    const scoreFn = (x) => adaptedGateScore(x, { pPlusAdapted: Q, pMinus, dropSet: Dtau });
    // t_k controls the FP surface: fit against the broad pool when given, else Ā_res.
    const { t } = fitThreshold(scoreFn, R, thrNeg || AbarRes);

    // HARD peel positives; a part must cover ≥ minCover of them or the residual
    // is exhausted (stops the low-λ tiny-gate runaway).
    const kept = R.filter((y) => scoreFn(y) <= t);
    const cover = R.length - kept.length;
    if (cover < minCover) break;

    gates.push({ Q, pMinus, t, support, gain, nFP, cover });
    R = kept;
    for (let j = 0; j < AbarH.length; j++) if (!fired[j] && scoreFn(AbarH[j]) > t) fired[j] = true;
  }

  return { Dtau, gates, pMinusGlobalSize };
};

export default discoverSeq;
