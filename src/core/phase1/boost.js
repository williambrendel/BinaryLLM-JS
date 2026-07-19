"use strict";

/**
 * @file boost.js
 * @brief §7 — the AdaBoost reweighting loop. Each round: rebuild the `w`-weighted affinity; extract the dominant
 * part (replicator); grow its support to a WEAK-learner recall (`recallTau`/`tauFloor`); fit its m-of-n gate;
 * accept iff weighted-recall ≥ 0.5 AND precision ≥ p_min; take the AdaBoost vote α = ½log(rec/(1−rec));
 * down-weight covered positives and renormalize. Stop on union recall > 0.99 or the dual bound.
 */

import buildAffinity from "./affinity.js";
import replicate from "./replicator.js";
import mfit from "./mfit.js";
import andCount from "../math/sparse/andCount.js";

/**
 * @function boost
 * @param {Array<number[]>} A - positive contexts (train).
 * @param {object} neg - buildNegSet output (frozen confusables + stats).
 * @param {object} [opts] @param {number} [opts.rho=40] @param {number} [opts.maxRounds=50]
 *   @param {"exp"|"dc"} [opts.solver="exp"] @param {number} [opts.alphaR] @param {boolean} [opts.recallTau=false]
 *   @param {number} [opts.tauFloor=0.5] @param {number} [opts.suppPatience]
 * @returns {{G:Array, pMin:number, unionRecall:number, curve:Array, rounds:number, stop:string}}
 */
export const boost = (A, neg, opts = {}) => {
  const { rho = 40, maxRounds = 50, solver = "exp", alphaR, epsMin = 1e-10, recallTau = false, tauFloor = 0.5, suppPatience, recallFloor = 0.5, softFloor = false, minRecall = 0.25, andObjective = false } = opts;
  const nA = A.length, pMin = nA / (nA + neg.negN);
  let w = new Float64Array(nA).fill(1 / nA);
  const G = [], curve = [], covered = new Array(nA).fill(false);
  let stop = "max_rounds", noValidM = 0;

  for (let round = 0; round < maxRounds; round++) {
    const { u, edges } = buildAffinity(A, w, neg, { alphaR, andObjective });
    const r = replicate(u, edges, { solver, rho, ...(suppPatience ? { suppPatience } : {}) });
    if (r.Q.length === 0) { stop = "empty_part"; break; }

    // recallTau: grow Q from the top-x* bits until WEIGHTED recall (m=1, current AdaBoost distribution) just
    // clears tauFloor — a genuine WEAK learner, leaving the rest for reweighting. Else the full dominant set.
    let Qbits;
    if (recallTau) {
      const order = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).sort((a, b) => b[1] - a[1]).map((e) => e[0]);
      const cov = new Uint8Array(nA); let wcov = 0; Qbits = [];      // w is renormalized each round ⇒ Σw=1
      for (const b of order) {
        Qbits.push(b);
        for (let i = 0; i < nA; i++) if (!cov[i]) { const yi = A[i]; for (let t = 0; t < yi.length; t++) if (yi[t] === b) { cov[i] = 1; wcov += w[i]; break; } }
        if (wcov >= tauFloor) break;
      }
    } else Qbits = r.Q.map((p) => neg.pPlus[p]);
    Qbits.sort((a, b) => a - b);
    if (Qbits.length === 0) { stop = "empty_part"; break; }

    const fit = mfit(Qbits, A, w, neg.Neg, { recallFloor, softFloor, minRecall });
    if (!fit.valid) { noValidM++; stop = "no_valid_m"; break; }            // even m=1 (OR) under recallFloor ⇒ multimodal/diffuse class — lower recallFloor to admit sub-0.5 sense-cliques as weak learners
    // dual bound: a below-p_min part means AdaBoost's guarantee is spent → stop. But for a multimodal/diffuse class
    // the per-sense cliques can each sit below p_min (each fires on the OTHER senses' confusables); breaking there
    // discards the whole class (music: 0 parts). Under softFloor, ADMIT them as weak learners — union_recall/max_rounds
    // still bound the loop, and the outer prefix early-stop + θ-head control the resulting FP.
    if (fit.precision < pMin && !softFloor) { stop = "dual_bound_precision"; break; }

    // AdaBoost vote with ε-clip (perfect-recall part else gives α=∞). Default ε=1−recall is only valid for recall≥0.5;
    // a softFloor part (recall<0.5) would get a NEGATIVE α, which INVERTS a one-sided OR gate (fires⇒evidence-FOR the
    // class) into evidence-against — flipping both the head and the reweighting. Under softFloor use the proper BALANCED
    // error (½ missed-positives + ½ false-positives) so a low-FP part earns a POSITIVE vote even at sub-0.5 recall.
    let eps;
    if (softFloor) {
      const tp = fit.recall * nA, fp = fit.precision > 0 ? tp * (1 - fit.precision) / fit.precision : nA;
      eps = 0.5 * (1 - fit.recall) + 0.5 * Math.min(1, fp / (neg.negN || 1));
    } else eps = 1 - fit.recall;
    eps = Math.min(Math.max(eps, epsMin), 1 - epsMin);
    const alpha = 0.5 * Math.log((1 - eps) / eps);
    G.push({ Qbits, m: fit.m, alpha, recall: fit.recall, precision: fit.precision });

    // standard AdaBoost reweight: down-weight covered positives, renormalize
    let Wnew = 0;
    for (let i = 0; i < nA; i++) { const h = andCount(A[i], Qbits) >= fit.m ? 1 : 0; if (h) covered[i] = true; w[i] *= Math.exp(-alpha * h); Wnew += w[i]; }
    for (let i = 0; i < nA; i++) w[i] /= Wnew;

    const unionRecall = covered.filter(Boolean).length / nA;
    curve.push({ round: round + 1, alpha, recall: fit.recall, precision: fit.precision, m: fit.m, size: Qbits.length, unionRecall });
    if (unionRecall > 0.99) { stop = "union_recall"; break; }
  }
  return { G, pMin, unionRecall: covered.filter(Boolean).length / nA, curve, rounds: G.length, stop, noValidM };
};

export default boost;
