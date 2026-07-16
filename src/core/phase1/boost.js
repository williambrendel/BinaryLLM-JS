"use strict";

/**
 * @file boost.js
 * @brief §7 — the AdaBoost reweighting loop (NO peel, NO residual). Each round:
 * rebuild `w`-weighted `u`,`M`; extract the dominant part (replicator); fit its
 * m-of-n gate; accept iff weighted-recall ≥ 0.5 AND precision ≥ p_min; take the
 * AdaBoost vote α_k = ½log(rec/(1−rec)); down-weight covered positives and
 * renormalize. Stop on true (unweighted) union recall > 0.99 or the dual bound.
 */

import buildAffinity from "./affinity.js";
import replicate from "./replicator.js";
import mfit from "./mfit.js";
import andCount from "../math/sparse/andCount.js";

/**
 * @function boost
 * @param {Array<number[]>} A - positive contexts (train).
 * @param {object} neg - buildNegSet output (frozen confusables + stats).
 * @param {object} [opts] @param {number} [opts.rho=20] @param {number} [opts.maxRounds=50]
 * @param {"std"|"exp"|"hybrid"} [opts.solver="exp"] @param {number} [opts.alphaR]
 * @returns {{G:Array, pMin:number, unionRecall:number, curve:Array, rounds:number, stop:string}}
 */
export const boost = (A, neg, opts = {}) => {
  const { rho = 40, maxRounds = 50, solver = "exp", alphaR, epsMin = 1e-10, commonCoef = 1, commonCoefU, commonCoefM, scaffoldOnly = false, coreT = 0, recallTau = false, tauFloor = 0.5, peel = false, peelCoef = 0, maxIter, replEps, suppPatience, shiftConst, shiftDiag, eta, expU, dt, normEdges, normT, domset = false, domShift = 1, metric, metricScale } = opts;
  const nA = A.length, pMin = nA / (nA + neg.negN);
  let w = new Float64Array(nA).fill(1 / nA);
  const G = [], curve = [], covered = new Array(nA).fill(false);
  const peeled = new Set();
  let stop = "max_rounds", noValidM = 0;
  const prof = opts.prof || null, nowMs = () => Number(process.hrtime.bigint()) / 1e6;   // optional phase timing

  for (let round = 0; round < maxRounds; round++) {
    let ts = prof && nowMs();
    const { u, edges } = buildAffinity(A, w, neg, { alphaR, commonSet: neg.DtauSel, commonCoef, commonCoefU, commonCoefM, exclude: peel ? peeled : undefined, excludeCoef: peelCoef, normEdges, normT, metric, metricScale });
    if (prof) { prof.affinity += nowMs() - ts; prof.edges = edges.m.length; ts = nowMs(); }
    // domset: classic pure-M dominant set — drop the unary u and the −ρ spread regularizer, shift M to nonneg
    // (Pelillo α=|min M|≈8), std replicator ⇒ a TINY tight clique (recall often <50%), to be augmented below.
    let uUse = u, rhoUse = rho, solverUse = solver, extra = {};
    if (domset) { uUse = new Float64Array(u.length); rhoUse = 0; solverUse = "std"; let mM = 0; for (let e = 0; e < edges.m.length; e++) if (edges.m[e] < mM) mM = edges.m[e]; extra = { shiftConst: (Math.abs(mM) + 1e-9) * domShift }; }   // ×domShift: larger ⇒ flatter ⇒ bigger core
    const r = replicate(uUse, edges, { solver: solverUse, rho: rhoUse, ...extra, ...(maxIter ? { maxIter } : {}), ...(replEps ? { eps: replEps } : {}), ...(suppPatience ? { suppPatience } : {}), ...(!domset && shiftConst !== undefined ? { shiftConst } : {}), ...(shiftDiag ? { shiftDiag } : {}), ...(eta ? { eta } : {}), ...(expU ? { expU } : {}), ...(dt ? { dt } : {}) });
    if (prof) { prof.replicate += nowMs() - ts; prof.iters += r.iters; prof.supp += r.Q.length; prof.n = u.length; ts = nowMs(); }
    if (r.Q.length === 0) { stop = "empty_part"; break; }
    // Support extraction. recallTau: grow Q from the top-x* bits until WEIGHTED recall (m=1, current
    // AdaBoost distribution) just clears tauFloor — a genuine WEAK learner (~50%), leaving the rest for
    // reweighting. Else generic tauSupp (full dominant set = strong part), optionally core-trimmed.
    let Qbits;
    if (recallTau) {
      // order = top-x* bits by replicator weight. domset: the clique is tiny (can't reach tauFloor within it),
      // so EXTEND beyond it — append the non-clique P⁺ bits ranked by unary log-odds (most discriminative first).
      let order = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).sort((a, b) => b[1] - a[1]).map((e) => e[0]);
      if (domset) { const inCl = new Set(r.Q); const ext = []; for (let p = 0; p < neg.pPlus.length; p++) if (!inCl.has(p)) ext.push([neg.pPlus[p], u[p]]); ext.sort((a, b) => b[1] - a[1]); order = order.concat(ext.map((e) => e[0])); }
      const cov = new Uint8Array(nA); let wcov = 0; Qbits = [];      // w is renormalized each round ⇒ Σw=1
      for (const b of order) {
        Qbits.push(b);
        for (let i = 0; i < nA; i++) if (!cov[i]) { const yi = A[i]; for (let t = 0; t < yi.length; t++) if (yi[t] === b) { cov[i] = 1; wcov += w[i]; break; } }
        if (wcov >= tauFloor) break;
      }
    } else if (coreT > 0) { const k = r.Q.length; Qbits = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).filter(([, wt]) => wt * k >= coreT).map(([b]) => b); }
    else Qbits = r.Q.map((p) => neg.pPlus[p]);
    // scaffoldOnly: common bits BRIDGE in M but are stripped from the part → never a gate requirement
    if (scaffoldOnly) Qbits = Qbits.filter((b) => !neg.DtauSel.has(b));
    Qbits.sort((a, b) => a - b);
    if (prof) { prof.support += nowMs() - ts; ts = nowMs(); }
    if (Qbits.length === 0) { stop = "empty_part"; break; }
    const fit = mfit(Qbits, A, w, neg.Neg);
    if (prof) { prof.mfit += nowMs() - ts; ts = nowMs(); }
    if (!fit.valid) { noValidM++; stop = "no_valid_m"; break; }            // support ceiling < 0.5 ⇒ ρ too small (§6)
    if (fit.precision < pMin) { stop = "dual_bound_precision"; break; }

    // AdaBoost vote with ε-clip (perfect-recall part else gives α=∞); recall≥0.5 already (fit.valid)
    const eps = Math.min(Math.max(1 - fit.recall, epsMin), 1 - epsMin);
    const alpha = 0.5 * Math.log((1 - eps) / eps);
    G.push({ Qbits, m: fit.m, alpha, recall: fit.recall, precision: fit.precision });
    if (peel) for (const b of Qbits) peeled.add(b);                        // claim these bits — peel next round

    // standard AdaBoost reweight: down-weight covered positives, renormalize
    let Wnew = 0;
    for (let i = 0; i < nA; i++) { const h = andCount(A[i], Qbits) >= fit.m ? 1 : 0; if (h) covered[i] = true; w[i] *= Math.exp(-alpha * h); Wnew += w[i]; }
    for (let i = 0; i < nA; i++) w[i] /= Wnew;
    if (prof) prof.reweight += nowMs() - ts;

    const unionRecall = covered.filter(Boolean).length / nA;
    curve.push({ round: round + 1, alpha, recall: fit.recall, precision: fit.precision, m: fit.m, size: Qbits.length, unionRecall, cliffGap: r.cliffGap, recallAt1: fit.recallAt1 });
    if (unionRecall > 0.99) { stop = "union_recall"; break; }
  }
  return { G, pMin, unionRecall: covered.filter(Boolean).length / nA, curve, rounds: G.length, stop, noValidM };
};

export default boost;
