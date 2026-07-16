"use strict";

/**
 * @file fit.js
 * @brief §8 — the deployed phase-1 fit: `keep+earlystop`, the CV-validated config.
 *
 * Composes the validated pipeline into one call:
 *   1. Split A → train / val (val holds out the tail of A).
 *   2. buildNegSet on TRAIN with maskNodes=∅ (KEEP: common bits stay affinity nodes for every target —
 *      recall lives in their unary channel; their edges scaffold precision).
 *   3. boost on TRAIN (train-pure AdaBoost: train α, train reweighting, coverage-greedy ordering).
 *   4. Validation prefix-selection: keep the part-prefix G[:r*] maximizing val (recall − confFP). This is
 *      OUTER early-stopping only — it never enters the boosting objective, so the AdaBoost guarantee holds
 *      (it leverages the frequent-first ordering; the overfit tail lives in the unary and is what we trim).
 *
 * Why per-target ban/keep was dropped: bank's ban-preference was slice noise (5-fold CV: ban is fragile —
 * state 18±22%); keep+es is the only config stable AND strong on all three (bank 58/9.3, state 77.7/15.3,
 * time 81.5/9.3). See docs/phase1_report.md §14.
 */

import buildNegSet from "./negSet.js";
import boost from "./boost.js";
import adaptiveCore from "./adaptiveCore.js";
import makeHead from "./head.js";

const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);

/**
 * @function fitClass
 * @param {Array<number[]>} A - positive contexts for one target class.
 * @param {Array<number[]>} negPool - non-target contexts to draw confusables from.
 * @param {Iterable<number>} Mglob - global common-bit (SELECT) mask.
 * @param {object} [opts]
 * @param {number} [opts.valFrac=0.25] - fraction of A held out (tail) for prefix-selection.
 * @param {number} [opts.rho=80] @param {"std"|"exp"|"hybrid"} [opts.solver="exp"] @param {number} [opts.delta=0.05]
 * @param {number} [opts.rho],... - remaining opts forwarded to boost.
 * @returns {{head:object, G:Array, rStar:number, roundsTotal:number, neg:object,
 *   valRecall:number, valConfFP:number}}
 */
export const fitClass = (A, negPool, Mglob, opts = {}) => {
  // small classes over-fire (few positives → noisy val θ, loose parts) → scale the FP penalty λ up as |A|
  // shrinks: λ = clamp(1500/|A|, 1, 3). bank(500)→3, state/time→1. Overridable via opts.fpWeight.
  const autoFp = Math.min(3, Math.max(1, 1500 / A.length));
  // solver="dc" (matrix-split) is the default: parameter-free, converges to the true optimum, negatives-native,
  // per-iter cost ≈ exp (M⁺/M⁻ partition the nnz). Parts are solver-invariant so metrics ≈ the exp-CV'd config
  // (verified within CV noise); a full re-CV to refresh the headline numbers is pending.
  // suppPatience=3: support-stability early-stop in the replicator (validated ~2.6× faster, quality-neutral —
  // the support set settles long before the weights fully converge, and we only use the ranking).
  const { valFrac = 0.25, rho = 80, solver = "dc", delta = 0.05, fpWeight = autoFp, tuneTheta = true, earlyStop = true, recallTau = true, tauFloor = 0.6, suppPatience = 3, maskNodes = new Set(), core = "boost", ...boostOpts } = opts;
  const nTr = Math.max(1, Math.floor(A.length * (1 - valFrac)));
  const tr = A.slice(0, nTr), val = A.slice(nTr);

  const neg = buildNegSet(tr, negPool, Mglob, { delta, maskNodes, negIndex: opts.negIndex, negLen: opts.negLen });  // node mask (default ∅ = keep); optional shared inverted index (§opt)
  // recallTau: extract each part as a WEAK learner (support grown from top-x* until weighted recall just
  // clears 50%), not the full strong dominant set — the correct weak-classifier input to AdaBoost.
  // core="adaptive" (§8 candidate): FP-aware concentration-driven disjoint cores instead of the AdaBoost loop.
  const { G } = core === "adaptive"
    ? adaptiveCore(tr, neg, { rho, solver, ...boostOpts })
    : boost(tr, neg, { rho, solver, recallTau, tauFloor, ...boostOpts });

  // outer early-stop: prefix maximizing val (recall − λ·confFP), λ=fpWeight; ties → shortest prefix.
  // earlyStop=false ⇒ keep the FULL discovered set (no trim).
  let rStar = G.length, vr0 = 0, vf0 = 0;
  if (earlyStop) {
    let best = -Infinity;
    for (let r = 1; r <= G.length; r++) {
      const h = makeHead(G.slice(0, r));
      const vr = rate((y) => h.fires(y), val), vf = rate((y) => h.fires(y), neg.Neg);
      if (vr - fpWeight * vf > best) { best = vr - fpWeight * vf; rStar = r; vr0 = vr; vf0 = vf; }
    }
  }
  const G2 = G.slice(0, rStar);
  let head = makeHead(G2);

  // §7.2 operating point: the FP floor is low-m parts firing loosely, redundantly — not fixable by part
  // removal, only by the α-sum threshold. Tune θ on VALIDATION (same recall−λ·FP objective), so the
  // recall/FP trade rides on the honest val ROC, not the ½Σα heuristic. λ>1 buys precision.
  if (tuneTheta) {
    const Sval = val.map((y) => head.S(y)), Sneg = neg.Neg.map((y) => head.S(y));
    const cands = [...new Set([0, ...Sval, ...Sneg])].sort((a, b) => a - b);
    let bTh = head.theta, bObj = -Infinity;
    for (const c of cands) {
      const vr = Sval.length ? Sval.filter((s) => s > c).length / Sval.length : 0;
      const vf = Sneg.length ? Sneg.filter((s) => s > c).length / Sneg.length : 0;
      if (vr - fpWeight * vf > bObj) { bObj = vr - fpWeight * vf; bTh = c; }
    }
    head = makeHead(G2, bTh);
  }
  return { head, G: G2, rStar, roundsTotal: G.length, neg, theta: head.theta, valRecall: vr0, valConfFP: vf0 };
};

export default fitClass;
