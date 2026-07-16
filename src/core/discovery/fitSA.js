"use strict";

/**
 * @file fitSA.js
 * @brief 6d — simulated annealing over the bit→part grouping, objective = the
 * §5.0 ENSEMBLE max-pool gain (residualized), with split/merge moves. Optimizes
 * the grouping GLOBALLY instead of greedily, sidestepping GROW's greedy synergy
 * to test whether greedy clique-growth left value on the table.
 *
 * This is a METHOD, not a baseline (§6d, §11): if it beats 6a/6b the finding is
 * "greedy synergy is suboptimal — global grouping helps," NOT "coherence is
 * worthless" (6c-ii is that control). Objective reads gain and Ā_h; a baseline
 * must not (§11).
 */

import peelNegative from "./peelNegative.js";
import fitThreshold from "./thresholdFit.js";
import { adaptedGateScore } from "../gate/adaptedGate.js";
import { buildHitLists, makeCounter, gainSupport } from "./synergyGain.js";
import andCount from "../math/sparse/andCount.js";

const lcg = (seed) => () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 0x100000000;

const adaptedPositive = (A, Dset) => { const s = new Set(); for (const y of A) for (const b of y) if (!Dset.has(b)) s.add(b); return [...s]; };

/**
 * @function fitSA
 * @param {Array<number[]>} A @param {Array<number[]>} AbarH @param {number[]} Dtau
 * @param {number} K @param {object} [opts]
 * @returns {{Dtau:number[], gates:Array, algo:string}}
 */
export const fitSA = (A, AbarH, Dtau, K, opts = {}) => {
  const { m = 2, kappa = 2, sMin = 3, seed = 1, iters = 3000, t0 = 0.5, cool = 0.999 } = opts;
  const rnd = lcg(seed >>> 0 || 1);
  const Dset = new Set(Dtau);
  const Pplus = adaptedPositive(A, Dset);
  const nPos = A.length, nNeg = AbarH.length;

  const hitP = buildHitLists(A), hitN = buildHitLists(AbarH);
  const posCounter = makeCounter(hitP, nPos), negCounter = makeCounter(hitN, nNeg);

  // grouping: bit index → chunk id (single membership for SA simplicity)
  let owner = Pplus.map((_, i) => i % K);
  const chunkBits = (own, k) => { const q = []; for (let i = 0; i < Pplus.length; i++) if (own[i] === k) q.push(Pplus[i]); return q.sort((a, b) => a - b); };
  // ensemble gain = Σ_k GAIN(Q_k ; A, Ā_res^k). Approximate Ā_res^k by full Ā_h
  // (SA over the sum of per-chunk gains); leave-one-out refit happens on readout build.
  const objective = (own) => { let g = 0; for (let k = 0; k < K; k++) { const Q = chunkBits(own, k); if (Q.length >= 2) g += gainSupport(Q, posCounter, negCounter, nPos, nNeg).gain; } return g; };

  let cur = objective(owner), T = t0;
  for (let it = 0; it < iters; it++) {
    const i = Math.floor(rnd() * Pplus.length), prev = owner[i];
    owner[i] = Math.floor(rnd() * K);                       // move-bit (split/merge emerge from reassignment)
    const next = objective(owner);
    if (next >= cur || rnd() < Math.exp((next - cur) / T)) cur = next; else owner[i] = prev;
    T *= cool;
  }

  // Materialize gates with leave-one-out residual negatives (§5.0).
  const raw = Array.from({ length: K }, (_, k) => chunkBits(owner, k)).filter((Q) => Q.length >= 2);
  const gates = [];
  const scoreG = (Q, pMinus) => (x) => adaptedGateScore(x, { pPlusAdapted: Q, pMinus, dropSet: Dtau });
  // provisional gates (for leave-one-out firing)
  let prov = raw.map((Q) => { const P = A.filter((y) => andCount(y, Q) >= m); const { pMinus } = peelNegative(Q, AbarH, P.length ? P : A, { Dtau, kappa, m }); const { t } = fitThreshold(scoreG(Q, pMinus), P.length ? P : A, AbarH); return { Q, pMinus, t }; });
  raw.forEach((Q, k) => {
    const P = A.filter((y) => andCount(y, Q) >= m);
    if (P.length < sMin) return;
    const AbarRes = AbarH.filter((y) => !prov.some((g, kk) => kk !== k && scoreG(g.Q, g.pMinus)(y) > g.t));
    const { pMinus } = peelNegative(Q, AbarRes.length ? AbarRes : AbarH, P, { Dtau, kappa, m });
    const { t } = fitThreshold(scoreG(Q, pMinus), P, AbarRes.length ? AbarRes : AbarH);
    gates.push({ Q, pMinus, t, size: P.length });
  });
  return { Dtau, gates, algo: "sa_gain" };
};

export default fitSA;
