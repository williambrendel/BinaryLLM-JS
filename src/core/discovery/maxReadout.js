"use strict";

/**
 * @file maxReadout.js
 * @brief The §7 final scorer — MAX POOL over the gate set, evaluated on margins.
 *
 * Each gate `k` is `(Q_k, p⁻_k, t_k)` plus the shared `D_τ`; its score is §4:
 *   score_k(x) = ( |Q_k ∧ x| − |p⁻_k ∧ x| ) / ( |x| − |D_τ ∧ x| )    (= adaptedGateScore)
 * The weak classifier fires iff `score_k(x) > t_k`, so the pool is over the
 * MARGIN `m_k = score_k − t_k`, not the raw score (§7, §11):
 *   S_τ(x) = max_k m_k(x),   fires(x) = S_τ(x) > 0 ⟺ ⋁_k(score_k > t_k).
 *
 * Sparse inference: an inverted index `bit → {k : bit ∈ Q_k}` restricts the max
 * to gates whose clique intersects `x` (an inactive gate has `|Q_k ∧ x| = 0`, so
 * its numerator ≤ 0 and it cannot be the max among firing gates).
 */

import { adaptedGateScore } from "../gate/adaptedGate.js";

/**
 * @function makeReadout
 * @param {Array<{Q:number[], pMinus:number[], t:number}>} gates - the part gates.
 * @param {number[]} Dtau - shared global drop set (sorted).
 * @returns {{score:Function, margin:Function, S:Function, fires:Function, kStar:Function, activeGates:Function}}
 */
export const makeReadout = (gates, Dtau) => {
  const inv = new Map();                                   // bit → [gate indices]
  gates.forEach((g, k) => { for (const b of g.Q) { let l = inv.get(b); if (!l) { l = []; inv.set(b, l); } l.push(k); } });

  const score = (x, k) => adaptedGateScore(x, { pPlusAdapted: gates[k].Q, pMinus: gates[k].pMinus, dropSet: Dtau });
  const margin = (x, k) => score(x, k) - gates[k].t;

  const activeGates = (x) => {
    const seen = new Set();
    for (const b of x) { const l = inv.get(b); if (l) for (const k of l) seen.add(k); }
    return [...seen];
  };

  // max margin over gates whose clique intersects x; −Infinity if none active.
  const evalMax = (x) => {
    let best = -Infinity, bk = -1;
    for (const k of activeGates(x)) { const mk = margin(x, k); if (mk > best) { best = mk; bk = k; } }
    return { S: best, kStar: bk };
  };

  return {
    score, margin, activeGates,
    S: (x) => evalMax(x).S,
    fires: (x) => evalMax(x).S > 0,
    kStar: (x) => evalMax(x).kStar,
  };
};

export default makeReadout;
