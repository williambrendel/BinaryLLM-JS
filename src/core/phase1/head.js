"use strict";

/**
 * @file head.js
 * @brief §7.2 — the deployed head: the α-weighted AdaBoost SUM of the m-of-n
 * weak parts (a LINEAR combiner, distinct from the gain-formulation's max-pool).
 *   S_τ(x) = Σ_k α_k·h_k(x),   h_k(x) = 1[ |Q_k ∧ x| ≥ m_k ]
 *   fires(x) = S_τ(x) > θ,     θ = ½ Σ_k α_k  (default; tunable)
 * Also exposes a max-pool readout over the SAME parts for the §11 head A/B.
 */

import andCount from "../math/sparse/andCount.js";

/**
 * @function makeHead
 * @param {Array<{Qbits:number[], m:number, alpha:number}>} G - phase-1 parts.
 * @param {number} [theta] - fire threshold; default ½Σα.
 * @returns {{S:Function, fires:Function, theta:number, maxpoolFires:Function}}
 */
export const makeHead = (G, theta) => {
  const th = theta ?? 0.5 * G.reduce((a, g) => a + g.alpha, 0);
  const inv = new Map(); G.forEach((g, k) => { for (const b of g.Qbits) { let l = inv.get(b); if (!l) { l = []; inv.set(b, l); } l.push(k); } });
  const active = (x) => { const s = new Set(); for (const b of x) { const l = inv.get(b); if (l) for (const k of l) s.add(k); } return s; };

  const S = (x) => { let s = 0; for (const k of active(x)) if (andCount(x, G[k].Qbits) >= G[k].m) s += G[k].alpha; return s; };
  return {
    theta: th,
    S,
    fires: (x) => S(x) > th,
    // §11 head A/B: max-pool (OR) over the same parts — fires if ANY part fires
    maxpoolFires: (x) => { for (const k of active(x)) if (andCount(x, G[k].Qbits) >= G[k].m) return true; return false; },
  };
};

export default makeHead;
