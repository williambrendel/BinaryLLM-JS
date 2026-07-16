"use strict";

/**
 * @file pruneClique.js
 * @brief PRUNE — the per-part deterministic mask step. Holding a part `Q` fixed,
 * greedily drop the bit whose removal IMPROVES-OR-DOESN'T-HURT the part's own
 * m-of-n gain, until the best removal would hurt.
 *
 *   PRUNE(Q, P, N):
 *     repeat:
 *       b* = argmax_{b∈Q} GAIN(Q ∖ b ; P, N)
 *       if GAIN(Q ∖ b*) < GAIN(Q): break        # best removal hurts → done
 *       Q ← Q ∖ b*
 *     return Q
 *
 * Why it works where the Monte-Carlo edge sampler fails: a common-word bit that
 * rode into `Q` on its neighbours' coattails contributes ~0 to the part's gain,
 * so its removal-Δ ≥ 0 and it is dropped. The sampler couldn't isolate it —
 * good block-mates inflate its edge mass (its per-bit "strength" ≈ a driver's,
 * pull-coefficient high). Asking deterministically "does THIS bit earn its place
 * in THIS part" exposes it trivially. Companion to {@link module:core/discovery/growClique}.
 */

import { buildHitLists, makeCounter, gainSupport } from "./synergyGain.js";

const S_FLOOR = 2;
const without = (Q, b) => Q.filter((x) => x !== b);

/**
 * @function pruneClique
 * @param {number[]} Q - the part (sorted bit ids) to prune.
 * @param {Array<number[]>} Pctx - the part's positive slice.
 * @param {Array<number[]>} Nctx - residual negatives.
 * @param {object} [opts]
 * @param {number} [opts.minBits=2] - never prune below this (the support floor).
 * @returns {{Q:number[], dropped:number[], gain:number}} the pruned clique
 *   (sorted), the bits removed (in drop order), and the final gain.
 */
export const pruneClique = (Q, Pctx, Nctx, opts = {}) => {
  const { minBits = S_FLOOR } = opts;
  const nPos = Pctx.length, nNeg = Nctx.length;
  const posCounter = makeCounter(buildHitLists(Pctx), nPos), negCounter = makeCounter(buildHitLists(Nctx), nNeg);
  const gv = (q) => gainSupport(q, posCounter, negCounter, nPos, nNeg).gain;

  let cur = Q.slice(); const dropped = [];
  let curGain = gv(cur);
  while (cur.length > minBits) {
    let best = -1, bestGain = -Infinity;
    for (const b of cur) { const g = gv(without(cur, b)); if (g > bestGain) { bestGain = g; best = b; } }
    if (best < 0 || bestGain < curGain) break;      // best removal hurts → each remaining bit earns its place
    cur = without(cur, best); dropped.push(best); curGain = bestGain;
  }
  return { Q: cur, dropped, gain: curGain };
};

export default pruneClique;
