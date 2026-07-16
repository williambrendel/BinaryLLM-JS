"use strict";

/**
 * @file growClique.js
 * @brief GROW (§5.2) — greedily assemble a discriminative clique `Q` that
 * jointly separates a positive slice `P` from residual negatives `N` better
 * than its bits do alone. The graph is SYNERGY (gain improvement), not
 * co-occurrence (§5, §11).
 *
 * Coherence floor: the m-of-n gate is scored at support ≥ 2 (`sFloor`), so a
 * clique must fire on ≥2 CO-OCCURRING bits. `s=1` is an OR that lets two senses
 * blend into a union (the length-detector failure) — forbidden. This forces a
 * PAIR seed (the smallest clique that fires at support 2); a mixed pair {s1,s2}
 * has zero coverage at support 2 because the senses never co-occur, so growth
 * starts pure and, since adding a foreign-sense bit adds no support-2 coverage,
 * stays pure. Stop when no bit improves the gate (synergy-completeness, NOT a
 * gain plateau — §11).
 *
 * The mask is implicit: a bit with `d(b) ≈ 1` has zero marginal gain and zero
 * synergy, so it is never selected; the returned `Q` equals `p⁺_adapted,k`.
 */

import { buildHitLists, makeCounter, gainSupport, bitLift } from "./synergyGain.js";
import pruneClique from "./pruneClique.js";

const S_FLOOR = 2;

/** Insert `b` into sorted array `Q` (copy), keeping ascending order. */
const insertSorted = (Q, b) => {
  const out = new Array(Q.length + 1);
  let i = 0;
  while (i < Q.length && Q[i] < b) { out[i] = Q[i]; i++; }
  out[i] = b;
  for (let j = i; j < Q.length; j++) out[j + 1] = Q[j];
  return out;
};
const pair = (a, b) => (a < b ? [a, b] : [b, a]);

/**
 * @function growClique
 * @param {Array<number[]>} Pctx - positive slice contexts (sorted bit arrays).
 * @param {Array<number[]>} Nctx - residual negative contexts.
 * @param {object} [opts]
 * @param {number} [opts.kappa=2] - enrichment cutoff; eligible bit iff d(b) > κ.
 * @param {number} [opts.minSupport=2] - min positive contexts a bit must hit.
 * @param {number} [opts.maxBits=64] - safety cap on clique size.
 * @param {number} [opts.topSeed=60] - cap on eligible bits entering pair search.
 * @param {number} [opts.seed] - fixed seed bit (6b init); paired with its best
 *   partner to form the support-2 seed.
 * @param {boolean} [opts.prune=false] - run the deterministic PRUNE mask step on
 *   the grown clique, dropping any bit whose removal doesn't hurt the part's gain
 *   (coattail/common-word bits). GROW rarely adds such bits, but PRUNE is a cheap
 *   safety net and the mask step for proposal-based methods.
 * @param {Iterable<number>} [opts.dropSet] - `D_τ`; these bits are NEVER
 *   eligible, enforcing `Q ⊆ p⁺_glob ∧ M_τ` explicitly (the mask, not left to
 *   `d(b)≈1`, which distorts when `Ā_res` shrinks).
 * @returns {{Q:number[], gain:number, support:number}} the clique (sorted), its
 *   m-of-n gain, and support; `Q=[]` when no coherent support-2 seed pair exists.
 */
export const growClique = (Pctx, Nctx, opts = {}) => {
  const { kappa = 2, minSupport = 2, maxBits = 64, topSeed = 60, seed: fixedSeed, dropSet, prune = false } = opts;
  const nPos = Pctx.length, nNeg = Nctx.length;
  if (nPos === 0) return { Q: [], gain: 0, support: 0 };

  const drop = dropSet instanceof Set ? dropSet : new Set(dropSet || []);
  const hitP = buildHitLists(Pctx), hitN = buildHitLists(Nctx);
  const posCounter = makeCounter(hitP, nPos), negCounter = makeCounter(hitN, nNeg);
  const gv = (Q, sFloor = S_FLOOR) => gainSupport(Q, posCounter, negCounter, nPos, nNeg, sFloor).gain;

  // Eligible bits: not in the drop set, enough positive support, enriched (d > κ).
  const E = [];
  for (const [b, l] of hitP) if (!drop.has(b) && l.length >= minSupport && bitLift(b, hitP, nPos, hitN, nNeg) > kappa) E.push(b);
  if (E.length < 2) return { Q: [], gain: 0, support: 0 };

  // Rank eligible bits by single-bit enrichment to bound the pair search.
  const ranked = E.map((b) => [b, gv([b], 1)]).sort((a, b) => b[1] - a[1]).slice(0, topSeed).map((e) => e[0]);

  // Seed = best support-2 pair (a mixed-sense pair scores ~0: senses don't co-occur).
  let seedQ = null, seedGain = 0, seedSup = S_FLOOR;
  if (fixedSeed !== undefined && E.includes(fixedSeed)) {
    for (const b of ranked) {
      if (b === fixedSeed) continue;
      const g = gainSupport(pair(fixedSeed, b), posCounter, negCounter, nPos, nNeg);
      if (g.gain > seedGain) { seedGain = g.gain; seedQ = pair(fixedSeed, b); seedSup = g.support; }
    }
  } else {
    for (let i = 0; i < ranked.length; i++) for (let j = i + 1; j < ranked.length; j++) {
      const g = gainSupport(pair(ranked[i], ranked[j]), posCounter, negCounter, nPos, nNeg);
      if (g.gain > seedGain) { seedGain = g.gain; seedQ = pair(ranked[i], ranked[j]); seedSup = g.support; }
    }
  }
  if (!seedQ || seedGain <= 0) return { Q: [], gain: 0, support: 0 };

  let Q = seedQ, curGain = seedGain, curSup = seedSup;
  const inQ = new Set(Q);
  while (Q.length < maxBits) {
    let best = -1, bestGain = curGain, bestSup = curSup;
    for (const b of E) {
      if (inQ.has(b)) continue;
      const g = gainSupport(insertSorted(Q, b), posCounter, negCounter, nPos, nNeg);
      if (g.gain > bestGain) { bestGain = g.gain; best = b; bestSup = g.support; }
    }
    if (best < 0 || bestGain <= curGain) break;   // synergy(Q, b*) ≤ 0 → complete
    Q = insertSorted(Q, best); inQ.add(best); curGain = bestGain; curSup = bestSup;
  }
  if (prune && Q.length > 2) { const p = pruneClique(Q, Pctx, Nctx, { minBits: 2 }); Q = p.Q; curGain = p.gain; }
  return { Q, gain: curGain, support: curSup };
};

export default growClique;
