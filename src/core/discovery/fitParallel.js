"use strict";

/**
 * @file fitParallel.js
 * @brief 6b — parallel, fixed-K discovery (the default). No residual ordering,
 * no first-mover: K parts grow on equal footing via a discriminative EM. The
 * E-step assigns each positive to the part of maximum MARGIN (argmax over
 * classifiers, NOT k-means over a metric — §6b/§11). The M-step refits each
 * part against LEAVE-ONE-OUT residual negatives computed Jacobi-style (from the
 * previous round's gates), so parts stay decoupled and no first-mover bias
 * creeps back (§6b).
 */

import growClique from "./growClique.js";
import peelNegative from "./peelNegative.js";
import fitThreshold from "./thresholdFit.js";
import { adaptedGateScore } from "../gate/adaptedGate.js";
import { buildHitLists, makeCounter, gainSupport, bitLift } from "./synergyGain.js";
import andCount from "../math/sparse/andCount.js";

const scoreOf = (gate, Dtau, x) => adaptedGateScore(x, { pPlusAdapted: gate.Q, pMinus: gate.pMinus, dropSet: Dtau });
const marginOf = (gate, Dtau, x) => (gate.empty ? -Infinity : scoreOf(gate, Dtau, x) - gate.t);

/** Eligible bits (∉ D_τ, support ≥ minSupport, d > κ) over (A, Ā_h), ranked by
 * single-bit gain — the pool re-seeding draws fresh seeds from. */
const rankedEligibleBits = (A, AbarH, { kappa, minSupport, Dtau }) => {
  const nPos = A.length, nNeg = AbarH.length;
  const hitP = buildHitLists(A), hitN = buildHitLists(AbarH);
  const posCounter = makeCounter(hitP, nPos), negCounter = makeCounter(hitN, nNeg), Dset = new Set(Dtau);
  const E = [];
  for (const [b, l] of hitP) if (!Dset.has(b) && l.length >= minSupport && bitLift(b, hitP, nPos, hitN, nNeg) > kappa) E.push([b, gainSupport([b], posCounter, negCounter, nPos, nNeg, 1).gain]);
  return E.sort((a, b) => b[1] - a[1]).map((e) => e[0]);
};

/**
 * SEED_SPREAD (§6b) — K enriched seed bits chosen to land in DIFFERENT senses:
 * seed 1 = best single-bit gain; each next seed = best-gain bit whose carrier
 * contexts overlap least with the already-covered positives. Co-occurrence is
 * touched ONLY here, and only to separate seeds (§11).
 */
const seedSpread = (A, AbarH, K, { kappa, minSupport }) => {
  const nPos = A.length, nNeg = AbarH.length;
  const hitP = buildHitLists(A), hitN = buildHitLists(AbarH);
  const posCounter = makeCounter(hitP, nPos), negCounter = makeCounter(hitN, nNeg);
  const E = [];
  for (const [b, l] of hitP) if (l.length >= minSupport && bitLift(b, hitP, nPos, hitN, nNeg) > kappa) {
    E.push({ b, gain: gainSupport([b], posCounter, negCounter, nPos, nNeg).gain, carriers: new Set(l) });
  }
  E.sort((a, b) => b.gain - a.gain);
  if (E.length === 0) return [];

  const seeds = [E[0].b];
  const covered = new Set(E[0].carriers);
  while (seeds.length < K) {
    let best = null, bestScore = -Infinity;
    for (const e of E) {
      if (seeds.includes(e.b)) continue;
      let overlap = 0; for (const i of e.carriers) if (covered.has(i)) overlap++;
      const jac = overlap / (e.carriers.size + covered.size - overlap || 1);
      const spreadScore = e.gain * (1 - jac);                 // high gain, low overlap
      if (spreadScore > bestScore) { bestScore = spreadScore; best = e; }
    }
    if (!best) break;
    seeds.push(best.b);
    for (const i of best.carriers) covered.add(i);
  }
  return seeds;
};

/**
 * @function fitParallel
 * @param {Array<number[]>} A - positive contexts.
 * @param {Array<number[]>} AbarH - hard negatives.
 * @param {number[]} Dtau - shared drop set.
 * @param {number} K - number of parts (swept externally; read the elbow, §9.2).
 * @param {object} [opts]
 * @param {number} [opts.m=2] @param {number} [opts.kappa=2]
 * @param {number} [opts.sMin=3] @param {number} [opts.maxIters=8]
 * @returns {{Dtau:number[], gates:Array, iters:number, empties:number}}
 */
export const fitParallel = (A, AbarH, Dtau, K, opts = {}) => {
  const { m = 2, kappa = 2, sMin = 3, maxIters = 8, reseed = true } = opts;
  const seeds = seedSpread(A, AbarH, K, { kappa, minSupport: m });
  const ranked = reseed ? rankedEligibleBits(A, AbarH, { kappa, minSupport: m, Dtau }) : [];

  // INIT — build each part fully on its seed's own carriers, vs full Ā_h.
  let gates = seeds.map((sd) => {
    const { Q } = growClique(A, AbarH, { kappa, minSupport: m, seed: sd, dropSet: Dtau });
    if (Q.length === 0) return { empty: true, Q: [], pMinus: [], t: Infinity };
    const A_k0 = A.filter((y) => andCount(y, Q) >= m);
    const { pMinus } = peelNegative(Q, AbarH, A_k0.length ? A_k0 : A, { Dtau, kappa, m });
    const { t } = fitThreshold((x) => adaptedGateScore(x, { pPlusAdapted: Q, pMinus, dropSet: Dtau }), A_k0.length ? A_k0 : A, AbarH);
    return { empty: false, Q, pMinus, t };
  });

  let owner = new Int32Array(A.length).fill(-1), iters = 0;
  for (; iters < maxIters; iters++) {
    // E-step: assign each positive to its max-margin part.
    const next = new Int32Array(A.length);
    for (let i = 0; i < A.length; i++) {
      let bk = -1, best = -Infinity;
      for (let k = 0; k < gates.length; k++) { const mk = marginOf(gates[k], Dtau, A[i]); if (mk > best) { best = mk; bk = k; } }
      next[i] = bk;
    }
    let stable = true; for (let i = 0; i < next.length; i++) if (next[i] !== owner[i]) { stable = false; break; }
    owner = next;
    if (stable && iters > 0) break;

    // Jacobi residual negatives: negs fired by SOME OTHER part, using THIS round's gates.
    const firedBy = AbarH.map((y) => { const fs = []; for (let k = 0; k < gates.length; k++) if (marginOf(gates[k], Dtau, y) > 0) fs.push(k); return fs; });

    // M-step: refit each part on its members vs leave-one-out residual negs.
    const prev = gates;
    const used = new Set();
    const buildGate = (Q, A_k, AbarRes) => {
      const { pMinus } = peelNegative(Q, AbarRes.length ? AbarRes : AbarH, A_k, { Dtau, kappa, m });
      const { t } = fitThreshold((x) => adaptedGateScore(x, { pPlusAdapted: Q, pMinus, dropSet: Dtau }), A_k, AbarRes.length ? AbarRes : AbarH);
      for (const b of Q) used.add(b);
      return { empty: false, Q, pMinus, t };
    };
    // pass 1: refit parts that keep ≥ sMin members; others deferred for re-seeding
    const refit = prev.map((g, k) => {
      const A_k = []; for (let i = 0; i < A.length; i++) if (owner[i] === k) A_k.push(A[i]);
      if (A_k.length < sMin) return null;
      const AbarRes = AbarH.filter((_, j) => !firedBy[j].some((kk) => kk !== k));   // negs no OTHER part owns
      const { Q } = growClique(A_k, AbarRes.length ? AbarRes : AbarH, { kappa, minSupport: m, dropSet: Dtau });
      return Q.length ? buildGate(Q, A_k, AbarRes) : null;
    });
    // pass 2: re-seed each collapsed part from the best UNUSED eligible bit, so a
    // dominant collocation can't starve the rest to EMPTY (the K=1 collapse). §6b.
    let ptr = 0;
    gates = refit.map((g) => {
      if (g) return g;
      if (!reseed) return { empty: true, Q: [], pMinus: [], t: Infinity };
      while (ptr < ranked.length && used.has(ranked[ptr])) ptr++;
      if (ptr >= ranked.length) return { empty: true, Q: [], pMinus: [], t: Infinity };
      const seed = ranked[ptr++]; used.add(seed);
      const { Q } = growClique(A, AbarH, { kappa, minSupport: m, seed, dropSet: Dtau });
      if (Q.length === 0) return { empty: true, Q: [], pMinus: [], t: Infinity };
      const A_k0 = A.filter((y) => andCount(y, Q) >= m);
      if (A_k0.length < sMin) return { empty: true, Q: [], pMinus: [], t: Infinity };
      return buildGate(Q, A_k0, AbarH);
    });
  }

  const kept = gates.filter((g) => !g.empty);
  return { Dtau, gates: kept, iters, empties: gates.length - kept.length };
};

export default fitParallel;
