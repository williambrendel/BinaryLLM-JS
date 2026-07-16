"use strict";

/**
 * @file hardNegatives.js
 * @brief §3 precondition — the coarse gate `g0`, its threshold `t0`, the hard
 * negatives `Ā_h`, and the shared global drop set `D_τ`. Built ONCE per target,
 * before discovery.
 *
 *   p⁺_glob = ⋁A                       (scaffolding only — never a gate)
 *   D_τ     = M_glob ∧ p⁺_glob         (common ∩ seen-in-A; global, vs full corpus)
 *   g0:  score_g0(y) = |p⁺_glob ∧ M_τ ∧ y| / (|y| − |D_τ ∧ y|)   (positive-only)
 *   Ā_h = { j ∈ Ā : score_g0(y_j) > t0 }                         (confusable look-alikes)
 *
 * `Ā_h` is the negative pool for everything downstream (§3); `D_τ` stays global.
 */

import { adaptedGateScore } from "../gate/adaptedGate.js";
import fitThreshold from "./thresholdFit.js";

const unionBits = (contexts) => {
  const s = new Set();
  for (const y of contexts) for (const b of y) s.add(b);
  return Uint32Array.from(s).sort();
};
const intersectSorted = (a, bset) => a.filter((b) => bset.has(b));

/**
 * @function buildHardNegatives
 * @param {Array<number[]>} A - positive contexts.
 * @param {Array<number[]>} Abar - negative pool (full).
 * @param {Iterable<number>} Mglob - bit ids composing common words.
 * @param {object} [opts]
 * @param {number} [opts.minSize=0] - floor on |Ā_h|: if the gain-optimal `t0`
 *   yields fewer, take the top-`minSize` negatives by `score_g0` instead. The
 *   coarse gate can be so selective on real corpora (e.g. 72 of 72k) that the
 *   hard pool starves discovery; this guarantees enough confusable negatives to
 *   fit K parts against.
 * @param {number} [opts.size=0] - EXACT |Ā_h|: take exactly the top-`size`
 *   negatives by `score_g0`, ignoring `t0` and `minSize`. Use to hold the
 *   hard-negative pool CONSTANT across a sweep (e.g. §9.2 Heaps): the
 *   threshold-based pool grows with |A| and confounds K.
 * @returns {{Dtau:number[], pPlusGlobAdapted:number[], t0:number, scoreG0:Function,
 *   AbarH:Array<number[]>, AbarHidx:number[]}}
 */
export const buildHardNegatives = (A, Abar, Mglob, opts = {}) => {
  const { minSize = 0, size = 0 } = opts;
  const Mset = Mglob instanceof Set ? Mglob : new Set(Mglob);
  const pPlusGlob = [...unionBits(A)];
  const Dtau = intersectSorted(pPlusGlob, Mset);                 // common ∩ seen-in-A
  const Dset = new Set(Dtau);
  const pPlusGlobAdapted = pPlusGlob.filter((b) => !Dset.has(b)); // p⁺_glob ∧ M_τ

  const scoreG0 = (y) => adaptedGateScore(y, { pPlusAdapted: pPlusGlobAdapted, pMinus: [], dropSet: Dtau });
  const { t: t0 } = fitThreshold(scoreG0, A, Abar);

  let AbarH = [], AbarHidx = [];
  if (size > 0) {                                                // exact top-size (constant pool)
    const ranked = Abar.map((y, j) => [j, scoreG0(y)]).sort((a, b) => b[1] - a[1]).slice(0, size);
    AbarHidx = ranked.map((e) => e[0]); AbarH = AbarHidx.map((j) => Abar[j]);
    return { Dtau, pPlusGlobAdapted, t0, scoreG0, AbarH, AbarHidx };
  }
  for (let j = 0; j < Abar.length; j++) if (scoreG0(Abar[j]) > t0) { AbarH.push(Abar[j]); AbarHidx.push(j); }
  if (AbarH.length < minSize) {                                  // floor: top-minSize by score
    const ranked = Abar.map((y, j) => [j, scoreG0(y)]).sort((a, b) => b[1] - a[1]).slice(0, minSize);
    AbarHidx = ranked.map((e) => e[0]);
    AbarH = AbarHidx.map((j) => Abar[j]);
  }

  return { Dtau, pPlusGlobAdapted, t0, scoreG0, AbarH, AbarHidx };
};

export default buildHardNegatives;
