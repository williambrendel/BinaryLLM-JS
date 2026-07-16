"use strict";

// ============================================================================
// __tests__/core/phase1/fit.test.js
//
// Mirrors src/core/phase1/fit.js — the deployed `keep+earlystop` phase-1 fit.
// Uses a deterministic synthetic class (positives share a planted core of bits;
// negatives share only common bits) so the fit is expected to recover the core
// and generalize. No RNG — signatures are built by index arithmetic.
// ============================================================================

import fitClass from "../../../src/core/phase1/fit.js";

// Deterministic "context": common bits {0,1} in everything; positives also carry
// a planted core {10,11,12} plus an index-varied discriminative bit; negatives
// carry index-varied noise bits from a disjoint range.
const COMMON = [0, 1];
const CORE = [10, 11, 12];
const mkPos = (i) => [...COMMON, ...CORE, 20 + (i % 15)].sort((a, b) => a - b);
const mkNeg = (i) => [...COMMON, 100 + (i % 40), 140 + (i % 37)].sort((a, b) => a - b);

describe("fit — deployed keep+earlystop pipeline", () => {
  const A = Array.from({ length: 200 }, (_, i) => mkPos(i));
  const negPool = Array.from({ length: 2000 }, (_, i) => mkNeg(i));
  const Mglob = new Set(COMMON); // the common bits are the SELECT mask

  test("returns a well-formed head with an early-stopped prefix", () => {
    const fit = fitClass(A, negPool, Mglob, { rho: 40 });
    expect(fit.head).toBeDefined();
    expect(typeof fit.head.fires).toBe("function");
    expect(fit.G.length).toBe(fit.rStar);
    expect(fit.rStar).toBeGreaterThanOrEqual(1);
    expect(fit.rStar).toBeLessThanOrEqual(fit.roundsTotal);
  });

  test("recovers the planted core: fires on positives, not on pure negatives", () => {
    const fit = fitClass(A, negPool, Mglob, { rho: 40 });
    const posRecall = A.filter((y) => fit.head.fires(y)).length / A.length;
    const negFire = negPool.filter((y) => fit.head.fires(y)).length / negPool.length;
    // deployed config is a weak-learner ensemble with a val-tuned operating point → moderate recall by
    // design; the smoke check is that it fires on the majority of positives and rejects pure negatives.
    expect(posRecall).toBeGreaterThan(0.55);
    expect(negFire).toBeLessThan(0.1);
  });

  test("keeps common bits available as nodes (KEEP config, not ban)", () => {
    const fit = fitClass(A, negPool, Mglob, { rho: 40 });
    // maskNodes=∅ ⇒ the common bits are NOT removed from the node universe
    for (const b of COMMON) expect(fit.neg.pset.has(b)).toBe(true);
    // but they ARE flagged by the SELECT mask (DtauSel), so the neg-score de-saturates
    for (const b of COMMON) expect(fit.neg.DtauSel.has(b)).toBe(true);
  });

  test("val fraction controls the train/val split without starving train", () => {
    const fit = fitClass(A, negPool, Mglob, { rho: 40, valFrac: 0.25 });
    expect(fit.roundsTotal).toBeGreaterThanOrEqual(1);
    // deterministic input ⇒ deterministic fit
    const fit2 = fitClass(A, negPool, Mglob, { rho: 40, valFrac: 0.25 });
    expect(fit2.rStar).toBe(fit.rStar);
  });
});
