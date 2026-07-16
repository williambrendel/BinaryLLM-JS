"use strict";

// __tests__/core/phase1/boost.test.js — the AdaBoost loop extracts m-of-n parts that cover the positives.
// Planted core {10,11,12}+common{0,1}; negatives share only common bits. Both solvers must recover it.

import buildNegSet from "../../../src/core/phase1/negSet.js";
import boost from "../../../src/core/phase1/boost.js";

// clique {10,11,12} co-occurs in 70% of positives (above chance); negatives carry only common + disjoint noise.
const mkPos = (i) => (i % 10 < 7 ? [0, 1, 10, 11, 12, 20 + (i % 15)] : [0, 1, 30 + (i % 3), 20 + (i % 15)]).sort((a, b) => a - b);
const A = Array.from({ length: 300 }, (_, i) => mkPos(i));
const negPool = Array.from({ length: 2000 }, (_, i) => [0, 1, 100 + (i % 40), 140 + (i % 37)].sort((a, b) => a - b));
const Mglob = new Set([0, 1]);
const neg = buildNegSet(A, negPool, Mglob, { maskNodes: new Set() });
// rho=20 (not the deployed 80): ρ scales with the affinity magnitude, and this ~30-bit toy graph has far
// smaller M than the real ~5000-bit corpus affinity, so a smaller regularizer is the equivalent operating point.

describe("boost — AdaBoost part extraction", () => {
  test("recovers parts covering the planted class (exp + recallTau)", () => {
    const r = boost(A, neg, { rho: 20, solver: "exp", recallTau: true, tauFloor: 0.6, suppPatience: 3 });
    expect(r.G.length).toBeGreaterThan(0);
    expect(r.unionRecall).toBeGreaterThan(0.5);
    for (const g of r.G) { expect(g.m).toBeGreaterThanOrEqual(1); expect(g.Qbits.length).toBeGreaterThan(0); expect(g.alpha).toBeGreaterThan(0); }
  });

  test("the extracted parts reference the planted core bits", () => {
    const r = boost(A, neg, { rho: 20, solver: "exp", recallTau: true, tauFloor: 0.6 });
    const allBits = new Set(r.G.flatMap((g) => g.Qbits));
    expect([10, 11, 12].some((b) => allBits.has(b))).toBe(true);
  });

  test("dc solver also produces valid parts", () => {
    const r = boost(A, neg, { rho: 20, solver: "dc", recallTau: true, tauFloor: 0.6, suppPatience: 3 });
    expect(r.G.length).toBeGreaterThan(0);
    expect(r.unionRecall).toBeGreaterThan(0.5);
  });
});
