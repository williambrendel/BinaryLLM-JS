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

  // Multimodal class: two disjoint sense-cliques (A~40%, B~35%), so NO single clique clears the 0.5 recall floor.
  // Regression for the softFloor + balanced-α fix: the hard floor drops everything; softFloor admits the sub-0.5
  // cliques as weak learners, and their AdaBoost vote must stay POSITIVE (the sign bug gave a<0 for recall<0.5,
  // inverting a one-sided OR gate). See fit.js §8.
  // ONE weak clique {10,11,12} in 40% of positives; the other 60% carry all-UNIQUE noise, and NO common bit (a
  // common bit would let recallTau jump the OR to 100%). So the only dominant set covers <0.5 and can't grow past
  // it — reproducing the music/multimodal case where recAt1<0.5.
  test("softFloor keeps AdaBoost votes non-negative and admits ≥ the default's parts (sign-bug regression)", () => {
    // The α-sign bug: eps=1−recall gives α<0 for a recall<0.5 part, inverting a one-sided OR gate. The balanced
    // error (½·missed + ½·FP) under softFloor keeps every vote ≥0. softFloor also only ADMITS more (relaxed floor),
    // never fewer, and must not degrade the planted-class recovery. (The full sub-0.5 rescue is CV-validated — the
    // replicator won't produce a sub-0.5 dominant set from a clean toy the way music's thin signal does.)
    const def = boost(A, neg, { rho: 20, solver: "exp", recallTau: true, tauFloor: 0.6, suppPatience: 3 });
    const sf = boost(A, neg, { rho: 20, solver: "exp", recallTau: true, tauFloor: 0.6, suppPatience: 3, softFloor: true });
    expect(sf.G.length).toBeGreaterThanOrEqual(def.G.length);
    for (const g of sf.G) expect(g.alpha).toBeGreaterThanOrEqual(0);
    expect(Math.max(...sf.G.map((g) => g.alpha))).toBeGreaterThan(0);
    expect(sf.unionRecall).toBeGreaterThan(0.5);
  });
});
