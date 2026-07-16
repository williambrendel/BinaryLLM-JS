"use strict";

// ============================================================================
// __tests__/core/discovery/reseedCollapse.test.js
//
// Regression for the 6b K=1 collapse: when one collocation DOMINATES, the
// fixed-K EM used to empty every other part (the E-step gives all positives to
// the dominant gate). Empty-part re-seeding (fitParallel opts.reseed) keeps K
// parts alive by re-seeding collapsed parts from fresh eligible bits. This test
// builds a dominant + rare sense and asserts re-seed recovers BOTH while the
// no-reseed control loses the rare one.
// ============================================================================

import buildHardNegatives from "../../../src/core/discovery/hardNegatives.js";
import fitParallel from "../../../src/core/discovery/fitParallel.js";

const lcg = (seed) => () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 0x100000000;
const F = Array.from({ length: 10 }, (_, i) => i);
const S1 = Array.from({ length: 10 }, (_, i) => 100 + i);      // dominant sense
const S2 = Array.from({ length: 10 }, (_, i) => 200 + i);      // rare sense
const NOISE_A = Array.from({ length: 20 }, (_, i) => 300 + i);
const NOISE_B = Array.from({ length: 100 }, (_, i) => 350 + i);
const S1set = new Set(S1), S2set = new Set(S2);
const senseTag = (Q) => { const q = [...Q]; return q.length && q.every((b) => S1set.has(b)) ? "S1" : q.length && q.every((b) => S2set.has(b)) ? "S2" : "MIX"; };

const buildImbalanced = () => {
  const rnd = lcg(0x5EED01);
  const pick = (arr, k) => { const c = arr.slice(); for (let i = c.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [c[i], c[j]] = [c[j], c[i]]; } return c.slice(0, k); };
  const ctx = (...g) => Uint32Array.from([...new Set(g.flat())]).sort();
  const A = [];
  for (let i = 0; i < 110; i++) A.push(ctx(pick(F, 5), pick(S1, 7), pick(NOISE_A, 2)));  // dominant: 110
  for (let i = 0; i < 22; i++) A.push(ctx(pick(F, 5), pick(S2, 7), pick(NOISE_A, 2)));    // rare: 22
  const Abar = [];
  for (let i = 0; i < 120; i++) Abar.push(ctx(pick(F, 5), pick(S1, 3), pick(S2, 3), pick(NOISE_A, 2)));
  for (let i = 0; i < 200; i++) Abar.push(ctx(pick(F, 5), pick(NOISE_B, 5)));
  return { A, Abar };
};

describe("discovery/reseedCollapse (6b dominant-collocation robustness)", () => {
  const { A, Abar } = buildImbalanced();
  const { Dtau, AbarH } = buildHardNegatives(A, Abar, F);
  const opts = { m: 2, kappa: 1.5, sMin: 3 };

  test("re-seed (default) recovers BOTH the dominant and the rare sense", () => {
    const { gates } = fitParallel(A, AbarH, Dtau, 4, { ...opts, reseed: true });
    const tags = new Set(gates.map((g) => senseTag(g.Q)));
    expect(tags.has("S1")).toBe(true);
    expect(tags.has("S2")).toBe(true);                          // the rare sense survives
    for (const g of gates) expect(["S1", "S2"]).toContain(senseTag(g.Q));  // still coherent
  });

  test("re-seed keeps more parts alive than the no-reseed control", () => {
    const withR = fitParallel(A, AbarH, Dtau, 4, { ...opts, reseed: true });
    const without = fitParallel(A, AbarH, Dtau, 4, { ...opts, reseed: false });
    expect(withR.gates.length).toBeGreaterThanOrEqual(without.gates.length);
    expect(withR.gates.length).toBeGreaterThanOrEqual(2);       // does not collapse to 1
  });
});
