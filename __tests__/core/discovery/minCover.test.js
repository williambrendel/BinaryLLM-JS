"use strict";

// ============================================================================
// __tests__/core/discovery/minCover.test.js
//
// Regression for the seq high-K runaway: the minCover floor makes a part explain
// ≥ minCover positives, so a low-λ peel cascade can't spawn hundreds of tiny
// single-context gates. Asserts every returned gate meets its floor and that a
// higher floor yields no more gates.
// ============================================================================

import buildHardNegatives from "../../../src/core/discovery/hardNegatives.js";
import discoverSeq from "../../../src/core/discovery/discoverSeq.js";

const lcg = (seed) => () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 0x100000000;
const F = Array.from({ length: 10 }, (_, i) => i);
const S1 = Array.from({ length: 10 }, (_, i) => 100 + i);
const S2 = Array.from({ length: 10 }, (_, i) => 200 + i);
const NOISE_A = Array.from({ length: 20 }, (_, i) => 300 + i);
const NOISE_B = Array.from({ length: 100 }, (_, i) => 350 + i);

const buildCorpus = () => {
  const rnd = lcg(0xC0DE01);
  const pick = (arr, k) => { const c = arr.slice(); for (let i = c.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [c[i], c[j]] = [c[j], c[i]]; } return c.slice(0, k); };
  const ctx = (...g) => Uint32Array.from([...new Set(g.flat())]).sort();
  const A = [];
  for (let i = 0; i < 60; i++) A.push(ctx(pick(F, 5), pick(S1, 7), pick(NOISE_A, 2)));
  for (let i = 0; i < 60; i++) A.push(ctx(pick(F, 5), pick(S2, 7), pick(NOISE_A, 2)));
  const Abar = [];
  for (let i = 0; i < 120; i++) Abar.push(ctx(pick(F, 5), pick(S1, 3), pick(S2, 3), pick(NOISE_A, 2)));
  for (let i = 0; i < 200; i++) Abar.push(ctx(pick(F, 5), pick(NOISE_B, 5)));
  return { A, Abar };
};

describe("discovery/minCover (seq runaway floor)", () => {
  const { A, Abar } = buildCorpus();
  const { Dtau, AbarH } = buildHardNegatives(A, Abar, F);
  const base = { m: 2, kappa: 1.5, sMin: 3, lambda: 0.001, maxK: 400 };

  test("every gate covers ≥ minCover positives", () => {
    for (const mc of [2, 10, 25]) {
      const { gates } = discoverSeq(A, AbarH, Dtau, { ...base, minCover: mc });
      for (const g of gates) expect(g.cover).toBeGreaterThanOrEqual(mc);
    }
  });

  test("a higher floor yields no more gates (caps the cascade)", () => {
    const k2 = discoverSeq(A, AbarH, Dtau, { ...base, minCover: 2 }).gates.length;
    const k25 = discoverSeq(A, AbarH, Dtau, { ...base, minCover: 25 }).gates.length;
    expect(k25).toBeLessThanOrEqual(k2);
  });
});
