"use strict";

// ============================================================================
// __tests__/core/discovery/pruneClique.test.js
//
// PRUNE drops the bits the Monte-Carlo edge sampler could NOT isolate: a
// common-word (mask, d≈1) bit and a wrong-sense bit that rode into a part on
// its neighbours' coattails contribute ~0 (or hurt) the part's own gain, so
// their removal-Δ ≥ 0 and PRUNE removes them — while the true sense bits, whose
// removal hurts, are kept. Deterministic per-part cleanup.
// ============================================================================

import buildHardNegatives from "../../../src/core/discovery/hardNegatives.js";
import pruneClique from "../../../src/core/discovery/pruneClique.js";

const lcg = (seed) => () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 0x100000000;
const F = Array.from({ length: 10 }, (_, i) => i);            // function words (mask, d≈1)
const S1 = Array.from({ length: 10 }, (_, i) => 100 + i);
const S2 = Array.from({ length: 10 }, (_, i) => 200 + i);
const NOISE_A = Array.from({ length: 20 }, (_, i) => 300 + i);
const NOISE_B = Array.from({ length: 100 }, (_, i) => 350 + i);

const build = () => {
  const rnd = lcg(0x9A5C01);
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

describe("discovery/pruneClique (deterministic mask step)", () => {
  const { A, Abar } = build();
  const { AbarH } = buildHardNegatives(A, Abar, F);
  const sense1 = A.slice(0, 60);                              // the part's positive slice

  // contaminated part: 4 real S1 bits + 2 function/mask bits + 1 wrong-sense (S2) bit
  const maskBits = [F[0], F[1]], wrong = S2[0], real = [S1[0], S1[1], S1[2], S1[3]];
  const Q0 = Uint32Array.from([...real, ...maskBits, wrong]).sort();

  test("drops the mask and wrong-sense bits, keeps the sense bits", () => {
    const { Q, dropped, gain } = pruneClique([...Q0], sense1, AbarH, { minBits: 2 });
    const kept = new Set(Q), drop = new Set(dropped);
    for (const b of maskBits) expect(drop.has(b)).toBe(true);   // coattail mask bits removed
    expect(drop.has(wrong)).toBe(true);                        // wrong-sense bit removed
    for (const b of real) expect(kept.has(b)).toBe(true);      // true sense bits earn their place
    expect(gain).toBeGreaterThan(0);
  });

  test("pruning never lowers the part's gain", () => {
    // gain of the pruned part ≥ gain of the contaminated part (removal-Δ ≥ 0 each step)
    const full = pruneClique([...Q0], sense1, AbarH, { minBits: Q0.length }); // no room to prune → baseline gain
    const pruned = pruneClique([...Q0], sense1, AbarH, { minBits: 2 });
    expect(pruned.gain).toBeGreaterThanOrEqual(full.gain - 1e-9);
  });

  test("a clean clique is left unchanged", () => {
    const clean = Uint32Array.from(real).sort();
    const { dropped } = pruneClique([...clean], sense1, AbarH, { minBits: 2 });
    expect(dropped.length).toBe(0);                            // nothing to drop — every bit earns its place
  });
});
