"use strict";

// ============================================================================
// __tests__/core/discovery/acceptance.test.js
//
// Runs every §6 method (6a/6b/6d + the 6c baselines) through the §9 acceptance
// harness on the planted 2-sense corpus and asserts the spec's ordering:
//   • 6b (coherent) achieves ~0% held-FP at full recall — beats random-chunk
//     grouping (6c-ii) and global SA (6d) decisively (the discriminative-clique
//     claim: coherence, not just the gate machinery, buys the FP reduction);
//   • per-example memorizes (K=|A|, overlap ≫ 1);
//   • containment holds for every real method; CSV serializes.
// ============================================================================

import buildHardNegatives from "../../../src/core/discovery/hardNegatives.js";
import discoverSeq from "../../../src/core/discovery/discoverSeq.js";
import fitParallel from "../../../src/core/discovery/fitParallel.js";
import fitSA from "../../../src/core/discovery/fitSA.js";
import { baselinePerExample, baselineRandomChunk } from "../../../src/core/discovery/baselines.js";
import { acceptanceRow, toCSV, halve } from "../../../src/core/discovery/acceptance.js";

const lcg = (seed) => () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 0x100000000;
const F = Array.from({ length: 10 }, (_, i) => i);
const S1 = Array.from({ length: 10 }, (_, i) => 100 + i);
const S2 = Array.from({ length: 10 }, (_, i) => 200 + i);
const NOISE_A = Array.from({ length: 20 }, (_, i) => 300 + i);
const NOISE_B = Array.from({ length: 100 }, (_, i) => 350 + i);

const buildCorpus = () => {
  const rnd = lcg(0xBEEF01);
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

describe("discovery/acceptance (all methods)", () => {
  const { A, Abar } = buildCorpus();
  const [s1tr, s1he] = halve(A.slice(0, 60)), [s2tr, s2he] = halve(A.slice(60));
  const Atrain = [...s1tr, ...s2tr], Aheld = [...s1he, ...s2he];
  const { Dtau, AbarH } = buildHardNegatives(Atrain, Abar, F);
  const cfg = { K: 2, m: 2, kappa: 1.5, representation: "synthetic" };
  const O = { m: 2, kappa: 1.5, sMin: 3 };

  const parallel = { ...fitParallel(Atrain, AbarH, Dtau, 2, O), algo: "parallel" };
  const seq = { ...discoverSeq(Atrain, AbarH, Dtau, { ...O, lambda: 0.01 }), algo: "seq" };
  const sa = fitSA(Atrain, AbarH, Dtau, 2, { ...O, iters: 1500 });
  const perEx = baselinePerExample(Atrain, AbarH, Dtau, O);
  const rc2 = baselineRandomChunk(Atrain, AbarH, Dtau, 2, { ...O, c: 2, seed: 7, shareBudget: 4 });

  const row = (mo) => acceptanceRow(mo, Atrain, Aheld, Abar, cfg, AbarH);
  const rPar = row(parallel), rSa = row(sa), rPer = row(perEx), rRc2 = row(rc2);

  test("6b coherent grouping: ~0% held-FP at full recall, clean tiling", () => {
    expect(rPar.held_recall).toBeGreaterThanOrEqual(0.9);
    expect(rPar.held_FP).toBeLessThan(0.05);
    expect(rPar.containment_ok).toBe(true);
    expect(rPar.neg_growth_ok).toBe(true);
    expect(rPar.overlap).toBeLessThan(1.5);            // parts tile, don't share vocabulary
  });

  test("6b beats random-chunk grouping (6c-ii) on held-FP — coherence earns it", () => {
    expect(rPar.held_FP).toBeLessThan(rRc2.held_FP - 0.1);
  });

  test("6b beats global SA-over-gain (6d) here — greedy synergy is not suboptimal", () => {
    expect(rPar.held_FP).toBeLessThan(rSa.held_FP);
  });

  test("per-example memorizes: K=|A|, heavy vocabulary overlap", () => {
    expect(rPer.K).toBe(Atrain.length);
    expect(rPer.overlap).toBeGreaterThan(5);           // 13× redundancy vs 6b's ~1×
  });

  test("containment holds for every real discovery method", () => {
    for (const r of [rPar, row(seq), rSa]) expect(r.containment_ok).toBe(true);
  });

  test("§9.5 CSV serializes with the expected schema", () => {
    const csv = toCSV([rPar, rRc2]);
    const [header, ...lines] = csv.split("\n");
    expect(header).toContain("algo,K,m,kappa");
    expect(header).toContain("held_FP");
    expect(header).toContain("containment_ok");
    expect(lines.length).toBe(2);
  });
});
