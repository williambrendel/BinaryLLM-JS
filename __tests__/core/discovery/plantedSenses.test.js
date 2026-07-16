"use strict";

// ============================================================================
// __tests__/core/discovery/plantedSenses.test.js
//
// End-to-end proof of the bit-first multi-part discovery spec on a DETERMINISTIC
// synthetic corpus with two planted senses:
//   • function words F (common, → D_τ), present everywhere
//   • sense-1 content bits S1, sense-2 content bits S2 (disjoint cliques)
//   • hard negatives = dense CROSS-SENSE MIXES (carry both S1 and S2 bits, so
//     they beat the coarse gate g0), which is the only genuinely-confusable
//     negative in this framework. The natural negative channel p⁻ for an S1
//     gate is then the OTHER sense's vocabulary S2 (§6c-ii / §9.3.1).
//
// Asserts: the coherence floor prevents blends; 6b (parallel/default) recovers
// exactly the two pure, sense-separated cliques with p⁻ = the other sense;
// 6a (sequential/fallback) finds both senses as pure cliques; every clique
// obeys the containment invariant ⋃Q ⊆ p⁺_glob ∧ M_τ; and the §7 max-pool beats
// g0 on FP at equal recall.
//
// κ = 1.5: balanced senses halve each sense bit's marginal lift over all of A
// (d ≈ (pos/mix density)·½), so the eligibility cutoff sits below 2 here.
// ============================================================================

import buildHardNegatives from "../../../src/core/discovery/hardNegatives.js";
import discoverSeq from "../../../src/core/discovery/discoverSeq.js";
import fitParallel from "../../../src/core/discovery/fitParallel.js";
import makeReadout from "../../../src/core/discovery/maxReadout.js";
import poolMetrics from "../../../src/core/discovery/metrics.js";

const lcg = (seed) => () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 0x100000000;

const F = Array.from({ length: 10 }, (_, i) => i);
const S1 = Array.from({ length: 10 }, (_, i) => 100 + i);
const S2 = Array.from({ length: 10 }, (_, i) => 200 + i);
const NOISE_A = Array.from({ length: 20 }, (_, i) => 300 + i);   // in OR(A): makes mixes g0-confusable
const NOISE_B = Array.from({ length: 100 }, (_, i) => 350 + i);  // plain-negative filler (easy)
const S1set = new Set(S1), S2set = new Set(S2);

const buildCorpus = () => {
  const rnd = lcg(0xBEEF01);
  const pick = (arr, k) => { const c = arr.slice(); for (let i = c.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [c[i], c[j]] = [c[j], c[i]]; } return c.slice(0, k); };
  const ctx = (...g) => Uint32Array.from([...new Set(g.flat())]).sort();

  const A = [];
  for (let i = 0; i < 60; i++) A.push(ctx(pick(F, 5), pick(S1, 7), pick(NOISE_A, 2)));   // sense 1
  for (let i = 0; i < 60; i++) A.push(ctx(pick(F, 5), pick(S2, 7), pick(NOISE_A, 2)));   // sense 2

  const Abar = [];
  for (let i = 0; i < 120; i++) Abar.push(ctx(pick(F, 5), pick(S1, 3), pick(S2, 3), pick(NOISE_A, 2))); // hard cross-sense mix
  for (let i = 0; i < 200; i++) Abar.push(ctx(pick(F, 5), pick(NOISE_B, 5)));                            // plain (easy)
  return { A, Abar };
};

const senseTag = (Q) => { const q = [...Q]; return q.length && q.every((b) => S1set.has(b)) ? "S1" : q.length && q.every((b) => S2set.has(b)) ? "S2" : "MIX"; };

describe("discovery/plantedSenses (end-to-end)", () => {
  const { A, Abar } = buildCorpus();
  const { Dtau, AbarH, t0, scoreG0 } = buildHardNegatives(A, Abar, F);
  const opts = { m: 2, kappa: 1.5, sMin: 3 };

  // containment RHS: p⁺_glob ∧ M_τ = OR(A) ∖ D_τ
  const orA = new Set(A.flatMap((y) => [...y]));
  const Dset = new Set(Dtau);
  const allowed = new Set([...orA].filter((b) => !Dset.has(b)));

  test("hard negatives are the confusable cross-sense mixes", () => {
    expect([...Dtau].every((b) => F.includes(b))).toBe(true);            // D_τ = function words
    const mixes = AbarH.filter((y) => [...y].some((b) => S1set.has(b)) && [...y].some((b) => S2set.has(b)));
    expect(mixes.length).toBeGreaterThan(80);                            // Ā_h dominated by mixes
    expect(AbarH.length).toBeLessThan(Abar.length);                     // Ā_h ≪ Ā (plains excluded)
  });

  test("6b parallel (K=2) recovers exactly the two pure senses", () => {
    const { gates, empties } = fitParallel(A, AbarH, Dtau, 2, opts);
    expect(gates.length).toBe(2);
    expect(empties).toBe(0);
    for (const g of gates) expect(["S1", "S2"]).toContain(senseTag(g.Q));  // each clique is coherent
    expect(new Set(gates.map((g) => senseTag(g.Q)))).toEqual(new Set(["S1", "S2"]));
  });

  test("6b p⁻ is the OTHER sense's vocabulary (coherent exclusion)", () => {
    const { gates } = fitParallel(A, AbarH, Dtau, 2, opts);
    for (const g of gates) {
      const other = senseTag(g.Q) === "S1" ? S2set : S1set;
      const hits = g.pMinus.filter((b) => other.has(b)).length;
      expect(hits).toBeGreaterThanOrEqual(3);                            // excludes the confusable sense
    }
  });

  test("6a sequential finds both senses as pure cliques (fallback)", () => {
    const { gates } = discoverSeq(A, AbarH, Dtau, { ...opts, lambda: 0.01 });
    expect(gates.length).toBeGreaterThanOrEqual(2);
    const tags = gates.map((g) => senseTag(g.Q));
    expect(tags).toContain("S1");                                        // a pure S1 clique exists
    expect(tags).toContain("S2");                                        // a pure S2 clique exists
  });

  test("containment invariant: every clique ⊆ p⁺_glob ∧ M_τ (both algorithms)", () => {
    const all = [
      ...fitParallel(A, AbarH, Dtau, 2, opts).gates,
      ...discoverSeq(A, AbarH, Dtau, { ...opts, lambda: 0.01 }).gates,
    ];
    for (const g of all) for (const b of g.Q) {
      expect(Dset.has(b)).toBe(false);                                   // never a drop-set (function) bit
      expect(allowed.has(b)).toBe(true);                                 // always a seen-in-A content bit
    }
  });

  test("§7 max-pool beats g0 on FP at equal recall", () => {
    const g0 = poolMetrics((x) => scoreG0(x) > t0, A, Abar);
    const rd = makeReadout(fitParallel(A, AbarH, Dtau, 2, opts).gates, Dtau);
    const mp = poolMetrics((x) => rd.fires(x), A, Abar);
    expect(mp.recall).toBeGreaterThanOrEqual(0.9);
    expect(mp.recall).toBeGreaterThanOrEqual(g0.recall - 0.02);          // recall conserved
    expect(mp.fpRate).toBeLessThan(g0.fpRate - 0.1);                     // FP substantially cut
  });
});
