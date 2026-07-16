"use strict";

// ============================================================================
// __tests__/core/gate/adaptedGate.test.js
//
// Proves the simplified adapted-mask gate  score = (a − m)/(a + u)  is exactly
// the full masked definition
//     ( |p⁺ ∧ M_τ ∧ x| − |p⁻ ∧ M_τ ∧ x| ) / |M_τ ∧ x|,   M_τ = ¬(M_glob ∧ p⁺)
// by evaluating the definition with DENSE bit arrays and the gate with the
// SPARSE andCount/andNotCount primitives, over many random instances.
// ============================================================================

import adaptedGateScore, {
  adaptedGateScore as scoreNamed,
  adaptedGateFires,
} from "../../../src/core/gate/adaptedGate.js";

// deterministic LCG so the random sweep is reproducible (numerical-recipes constants)
const lcg = (seed) => () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 0x100000000;

const sortedSet = (dense) => { const s = []; for (let b = 0; b < dense.length; b++) if (dense[b]) s.push(b); return s; };

// Reference: the FULL masked score computed straight from dense bit vectors.
const denseScore = (xd, pPlusD, pMinusD, mGlobD) => {
  const DIM = xd.length;
  let numP = 0, numM = 0, den = 0;
  for (let b = 0; b < DIM; b++) {
    const mtau = !(mGlobD[b] && pPlusD[b]);   // M_τ = ¬(M_glob ∧ p⁺)
    if (!(mtau && xd[b])) continue;           // bit must survive M_τ and be in x
    den++;
    if (pPlusD[b]) numP++;
    if (pMinusD[b]) numM++;
  }
  return den > 0 ? (numP - numM) / den : 0;
};

describe("core/gate/adaptedGate", () => {
  test("worked example matches the docstring", () => {
    const gate = { pPlusAdapted: [2, 7], pMinus: [9], dropSet: [5] };
    expect(adaptedGateScore([2, 3, 9], gate)).toBe(0); // a=1, m=1, d=|{2,3,9}∖{5}|=3 → 0/3
  });

  test("empty-support context scores 0, does not divide by zero", () => {
    // x lands entirely on dropped bits (D_τ = ¬M_τ): M_τ ∧ x = ∅
    const gate = { pPlusAdapted: [], pMinus: [], dropSet: [4] };
    expect(adaptedGateScore([4], gate)).toBe(0);
    expect(adaptedGateFires([4], gate, -0.5)).toBe(false);
  });

  test("simplified score equals the full dense M_τ definition (random sweep)", () => {
    const DIM = 64, TRIALS = 400;
    const rnd = lcg(0xC0FFEE);
    for (let trial = 0; trial < TRIALS; trial++) {
      // p⁺ and p⁻ are disjoint tails; M_glob is an independent common-bit mask.
      const pPlusD = new Uint8Array(DIM), pMinusD = new Uint8Array(DIM), mGlobD = new Uint8Array(DIM), xd = new Uint8Array(DIM);
      for (let b = 0; b < DIM; b++) {
        const r = rnd();
        if (r < 0.25) pPlusD[b] = 1; else if (r < 0.45) pMinusD[b] = 1; // disjoint by construction
        if (rnd() < 0.30) mGlobD[b] = 1;
        if (rnd() < 0.35) xd[b] = 1;
      }
      // Derived stored sets (all sparse, sorted): p⁺_adapted = p⁺ ∧ M_τ.
      // p⁺_adapted = p⁺ ∧ M_τ (survivors); D_τ = p⁺ ∧ ¬M_τ = p⁺ ∧ M_glob (drops).
      const pPlusAdaptedD = new Uint8Array(DIM), dropSetD = new Uint8Array(DIM);
      for (let b = 0; b < DIM; b++) if (pPlusD[b]) (mGlobD[b] ? dropSetD : pPlusAdaptedD)[b] = 1;
      const gate = { pPlusAdapted: sortedSet(pPlusAdaptedD), pMinus: sortedSet(pMinusD), dropSet: sortedSet(dropSetD) };
      const x = sortedSet(xd);

      const expected = denseScore(xd, pPlusD, pMinusD, mGlobD);
      expect(adaptedGateScore(x, gate)).toBeCloseTo(expected, 12);
    }
  });

  test("fires() agrees with score() > t across thresholds (random sweep)", () => {
    const DIM = 48, TRIALS = 300;
    const rnd = lcg(0x1234BEEF);
    for (let trial = 0; trial < TRIALS; trial++) {
      const pPlusD = new Uint8Array(DIM), pMinusD = new Uint8Array(DIM), mGlobD = new Uint8Array(DIM), xd = new Uint8Array(DIM);
      for (let b = 0; b < DIM; b++) {
        const r = rnd();
        if (r < 0.25) pPlusD[b] = 1; else if (r < 0.45) pMinusD[b] = 1;
        if (rnd() < 0.30) mGlobD[b] = 1;
        if (rnd() < 0.35) xd[b] = 1;
      }
      // p⁺_adapted = p⁺ ∧ M_τ (survivors); D_τ = p⁺ ∧ ¬M_τ = p⁺ ∧ M_glob (drops).
      const pPlusAdaptedD = new Uint8Array(DIM), dropSetD = new Uint8Array(DIM);
      for (let b = 0; b < DIM; b++) if (pPlusD[b]) (mGlobD[b] ? dropSetD : pPlusAdaptedD)[b] = 1;
      const gate = { pPlusAdapted: sortedSet(pPlusAdaptedD), pMinus: sortedSet(pMinusD), dropSet: sortedSet(dropSetD) };
      const x = sortedSet(xd);
      const s = adaptedGateScore(x, gate);
      const support = denseSupport(xd, pPlusD, mGlobD); // |M_τ ∧ x|
      for (const t of [-0.5, -0.1, 0, 0.1, 0.5]) {
        // fires(t) ⟺ x has support AND score(x) > t. No-divide form must agree.
        expect(adaptedGateFires(x, gate, t)).toBe(support > 0 && s > t);
      }
    }
  });

  test("default export equals the named export", () => {
    expect(adaptedGateScore).toBe(scoreNamed);
  });
});

// |M_τ ∧ x| computed densely — distinguishes "genuine 0 score" from
// "empty support" in the fires() cross-check.
const denseSupport = (xd, pPlusD, mGlobD) => {
  let den = 0;
  for (let b = 0; b < xd.length; b++) if (!(mGlobD[b] && pPlusD[b]) && xd[b]) den++;
  return den;
};
