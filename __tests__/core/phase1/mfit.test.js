"use strict";

// __tests__/core/phase1/mfit.test.js — mfit picks the m-of-n gate maximizing precision on the confusable
// negatives subject to weighted-recall ≥ floor. Deterministic synthetic: positives all share a core {0,1,2};
// negatives share only bit 0.

import mfit from "../../../src/core/phase1/mfit.js";

const A = Array.from({ length: 100 }, (_, i) => [0, 1, 2, 10 + (i % 5)]); // all carry the core {0,1,2}
const w = new Float64Array(100).fill(1 / 100);
const Neg = Array.from({ length: 100 }, (_, i) => [0, 50 + (i % 10)]);    // share bit 0 only

describe("mfit — m-of-n gate selection", () => {
  test("recovers a valid high-recall gate on the shared core", () => {
    const r = mfit([0, 1, 2], A, w, Neg);
    expect(r.valid).toBe(true);
    expect(r.recall).toBeGreaterThanOrEqual(0.99);   // every positive has {0,1,2}
    expect(r.m).toBeGreaterThanOrEqual(1);
    expect(r.precision).toBeGreaterThan(0.5);          // negs share ≤1 of the core → precise at m≥2
  });

  test("recallAt1 is the support ceiling (recall at m=1)", () => {
    const r = mfit([0, 1, 2], A, w, Neg);
    expect(r.recallAt1).toBeGreaterThanOrEqual(0.99);
  });

  test("invalid when no m clears the recall floor", () => {
    const r = mfit([999], A, w, Neg);                  // a bit no positive carries
    expect(r.valid).toBe(false);
    expect(r.recallAt1).toBe(0);
  });

  test("empty Qbits → invalid", () => {
    expect(mfit([], A, w, Neg).valid).toBe(false);
  });
});
