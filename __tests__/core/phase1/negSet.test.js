"use strict";

// __tests__/core/phase1/negSet.test.js — buildNegSet selects the positive bit-vocabulary (P⁺) and a frozen
// pool of confusable negatives, plus their marginals/joints. Deterministic synthetic: positives carry a core
// {10,11,12}+common{0,1}; negatives carry common{0,1}+disjoint noise.

import buildNegSet from "../../../src/core/phase1/negSet.js";

const A = Array.from({ length: 100 }, (_, i) => [0, 1, 10, 11, 12, 20 + (i % 10)].sort((a, b) => a - b));
const negPool = Array.from({ length: 1000 }, (_, i) => [0, 1, 100 + (i % 30), 140 + (i % 25)].sort((a, b) => a - b));
const Mglob = new Set([0, 1]);

describe("buildNegSet", () => {
  const neg = buildNegSet(A, negPool, Mglob, { maskNodes: new Set() });

  test("P⁺ contains the planted core bits", () => {
    expect(neg.pPlus).toEqual(expect.arrayContaining([10, 11, 12]));
    expect(neg.pset.has(11)).toBe(true);
  });

  test("selects a non-empty frozen negative pool", () => {
    expect(neg.negN).toBeGreaterThan(0);
    expect(neg.Neg.length).toBe(neg.negN);
  });

  test("exposes negative marginals and joints as Maps", () => {
    expect(neg.nMinus).toBeInstanceOf(Map);
    expect(neg.jMinus).toBeInstanceOf(Map);
  });

  test("maskNodes=∅ keeps common bits as P⁺ nodes", () => {
    expect(neg.pset.has(0)).toBe(true); // common bit retained as an affinity node
  });
});
