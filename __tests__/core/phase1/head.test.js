"use strict";

// __tests__/core/phase1/head.test.js — the α-sum head: S(x)=Σ α_k·1[|Q_k∧x|≥m_k], fires when S>θ.

import makeHead from "../../../src/core/phase1/head.js";

const G = [
  { Qbits: [1, 2], m: 2, alpha: 1.0 },   // needs BOTH 1 and 2
  { Qbits: [3, 4], m: 1, alpha: 0.5 },   // needs EITHER 3 or 4
];

describe("makeHead — α-sum head", () => {
  test("S sums the α of the firing parts", () => {
    const h = makeHead(G, 0.6);
    expect(h.S([1, 2, 3])).toBeCloseTo(1.5);   // both parts fire
    expect(h.S([1, 2])).toBeCloseTo(1.0);       // only part 0
    expect(h.S([3])).toBeCloseTo(0.5);          // only part 1
    expect(h.S([1])).toBeCloseTo(0.0);          // part 0 needs m=2
  });

  test("fires when S > θ", () => {
    const h = makeHead(G, 0.6);
    expect(h.fires([1, 2])).toBe(true);          // S=1.0 > 0.6
    expect(h.fires([3])).toBe(false);            // S=0.5 < 0.6
  });

  test("default θ = ½·Σα", () => {
    expect(makeHead(G).theta).toBeCloseTo(0.75);
  });

  test("maxpool fires if ANY part fires", () => {
    const h = makeHead(G);
    expect(h.maxpoolFires([3])).toBe(true);      // part 1 alone
    expect(h.maxpoolFires([1])).toBe(false);     // no part fires
  });
});
