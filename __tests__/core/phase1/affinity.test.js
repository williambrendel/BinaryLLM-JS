"use strict";

// __tests__/core/phase1/affinity.test.js — buildAffinity produces the unary log-odds u and the PMI-difference
// edge weights M over P⁺. Planted core {10,11,12} always co-occurs in positives, never in negatives.

import buildNegSet from "../../../src/core/phase1/negSet.js";
import buildAffinity from "../../../src/core/phase1/affinity.js";

// clique {10,11,12} co-occurs in 70% of positives (above chance ⇒ positive PMI edges); the rest carry decoys.
const mkPos = (i) => (i % 10 < 7 ? [0, 1, 10, 11, 12, 20 + (i % 12)] : [0, 1, 30 + (i % 3), 20 + (i % 12)]).sort((a, b) => a - b);
const A = Array.from({ length: 200 }, (_, i) => mkPos(i));
const negPool = Array.from({ length: 1000 }, (_, i) => [0, 1, 100 + (i % 30), 140 + (i % 25)].sort((a, b) => a - b));
const Mglob = new Set([0, 1]);
const neg = buildNegSet(A, negPool, Mglob, { maskNodes: new Set() });
const w = new Float64Array(A.length).fill(1 / A.length);

describe("buildAffinity", () => {
  const { u, edges, n } = buildAffinity(A, w, neg, {});
  const pos = new Map(neg.pPlus.map((b, i) => [b, i]));

  test("u is positive (discriminative) for the planted-core bits", () => {
    for (const b of [10, 11, 12]) expect(u[pos.get(b)]).toBeGreaterThan(0);
  });

  test("edges are a symmetric COO of equal-length arrays over P⁺ positions", () => {
    expect(edges.i.length).toBe(edges.m.length);
    expect(edges.j.length).toBe(edges.m.length);
    for (let e = 0; e < edges.i.length; e++) { expect(edges.i[e]).toBeLessThan(n); expect(edges.j[e]).toBeLessThan(n); }
  });

  test("the co-occurring core pair (10,11) has a positive edge", () => {
    const a = pos.get(10), b = pos.get(11); const lo = Math.min(a, b), hi = Math.max(a, b);
    let found = 0;
    for (let e = 0; e < edges.m.length; e++) if (Math.min(edges.i[e], edges.j[e]) === lo && Math.max(edges.i[e], edges.j[e]) === hi) found = edges.m[e];
    expect(found).toBeGreaterThan(0);
  });
});
