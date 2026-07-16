"use strict";

// __tests__/core/phase1/replicator.test.js — replicate() maximizes π=u+Mx−ρx over the simplex; support(x*)
// is the dominant part. Planted-clique graph: nodes {0,1,2} share strong +edges, {3,4} are isolated.

import replicate from "../../../src/core/phase1/replicator.js";

const u = new Float64Array([1, 1, 1, 0, 0]);
const edges = { i: Int32Array.from([0, 0, 1]), j: Int32Array.from([1, 2, 2]), m: Float64Array.from([5, 5, 5]) };

describe("replicate — dominant-set extraction", () => {
  test("exp recovers the planted clique {0,1,2}", () => {
    const r = replicate(u, edges, { solver: "exp", rho: 1, maxIter: 500 });
    expect(r.Q).toEqual(expect.arrayContaining([0, 1, 2]));
    expect(r.Q).not.toContain(3);
    expect(r.Q).not.toContain(4);
  });

  test("x* lies on the simplex (Σ=1, all ≥0)", () => {
    const r = replicate(u, edges, { solver: "exp", rho: 1 });
    let s = 0, minv = Infinity; for (const v of r.x) { s += v; if (v < minv) minv = v; }
    expect(s).toBeCloseTo(1, 5);
    expect(minv).toBeGreaterThanOrEqual(0);
  });

  test("dc recovers the same clique as exp", () => {
    const re = replicate(u, edges, { solver: "exp", rho: 1 });
    const rd = replicate(u, edges, { solver: "dc", rho: 1 });
    expect(new Set(rd.Q)).toEqual(new Set(re.Q));
  });

  test("suppPatience early-stop keeps the same support, no more iters", () => {
    const full = replicate(u, edges, { solver: "exp", rho: 1 });
    const early = replicate(u, edges, { solver: "exp", rho: 1, suppPatience: 3 });
    expect(new Set(early.Q)).toEqual(new Set(full.Q));
    expect(early.iters).toBeLessThanOrEqual(full.iters);
  });
});
