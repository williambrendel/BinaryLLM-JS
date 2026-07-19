"use strict";

// __tests__/core/phase1/greedy.test.js — the greedy fast path. Deterministic synthetic: every positive carries a
// discriminative core {0,1,2,3} plus one shared-noise bit; negatives carry {50,51} plus the SAME noise (so noise is
// non-discriminative, the core is). Greedy must pick the core, skip the noise, and classify held-out data.

import { bitRates, buildPartGreedy, buildPartMarginal, setThreshold, gateFires, fitClassGreedy } from "../../../src/core/phase1/greedy.js";

const N = 800;
const asc = (a) => [...a].sort((x, y) => x - y);
const A = Array.from({ length: N }, (_, i) => asc([0, 1, 2, 3, 100 + (i % 50)]));          // core {0,1,2,3} + noise
const negPool = Array.from({ length: N }, (_, i) => asc([50, 51, 100 + (i % 50)]));        // {50,51} + same noise, no core
const Mglob = new Set(); for (let b = 0; b <= 200; b++) Mglob.add(b);
const heldPos = Array.from({ length: 120 }, (_, i) => asc([0, 1, 2, 3, 100 + ((i + 7) % 50)]));
const heldNeg = Array.from({ length: 120 }, (_, i) => asc([50, 51, 100 + ((i + 3) % 50)]));

describe("greedy — per-bit rates & part construction", () => {
  test("bitRates: core bit is fully covered in positives, absent in negatives", () => {
    const { r, f } = bitRates([0, 100], A, negPool);
    expect(r.get(0)).toBeCloseTo(1, 5);        // every positive carries bit 0
    expect(f.get(0)).toBe(0);                  // no negative carries bit 0
    expect(f.get(100)).toBeGreaterThan(0);     // noise bit appears in negatives too
  });

  test("buildPartGreedy: picks the discriminative core, skips the noise", () => {
    const cands = [0, 1, 2, 3, 100, 101];
    const { r, f } = bitRates(cands, A, negPool);
    const part = buildPartGreedy(cands, r, f);
    expect(part.Q.some((b) => b <= 3)).toBe(true);       // at least one core bit
    expect(part.Q).not.toContain(100);                   // no non-discriminative noise
  });

  test("buildPartMarginal: α-free forward greedy stops after the core (rec=1)", () => {
    const cands = [0, 1, 2, 3, 100, 101];
    const { r, f } = bitRates(cands, A, negPool);
    const Q = buildPartMarginal(cands, r, f);
    expect(Q.length).toBeGreaterThanOrEqual(1);
    expect(Q.some((b) => b <= 3)).toBe(true);
    expect(Q).not.toContain(100);
  });
});

describe("greedy — density gate & threshold", () => {
  test("gateFires: core-carrying positive fires, core-free negative does not", () => {
    const Qs = [0, 1, 2, 3];
    expect(gateFires(A[0], Qs, 0.1)).toBe(true);         // density 4/5 ≥ 0.1
    expect(gateFires(negPool[0], Qs, 0.1)).toBe(false);  // density 0
  });

  test("setThreshold returns a separating threshold in (0, 0.5]", () => {
    const t = setThreshold([0, 1, 2, 3], A.slice(0, 200), negPool.slice(0, 200));
    expect(t).toBeGreaterThan(0);
    expect(t).toBeLessThanOrEqual(0.5);
  });
});

describe("greedy — end-to-end fitClassGreedy", () => {
  test("OR-select: high held recall, low FP", () => {
    const g = fitClassGreedy(A, negPool, Mglob, { candidates: "union", selectBy: "or" });
    expect(g.parts.length).toBeGreaterThanOrEqual(1);
    expect(heldPos.filter((x) => g.fires(x)).length / heldPos.length).toBeGreaterThan(0.9);
    expect(heldNeg.filter((x) => g.fires(x)).length / heldNeg.length).toBeLessThan(0.1);
  });

  test("marginal-select: matches OR (high recall, low FP)", () => {
    const g = fitClassGreedy(A, negPool, Mglob, { candidates: "union", selectBy: "marginal" });
    expect(heldPos.filter((x) => g.fires(x)).length / heldPos.length).toBeGreaterThan(0.9);
    expect(heldNeg.filter((x) => g.fires(x)).length / heldNeg.length).toBeLessThan(0.1);
  });
});
