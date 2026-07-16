"use strict";

// ============================================================================
// __tests__/core/predict/poolTree.test.js
//
// The second-pass tree uses a FIXED pool of parts as its only splitters. Given a
// pool that contains a separating part, it must learn the separable signal — beat
// unigram, clear the gate — and it must ignore useless pool parts.
// ============================================================================

import { buildPoolTree, buildUnigram, evaluate, calibrationGate } from "../../../src/core/predict/index.js";

const V = 3;
const featuresOf = (r) => r.bits;
const labelOf = (r) => r.label;
const recs = [];
for (let i = 0; i < 400; i++) { const on = i % 2 === 0; recs.push({ bits: on ? [3, 7] : [3, 9], label: on ? 1 : 2 }); }

describe("predict/poolTree", () => {
  const pool = [{ bits: [3], k: 1 }, { bits: [7], k: 1 }, { bits: [9], k: 1 }]; // {3} fires all (useless)
  const tree = buildPoolTree(recs, { featuresOf, labelOf, pool, sMin: 1, Dmax: 8 });

  test("learns the separable signal from the pool — beats unigram, clears the gate", () => {
    const t = evaluate(tree, recs, { V, labelOf });
    const u = evaluate(buildUnigram(recs, labelOf), recs, { V, labelOf });
    expect(t.acc).toBeGreaterThan(0.98);
    expect(t.ll).toBeLessThan(u.ll);
    expect(calibrationGate(t.ll, u.ll)).toBe(true);
    expect(tree.leaves).toBeGreaterThanOrEqual(2);
  });

  test("the all-firing part {3} is never chosen — it cannot split", () => {
    // root split must be {7} or {9}, never the useless {3}.
    expect([7, 9]).toContain(tree.root.part[0]);
  });

  test("routes [3,7] to the label-1 leaf", () => {
    const leaf = tree.route({ bits: [3, 7] });
    let best, bc = -1; for (const [w, c] of leaf.counts) if (c > bc) { bc = c; best = w; }
    expect(best).toBe(1);
  });
});
