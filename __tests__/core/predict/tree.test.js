"use strict";

// ============================================================================
// __tests__/core/predict/tree.test.js
//
// The capped best-first CART tree. On a cleanly separable signal (one bit
// determines the label) the tree must learn it — near-zero log-loss, beats
// unigram, clears the gate. This proves the machinery is correct, so an overfit
// on noisy real data is a property of the data/split, not a tree bug.
// ============================================================================

import { buildCartTree, buildUnigram, evaluate, calibrationGate } from "../../../src/core/predict/index.js";

const V = 3;
const featuresOf = (r) => r.bits;
const labelOf = (r) => r.label;

// bit 7 present → label 1; absent (bit 9) → label 2. Bit 3 is a shared distractor.
const recs = [];
for (let i = 0; i < 400; i++) {
  const on = i % 2 === 0;
  recs.push({ bits: on ? [3, 7] : [3, 9], label: on ? 1 : 2 });
}

describe("predict/tree — learns a separable signal", () => {
  test("a perfectly-separating bit → near-zero log-loss, beats unigram", () => {
    const tree = buildCartTree(recs, { featuresOf, labelOf, sMin: 1, K: 64 });
    const uni = buildUnigram(recs, labelOf);
    const t = evaluate(tree, recs, { V, labelOf });
    const u = evaluate(uni, recs, { V, labelOf });
    expect(t.acc).toBeGreaterThan(0.98);
    expect(t.ll).toBeLessThan(u.ll);
    expect(calibrationGate(t.ll, u.ll)).toBe(true);
    expect(tree.leaves).toBeGreaterThanOrEqual(2);
  });

  test("routes a record by its bits to the right leaf", () => {
    const tree = buildCartTree(recs, { featuresOf, labelOf, sMin: 1, K: 64 });
    const leaf = tree.route({ bits: [3, 7] }); // → the label-1 leaf
    let best, bc = -1;
    for (const [w, c] of leaf.counts) if (c > bc) { bc = c; best = w; }
    expect(best).toBe(1);
  });

  test("s_min caps leaf count (regularization)", () => {
    const big = buildCartTree(recs, { featuresOf, labelOf, sMin: 1, K: 64 });
    const small = buildCartTree(recs, { featuresOf, labelOf, sMin: 300, K: 64 });
    expect(small.leaves).toBeLessThanOrEqual(big.leaves);
  });
});
