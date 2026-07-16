"use strict";

// ============================================================================
// __tests__/core/predict/pathDecode.test.js
//
// The path decoder aggregates the root→leaf child distributions as a
// support-weighted correction to the unigram prior (spec §4.2). On a separable
// signal it must beat unigram and clear the gate — the §5.5 discriminator that a
// leaf-only readout cannot provide.
// ============================================================================

import { buildCartTree, buildUnigram, evaluate, evaluatePath, calibrationGate } from "../../../src/core/predict/index.js";

const V = 3;
const featuresOf = (r) => r.bits;
const labelOf = (r) => r.label;
const recs = [];
for (let i = 0; i < 400; i++) { const on = i % 2 === 0; recs.push({ bits: on ? [3, 7] : [3, 9], label: on ? 1 : 2 }); }

describe("predict/pathDecode", () => {
  const tree = buildCartTree(recs, { featuresOf, labelOf, sMin: 1, K: 64 });
  const u = evaluate(buildUnigram(recs, labelOf), recs, { V, labelOf });

  test("path decoder beats unigram on a separable signal, clears the gate", () => {
    const p = evaluatePath(tree, recs, { V, labelOf });
    expect(p.ll).toBeLessThan(u.ll);
    expect(calibrationGate(p.ll, u.ll)).toBe(true);
    expect(p.acc).toBeGreaterThan(0.98);
  });

  test("routePath returns a non-empty root→leaf path with label histograms", () => {
    const path = tree.routePath(recs[0]);
    expect(path.length).toBeGreaterThanOrEqual(1);
    for (const node of path) { expect(node.total).toBeGreaterThan(0); expect(node.counts.size).toBeGreaterThanOrEqual(1); }
  });
});
