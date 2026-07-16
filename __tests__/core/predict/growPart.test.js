"use strict";

// ============================================================================
// __tests__/core/predict/growPart.test.js
//
// growPart builds a coverage-fired part: it accretes bits by the per-class
// binary-entropy gain and fires on popcount(x∩p) ≥ k*. It must reach a
// CONJUNCTION a single bit cannot capture (label = A ∧ B), selecting the signal
// bits over rare fillers, and it must yield nothing on a label-pure node.
// ============================================================================

import { growPart } from "../../../src/core/predict/index.js";

const featArr = [], featSet = [], labels = [];
const add = (bits, lab) => { featArr.push(bits); featSet.push(new Set(bits)); labels.push(lab); };
let f = 1000; // unique filler bits keep |x| = 4 without adding any repeated signal
for (let i = 0; i < 100; i++) add([5, 8, f++, f++], 1); // both A,B → 1
for (let i = 0; i < 100; i++) add([5, f++, f++, f++], 0); // A only → 0
for (let i = 0; i < 100; i++) add([8, f++, f++, f++], 0); // B only → 0
for (let i = 0; i < 100; i++) add([f++, f++, f++, f++], 0); // neither → 0
const idx = [...Array(400).keys()];

describe("growPart", () => {
  test("grows the AND-conjunction {5,8} that no single bit captures", () => {
    const p = growPart(idx, featArr, featSet, labels, { D: 6, sMin: 1 });
    expect(p).not.toBeNull();
    expect(new Set(p.bits)).toEqual(new Set([5, 8])); // both signal bits, no fillers
    expect(p.fire.length).toBe(100); // exactly the both-samples
    for (const j of p.fire) expect(labels[j]).toBe(1); // fire child is label-pure
    expect(p.gain).toBeGreaterThan(0);
  });

  test("fires by coverage ≥ k* — {5,8} needs BOTH bits (strict AND)", () => {
    const p = growPart(idx, featArr, featSet, labels, { D: 6, sMin: 1 });
    expect(p.k).toBe(p.bits.length); // k* = |p| here: contains all of {5,8}
  });

  test("a label-pure node yields no part", () => {
    const pureIdx = idx.slice(0, 100); // all label 1
    expect(growPart(pureIdx, featArr, featSet, labels, { sMin: 1 })).toBeNull();
  });
});
