"use strict";

// ============================================================================
// __tests__/core/math/sparse/xor.test.js
//
// Mirrors src/core/math/sparse/xor.js — symmetric difference (elements in
// exactly one operand) of two sorted, strictly increasing index sets.
// ============================================================================

import xor, { xor as xorNamed } from "../../../../src/core/math/sparse/xor.js";

const arr = (x) => Array.from(x);

describe("sparse/xor", () => {
  test("keeps elements present in exactly one set", () => {
    expect(arr(xor([1, 3, 5], [2, 3, 6]))).toEqual([1, 2, 5, 6]);
  });

  test("disjoint sets keep everything", () => {
    expect(arr(xor([1, 3], [2, 4]))).toEqual([1, 2, 3, 4]);
  });

  test("identical sets cancel to empty", () => {
    expect(arr(xor([1, 2], [1, 2]))).toEqual([]);
  });

  test("subset leaves the non-shared remainder", () => {
    expect(arr(xor([2, 3], [1, 2, 3, 4]))).toEqual([1, 4]);
  });

  test("empty operand returns the other set", () => {
    expect(arr(xor([], [1, 2]))).toEqual([1, 2]);
    expect(arr(xor([1, 2], []))).toEqual([1, 2]);
  });

  test("Uint32Array inputs produce a Uint32Array result", () => {
    const out = xor(Uint32Array.from([1, 3]), Uint32Array.from([2, 3]));
    expect(out).toBeInstanceOf(Uint32Array);
    expect(arr(out)).toEqual([1, 2]);
  });

  test("handles a 0 element (0 is a valid part id) — regression", () => {
    expect(arr(xor([0, 1, 2], [0, 2, 3]))).toEqual([1, 3]);
    expect(arr(xor([0, 5], [3, 7]))).toEqual([0, 3, 5, 7]);
    expect(arr(xor([0], [0]))).toEqual([]);
    expect(arr(xor([0, 2, 4], [0, 1, 4]))).toEqual([1, 2]);
  });

  test("default export equals the named export", () => {
    expect(xor).toBe(xorNamed);
  });
});
