"use strict";

// ============================================================================
// __tests__/core/math/sparse/or.test.js
//
// Mirrors src/core/math/sparse/or.js — union (sorted merge with dedup) of two
// sorted, strictly increasing index sets.
// ============================================================================

import or, { or as orNamed } from "../../../../src/core/math/sparse/or.js";

const arr = (x) => Array.from(x);

describe("sparse/or", () => {
  test("merges two overlapping sets with dedup", () => {
    expect(arr(or([1, 3, 5], [2, 3, 6]))).toEqual([1, 2, 3, 5, 6]);
  });

  test("disjoint sets interleave in order", () => {
    expect(arr(or([1, 3], [2, 4]))).toEqual([1, 2, 3, 4]);
  });

  test("identical sets union to themselves", () => {
    expect(arr(or([1, 2], [1, 2]))).toEqual([1, 2]);
  });

  test("empty operand returns the other set", () => {
    expect(arr(or([], [1, 2]))).toEqual([1, 2]);
    expect(arr(or([1, 2], []))).toEqual([1, 2]);
  });

  test("unequal-length tails are appended", () => {
    expect(arr(or([1, 2], [3, 4, 5, 6]))).toEqual([1, 2, 3, 4, 5, 6]);
  });

  test("Uint32Array inputs produce a Uint32Array result", () => {
    const out = or(Uint32Array.from([1, 3]), Uint32Array.from([2, 3]));
    expect(out).toBeInstanceOf(Uint32Array);
    expect(arr(out)).toEqual([1, 2, 3]);
  });

  test("handles a 0 element (0 is a valid part id) — regression", () => {
    expect(arr(or([0, 5], [3, 7]))).toEqual([0, 3, 5, 7]);
    expect(arr(or([0, 1, 2], [3]))).toEqual([0, 1, 2, 3]);
    expect(arr(or([0], [0]))).toEqual([0]);
    expect(arr(or([0, 2, 4], [1, 3]))).toEqual([0, 1, 2, 3, 4]);
    expect(arr(or([3, 7], [0, 5]))).toEqual([0, 3, 5, 7]);
  });

  test("default export equals the named export", () => {
    expect(or).toBe(orNamed);
  });
});
