"use strict";

// ============================================================================
// __tests__/core/math/sparse/and.test.js
//
// Mirrors src/core/math/sparse/and.js — intersection of two sorted, strictly
// increasing index sets. Returns a (typed) array view of the common elements.
// ============================================================================

import and, { and as andNamed } from "../../../../src/core/math/sparse/and.js";

const arr = (x) => Array.from(x);

describe("sparse/and", () => {
  test("intersects two overlapping sets", () => {
    expect(arr(and([1, 2, 3, 5], [2, 3, 4, 5]))).toEqual([2, 3, 5]);
  });

  test("disjoint sets intersect to empty", () => {
    expect(arr(and([1, 3, 5], [2, 4, 6]))).toEqual([]);
  });

  test("identical sets intersect to themselves", () => {
    expect(arr(and([1, 2, 3], [1, 2, 3]))).toEqual([1, 2, 3]);
  });

  test("subset intersects to the subset", () => {
    expect(arr(and([2, 3], [1, 2, 3, 4]))).toEqual([2, 3]);
  });

  test("empty operand yields empty", () => {
    expect(arr(and([], [1, 2]))).toEqual([]);
    expect(arr(and([1, 2], []))).toEqual([]);
  });

  test("Uint32Array inputs produce a Uint32Array result", () => {
    const out = and(Uint32Array.from([1, 2, 3]), Uint32Array.from([2, 3, 4]));
    expect(out).toBeInstanceOf(Uint32Array);
    expect(arr(out)).toEqual([2, 3]);
  });

  test("writes into a provided output buffer", () => {
    const out = new Uint32Array(8);
    const res = and([1, 2, 3, 5], [2, 3, 4, 5], out);
    expect(res.buffer).toBe(out.buffer); // view onto the same buffer
    expect(arr(res)).toEqual([2, 3, 5]);
  });

  test("default export equals the named export", () => {
    expect(and).toBe(andNamed);
  });
});
