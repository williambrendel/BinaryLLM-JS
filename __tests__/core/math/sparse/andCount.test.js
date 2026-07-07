"use strict";

// ============================================================================
// __tests__/core/math/sparse/andCount.test.js
//
// Mirrors src/core/math/sparse/andCount.js — |A ∩ B| without materializing
// the intersection.
// ============================================================================

import andCount, { andCount as andCountNamed } from "../../../../src/core/math/sparse/andCount.js";

describe("sparse/andCount", () => {
  test("counts overlapping elements", () => {
    expect(andCount([1, 2, 3, 5], [2, 3, 4, 5])).toBe(3);
  });

  test("disjoint sets have no overlap", () => {
    expect(andCount([1, 3, 5], [2, 4, 6])).toBe(0);
  });

  test("identical sets overlap fully", () => {
    expect(andCount([1, 2, 3], [1, 2, 3])).toBe(3);
  });

  test("subset overlap equals the subset size", () => {
    expect(andCount([2, 3], [1, 2, 3, 4])).toBe(2);
  });

  test("empty operand yields 0", () => {
    expect(andCount([], [1, 2])).toBe(0);
    expect(andCount([1, 2], [])).toBe(0);
  });

  test("default export equals the named export", () => {
    expect(andCount).toBe(andCountNamed);
  });
});
