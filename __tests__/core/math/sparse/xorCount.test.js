"use strict";

// ============================================================================
// __tests__/core/math/sparse/xorCount.test.js
//
// Mirrors src/core/math/sparse/xorCount.js — |A △ B| = |A| + |B| - 2|A ∩ B|,
// the size of the symmetric difference without materializing it.
// ============================================================================

import xorCount, { xorCount as xorCountNamed } from "../../../../src/core/math/sparse/xorCount.js";

describe("sparse/xorCount", () => {
  test("counts elements present in exactly one set", () => {
    expect(xorCount([1, 3, 5], [2, 3, 6])).toBe(4);
  });

  test("disjoint sets sum their sizes", () => {
    expect(xorCount([1, 3], [2, 4])).toBe(4);
  });

  test("identical sets cancel to 0", () => {
    expect(xorCount([1, 2], [1, 2])).toBe(0);
  });

  test("subset counts only the non-shared remainder", () => {
    expect(xorCount([2, 3], [1, 2, 3, 4])).toBe(2);
  });

  test("empty operand returns the other size", () => {
    expect(xorCount([], [1, 2])).toBe(2);
    expect(xorCount([1, 2], [])).toBe(2);
  });

  test("default export equals the named export", () => {
    expect(xorCount).toBe(xorCountNamed);
  });
});
