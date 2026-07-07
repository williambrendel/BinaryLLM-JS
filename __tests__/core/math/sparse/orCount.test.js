"use strict";

// ============================================================================
// __tests__/core/math/sparse/orCount.test.js
//
// Mirrors src/core/math/sparse/orCount.js — |A ∪ B| without materializing the
// union.
// ============================================================================

import orCount, { orCount as orCountNamed } from "../../../../src/core/math/sparse/orCount.js";

describe("sparse/orCount", () => {
  test("counts the union with dedup", () => {
    expect(orCount([1, 3, 5], [2, 3, 6])).toBe(5);
  });

  test("disjoint sets sum their sizes", () => {
    expect(orCount([1, 3], [2, 4])).toBe(4);
  });

  test("identical sets union to their common size", () => {
    expect(orCount([1, 2], [1, 2])).toBe(2);
  });

  test("empty operand returns the other size", () => {
    expect(orCount([], [1, 2])).toBe(2);
    expect(orCount([1, 2], [])).toBe(2);
  });

  test("unequal-length tails are counted", () => {
    expect(orCount([1, 2], [3, 4, 5, 6])).toBe(6);
  });

  test("default export equals the named export", () => {
    expect(orCount).toBe(orCountNamed);
  });
});
