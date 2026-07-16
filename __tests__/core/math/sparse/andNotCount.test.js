"use strict";

// ============================================================================
// __tests__/core/math/sparse/andNotCount.test.js
//
// Mirrors src/core/math/sparse/andNotCount.js — |A ∖ B| without materializing
// the difference.
// ============================================================================

import andNotCount, { andNotCount as andNotCountNamed } from "../../../../src/core/math/sparse/andNotCount.js";

describe("sparse/andNotCount", () => {
  test("counts values of a absent from b", () => {
    expect(andNotCount([1, 2, 3, 5], [2, 3, 4])).toBe(2);
  });

  test("disjoint b counts all of a", () => {
    expect(andNotCount([1, 3, 5], [2, 4, 6])).toBe(3);
  });

  test("a ⊆ b counts 0", () => {
    expect(andNotCount([1, 2, 3], [1, 2, 3, 4])).toBe(0);
  });

  test("identical sets count 0", () => {
    expect(andNotCount([1, 2, 3], [1, 2, 3])).toBe(0);
  });

  test("b tail beyond a is ignored", () => {
    expect(andNotCount([1, 2], [2, 3, 4, 5])).toBe(1);
  });

  test("empty a counts 0; empty b counts all of a", () => {
    expect(andNotCount([], [1, 2])).toBe(0);
    expect(andNotCount([1, 2], [])).toBe(2);
  });

  test("value 0 (falsy part id) is counted when absent from b", () => {
    expect(andNotCount([0, 1, 2], [1])).toBe(2);
    expect(andNotCount([0, 1, 2], [0])).toBe(2);
  });

  test("agrees with the materialized andNot length", () => {
    const a = [0, 2, 4, 6, 8, 10], b = [2, 3, 6, 7, 10]; // a ∖ b = {0, 4, 8}
    expect(andNotCount(a, b)).toBe(3);
  });

  test("default export equals the named export", () => {
    expect(andNotCount).toBe(andNotCountNamed);
  });
});
