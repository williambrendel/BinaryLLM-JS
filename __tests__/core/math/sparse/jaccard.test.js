"use strict";

// ============================================================================
// __tests__/core/math/sparse/jaccard.test.js
//
// Mirrors src/core/math/sparse/jaccard.js — Jaccard similarity
// |A ∩ B| / |A ∪ B| over two sorted, strictly increasing index sets.
// ============================================================================

import jaccard, { jaccard as jaccardNamed } from "../../../../src/core/math/sparse/jaccard.js";

describe("sparse/jaccard", () => {
  test("partial overlap", () => {
    // |∩| = 1 (the element 3), |∪| = 5 -> 0.2
    expect(jaccard([1, 3, 5], [2, 3, 6])).toBeCloseTo(0.2, 10);
  });

  test("identical sets have similarity 1", () => {
    expect(jaccard([1, 2, 3], [1, 2, 3])).toBe(1);
  });

  test("disjoint sets have similarity 0", () => {
    expect(jaccard([1, 2], [3, 4])).toBe(0);
  });

  test("subset similarity is |subset| / |superset|", () => {
    expect(jaccard([2, 3], [1, 2, 3, 4])).toBeCloseTo(0.5, 10);
  });

  test("empty vs non-empty is 0", () => {
    expect(jaccard([], [1, 2])).toBe(0);
  });

  test("empty vs empty is NaN (0/0 boundary)", () => {
    expect(Number.isNaN(jaccard([], []))).toBe(true);
  });

  test("default export equals the named export", () => {
    expect(jaccard).toBe(jaccardNamed);
  });
});
