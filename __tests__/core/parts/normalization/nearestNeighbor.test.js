"use strict";

// ============================================================================
// __tests__/core/parts/normalization/nearestNeighbor.test.js
//
// Mirrors src/core/parts/normalization/nearestNeighbor.js — Jaccard NN search
// over an inverted index. Uses hand-made sorted signatures so the shared/union
// counts (and thus the exact Jaccard values) are known by construction.
//
// Jaccard = shared / (|F| + |F_cand| - shared).
// ============================================================================

import { buildInvertedIndex } from "../../../../src/core/parts/normalization/invertedIndex.js";
import { nearestNeighbor } from "../../../../src/core/parts/normalization/nearestNeighbor.js";

describe("nearestNeighbor — Jaccard NN search", () => {
  // word 0 → {1,2,3}, word 1 → {2,3,4}, word 2 → {10,11}
  const words = ["w0", "w1", "w2"];
  const signatures = [
    Uint32Array.from([1, 2, 3]),
    Uint32Array.from([2, 3, 4]),
    Uint32Array.from([10, 11]),
  ];
  const index = buildInvertedIndex(words, signatures);

  test("querying a word's own signature returns itself at j = 1", () => {
    const res = nearestNeighbor(signatures[0], index);
    expect(res.wi).toBe(0);
    expect(res.j).toBe(1);
  });

  test("a query sharing partial ids returns the higher-overlap word", () => {
    // {1,2,3,4} overlaps w0 on {1,2,3} (3 shared) and w1 on {2,3,4} (3 shared)
    // but let's bias toward w0: query {1,2,3} shares all of w0, 2 of w1.
    const res = nearestNeighbor(Uint32Array.from([1, 2, 3]), index);
    expect(res.wi).toBe(0);
    expect(res.j).toBe(1);
  });

  test("verifies a known Jaccard value: F={1,2,3} vs {2,3,4} → 2/4 = 0.5", () => {
    // Index of only w1 so the query {1,2,3} must match it.
    const idxOnlyW1 = buildInvertedIndex(["w1"], [Uint32Array.from([2, 3, 4])]);
    const res = nearestNeighbor(Uint32Array.from([1, 2, 3]), idxOnlyW1);
    expect(res.wi).toBe(0);
    // shared {2,3} = 2, |F| = 3, |F_cand| = 3 → 2 / (3 + 3 - 2) = 0.5
    expect(res.j).toBe(0.5);
  });

  test("excludeWi skips a word, returning the next-best neighbour", () => {
    // Query w0's own signature but exclude w0 → should fall to w1 (overlap {2,3}).
    const res = nearestNeighbor(signatures[0], index, 0);
    expect(res.wi).toBe(1);
    // shared {2,3} = 2, |F| = 3, |F_cand| = 3 → 0.5
    expect(res.j).toBe(0.5);
  });

  test("a no-overlap query returns { wi: -1, j: -1 }", () => {
    const res = nearestNeighbor(Uint32Array.from([100, 200]), index);
    expect(res).toEqual({ wi: -1, j: -1 });
  });

  test("an empty query returns { wi: -1, j: -1 }", () => {
    const res = nearestNeighbor(Uint32Array.from([]), index);
    expect(res).toEqual({ wi: -1, j: -1 });
  });

  test("prefers the strictly higher Jaccard when overlaps differ", () => {
    // Query {2,3,4,10}: overlaps w1 on {2,3,4} (3), w0 on {2,3} (2), w2 on {10} (1).
    // w1: 3 / (4 + 3 - 3) = 3/4 = 0.75 — the max.
    const res = nearestNeighbor(Uint32Array.from([2, 3, 4, 10]), index);
    expect(res.wi).toBe(1);
    expect(res.j).toBe(0.75);
  });
});
