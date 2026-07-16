"use strict";

// ============================================================================
// __tests__/core/parts/normalization/invertedIndex.test.js
//
// Mirrors src/core/parts/normalization/invertedIndex.js — buildInvertedIndex
// turns a vocabulary + its precomputed fuzzy signatures into postings
// (part id → word ids) plus cached signature sizes. Uses hand-made sorted
// Uint32Array signatures so the expected postings are exact.
// ============================================================================

import { buildInvertedIndex } from "../../../../src/core/parts/normalization/invertedIndex.js";

describe("invertedIndex — buildInvertedIndex", () => {
  // word 0 → {1,2,3}, word 1 → {2,3,4}
  const words = ["alpha", "beta"];
  const signatures = [
    Uint32Array.from([1, 2, 3]),
    Uint32Array.from([2, 3, 4]),
  ];

  test("stores the vocabulary and signatures verbatim", () => {
    const index = buildInvertedIndex(words, signatures);
    expect(index.words).toBe(words);
    expect(index.signatures).toBe(signatures);
  });

  test("caches sizes as signatures[wi].length", () => {
    const index = buildInvertedIndex(words, signatures);
    expect(index.sizes).toEqual([3, 3]);
    expect(index.sizes[0]).toBe(signatures[0].length);
    expect(index.sizes[1]).toBe(signatures[1].length);
  });

  test("builds postings mapping part id → word ids that contain it", () => {
    const index = buildInvertedIndex(words, signatures);
    // id 1 only in word 0; id 4 only in word 1; ids 2 & 3 in both.
    expect(index.postings.get(1)).toEqual([0]);
    expect(index.postings.get(2)).toEqual([0, 1]);
    expect(index.postings.get(3)).toEqual([0, 1]);
    expect(index.postings.get(4)).toEqual([1]);
  });

  test("has no posting entry for an id absent from all signatures", () => {
    const index = buildInvertedIndex(words, signatures);
    expect(index.postings.has(99)).toBe(false);
    expect(index.postings.get(99)).toBeUndefined();
  });

  test("postings has exactly one entry per distinct id", () => {
    const index = buildInvertedIndex(words, signatures);
    expect(index.postings.size).toBe(4); // ids 1,2,3,4
  });

  test("word ids in a posting list appear in ascending word order", () => {
    const index = buildInvertedIndex(words, signatures);
    expect(index.postings.get(2)).toEqual([0, 1]);
  });

  test("an empty vocabulary yields empty postings and sizes", () => {
    const index = buildInvertedIndex([], []);
    expect(index.sizes).toEqual([]);
    expect(index.postings.size).toBe(0);
  });

  test("a word with an empty signature contributes no postings but a size of 0", () => {
    const index = buildInvertedIndex(
      ["empty", "one"],
      [Uint32Array.from([]), Uint32Array.from([7])]
    );
    expect(index.sizes).toEqual([0, 1]);
    expect(index.postings.get(7)).toEqual([1]);
    expect(index.postings.size).toBe(1);
  });
});
