"use strict";

// ============================================================================
// __tests__/core/parts/generation/discriminativeTree.test.js
//
// Mirrors src/core/parts/generation/discriminativeTree.js — Track 3, the
// balanced-split discrimination tree. Verifies structural properties rather
// than exact counts (which depend on the entropy-balance criterion): parts get
// added, every added key is a real dict part with a length in [lMin, lMax], the
// routine terminates with a finite count, and it respects maxTree.
// ============================================================================

import { discriminativeTree } from "../../../../src/core/parts/generation/discriminativeTree.js";
import { PartDictionary } from "../../../../src/core/parts/dictionary.js";
import { Kind } from "../../../../src/core/parts/kind.js";

const allAlive = (words) => words.map((_, i) => i);

// A small vocab: two families that share nothing between them but a lot within.
// "preXXXX" (prefix "pre") vs "unYYYYY" (prefix "un"). A length-3 Start "pre"
// cleanly separates the first group from the rest.
const VOCAB = [
  "preface", "premise", "predict", "prepare",
  "unhappy", "unusual", "uncover", "undoing",
];
const FREQ = Float64Array.from([10, 10, 10, 10, 10, 10, 10, 10]);

describe("discriminativeTree — separates a split population", () => {
  test("adds parts and populates the addedSet", () => {
    const dict = new PartDictionary();
    const added = new Set();

    const n = discriminativeTree(VOCAB, FREQ, allAlive(VOCAB), dict, added, { lMin: 3, lMax: 7 });

    expect(Number.isFinite(n)).toBe(true);
    expect(n).toBeGreaterThan(0);
    expect(added.size).toBe(n);
  });

  test("a shared prefix produces a Start part that separates the two groups", () => {
    const dict = new PartDictionary();
    const added = new Set();

    discriminativeTree(VOCAB, FREQ, allAlive(VOCAB), dict, added, { lMin: 3, lMax: 7 });

    // Some Start part must exist that is present in exactly one of the families
    // — i.e. the tree found a discriminating boundary. "pre" (or a longer
    // superset thereof) is the canonical separator here.
    const startCount = dict.countOfKind(Kind.Start);
    const midCount = dict.countOfKind(Kind.Mid);
    const endCount = dict.countOfKind(Kind.End);
    expect(startCount + midCount + endCount).toBeGreaterThan(0);
    // The "pre" family shares the length-3 Start "pre"; it should be captured
    // as a discriminating boundary (present, and tagged in addedSet).
    expect(dict.hasStart("pre")).toBe(true);
    expect(added.has(Kind.Start + "|pre")).toBe(true);
  });

  test("every added key is a real dict part with length in [lMin, lMax]", () => {
    const dict = new PartDictionary();
    const added = new Set();
    const lMin = 3, lMax = 7;

    discriminativeTree(VOCAB, FREQ, allAlive(VOCAB), dict, added, { lMin, lMax });

    for (const key of added) {
      const kind = key.charCodeAt(0) - 48;
      const value = key.slice(key.indexOf("|") + 1);
      expect([Kind.Start, Kind.Mid, Kind.End]).toContain(kind);
      expect(dict.lookup(kind, value)).not.toBe(0xffffffff);
      expect(value.length).toBeGreaterThanOrEqual(lMin);
      expect(value.length).toBeLessThanOrEqual(lMax);
    }
  });
});

describe("discriminativeTree — termination and safety fuse", () => {
  test("terminates and returns a finite count on a trivial (single-word) input", () => {
    const words = ["solitary"];
    const freq = Float64Array.from([1]);
    const dict = new PartDictionary();
    const added = new Set();

    // Candidate lists of length < 2 are dropped, so nothing can split: 0 parts.
    const n = discriminativeTree(words, freq, allAlive(words), dict, added, { lMin: 3, lMax: 7 });

    expect(Number.isFinite(n)).toBe(true);
    expect(n).toBe(0);
    expect(added.size).toBe(0);
  });

  test("respects maxTree — never adds more parts than the fuse allows", () => {
    const dict = new PartDictionary();
    const added = new Set();
    const maxTree = 2;

    const n = discriminativeTree(VOCAB, FREQ, allAlive(VOCAB), dict, added, {
      lMin: 3,
      lMax: 7,
      maxTree,
    });

    expect(n).toBeLessThanOrEqual(maxTree);
    expect(added.size).toBe(n);
  });
});
