"use strict";

// ============================================================================
// __tests__/core/parts/decomposer.test.js
//
// Mirrors src/core/parts/decomposer.js — greedy decomposition of a word into
// dictionary part ids (whole fast path, positional letter fill), the
// singleton-run scanner, and per-byte delimiter decomposition.
// ============================================================================

import { decomposeWord, decomposeDelimiter, findSingletonRuns } from "../../../src/core/parts/decomposer.js";
import { PartDictionary, augmentWithAtoms } from "../../../src/core/parts/dictionary.js";
import { Kind, kInvalidPartId } from "../../../src/core/parts/kind.js";

describe("decomposer", () => {
  test("whole fast path emits a single id", () => {
    const d = new PartDictionary();
    const id = d.add(Kind.Whole, "the");
    augmentWithAtoms(d);
    expect(decomposeWord(d, "the")).toEqual([id]);
  });

  test("length-1 fragment fill covers every uncovered position", () => {
    const d = new PartDictionary();
    augmentWithAtoms(d); // only singletons available
    const ids = decomposeWord(d, "xyz");
    expect(ids).toHaveLength(3);
    expect(ids.every((i) => i !== kInvalidPartId)).toBe(true);
    // Position is carried by the kind: Start "x", Mid "y", End "z" (bare values).
    expect(ids).toEqual([
      d.lookup(Kind.Start, "x"),
      d.lookup(Kind.Mid, "y"),
      d.lookup(Kind.End, "z"),
    ]);
  });

  test("singleton runs mark the whole word when nothing is claimed", () => {
    const d = new PartDictionary();
    augmentWithAtoms(d);
    const runs = findSingletonRuns(d, "abc");
    expect(runs).toEqual([{ start: 0, end: 3, touchesWordStart: true, touchesWordEnd: true }]);
  });

  test("delimiter decompose falls back to per-byte", () => {
    const d = new PartDictionary();
    augmentWithAtoms(d);
    expect(decomposeDelimiter(d, "-")).toEqual([d.lookup(Kind.Delimiter, "-")]);
  });
});
