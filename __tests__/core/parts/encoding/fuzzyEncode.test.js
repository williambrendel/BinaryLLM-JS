"use strict";

// ============================================================================
// __tests__/core/parts/encoding/fuzzyEncode.test.js
//
// Mirrors src/core/parts/encoding/fuzzyEncode.js — the fuzzy (containment)
// encoder. Verifies the signature contains every registered prefix / suffix /
// interior substring / whole / positional letter the word covers, that the
// output is a sorted, de-duplicated Uint32Array, and that a Whole includes its
// own id.
// ============================================================================

import { Kind } from "../../../../src/core/parts/kind.js";
import { PartDictionary, augmentWithAtoms } from "../../../../src/core/parts/dictionary.js";
import { addBigramBackstop } from "../../../../src/core/parts/encoding/bigramBackstop.js";
import { fuzzyEncode } from "../../../../src/core/parts/encoding/fuzzyEncode.js";

describe("fuzzyEncode — containment signature", () => {
  test("contains registered prefixes, suffixes, interiors and whole", () => {
    const dict = new PartDictionary();
    const runId = dict.add(Kind.Start, "run"); // prefix of "running"
    const ingId = dict.add(Kind.End, "ing"); // suffix of "running"
    const nnId = dict.add(Kind.Mid, "nn"); // interior substring
    const wholeId = dict.add(Kind.Whole, "running"); // whole word
    augmentWithAtoms(dict);

    const F = fuzzyEncode(dict, "running");

    expect(F).toBeInstanceOf(Uint32Array);
    const ids = Array.from(F);
    expect(ids).toContain(runId);
    expect(ids).toContain(ingId);
    expect(ids).toContain(nnId);
    expect(ids).toContain(wholeId);
  });

  test("a whole word includes its whole id", () => {
    const dict = new PartDictionary();
    const wholeId = dict.add(Kind.Whole, "the");
    augmentWithAtoms(dict);

    const ids = Array.from(fuzzyEncode(dict, "the"));
    expect(ids).toContain(wholeId);
  });

  test("includes length-1 fragment ids for each character", () => {
    const dict = new PartDictionary();
    augmentWithAtoms(dict); // seeds the length-1 Start/Mid/End atoms

    const word = "cat";
    const ids = Array.from(fuzzyEncode(dict, word));
    // Position-partitioned: each char fires exactly ONE kind — "c" is the prefix
    // (Start), "a" is strictly interior (Mid), "t" is the suffix (End).
    expect(ids).toContain(dict.lookup(Kind.Start, "c"));
    expect(ids).toContain(dict.lookup(Kind.Mid, "a"));
    expect(ids).toContain(dict.lookup(Kind.End, "t"));
    // "c" is NOT also an interior Mid (it only occurs at the prefix).
    expect(ids).not.toContain(dict.lookup(Kind.Mid, "c"));
  });

  test("a single-char whole word encodes to just its Whole (no fragments)", () => {
    const dict = new PartDictionary();
    augmentWithAtoms(dict);
    const ids = Array.from(fuzzyEncode(dict, "a"));
    expect(ids).toEqual([dict.lookup(Kind.Whole, "a")]);
  });
});

describe("fuzzyEncode — result shape", () => {
  test("returns a sorted Uint32Array", () => {
    const dict = new PartDictionary();
    addBigramBackstop(dict);
    augmentWithAtoms(dict);

    const F = fuzzyEncode(dict, "running");
    expect(F).toBeInstanceOf(Uint32Array);
    const ids = Array.from(F);
    const sorted = [...ids].sort((a, b) => a - b);
    expect(ids).toEqual(sorted);
  });

  test("ids are de-duplicated (a repeated bigram appears once)", () => {
    const dict = new PartDictionary();
    // "nan" contains the Mid "an" once and the Mid "na" once; use a word with
    // a genuinely repeated interior bigram to check dedup.
    addBigramBackstop(dict);
    augmentWithAtoms(dict);

    const ids = Array.from(fuzzyEncode(dict, "banana")); // "an" and "na" repeat
    expect(new Set(ids).size).toBe(ids.length);
  });

  test("an empty word yields an empty array", () => {
    const dict = new PartDictionary();
    augmentWithAtoms(dict);
    expect(Array.from(fuzzyEncode(dict, ""))).toEqual([]);
  });
});
