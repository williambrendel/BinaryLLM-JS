"use strict";

// ============================================================================
// __tests__/core/parts/encoding/bpeEncode.test.js
//
// Mirrors src/core/parts/encoding/bpeEncode.js — the structural (BPE-style)
// [L, C, R] peel. Exercises the Whole fast-path, Start+End tiling, boundary
// anchoring, the discriminative-before-generative track tiebreak, and the
// odd-parity Letter leftover.
// ============================================================================

import { Kind } from "../../../../src/core/parts/kind.js";
import { PartDictionary, augmentWithAtoms } from "../../../../src/core/parts/dictionary.js";
import { addBigramBackstop } from "../../../../src/core/parts/encoding/bigramBackstop.js";
import { bpeEncode } from "../../../../src/core/parts/encoding/bpeEncode.js";

describe("bpeEncode — whole fast-path", () => {
  test("a registered Whole yields a single whole part, L/C/R empty", () => {
    const dict = new PartDictionary();
    const wholeId = dict.add(Kind.Whole, "the");
    augmentWithAtoms(dict);

    const sig = bpeEncode(dict, "the");
    expect(sig.whole).not.toBeNull();
    expect(sig.whole.kind).toBe(Kind.Whole);
    expect(sig.whole.value).toBe("the");
    expect(sig.whole.id).toBe(wholeId);
    expect(sig.start).toBeNull();
    expect(sig.end).toBeNull();
    expect(sig.mid).toEqual([]);
    expect(sig.parts).toHaveLength(1);
    expect(sig.parts[0]).toBe(sig.whole);
  });

  test("an empty word yields an all-empty signature", () => {
    const dict = new PartDictionary();
    augmentWithAtoms(dict);
    const sig = bpeEncode(dict, "");
    expect(sig).toEqual({ whole: null, start: null, mid: [], end: null, parts: [] });
  });
});

describe("bpeEncode — Start + End tiling", () => {
  test("a word split into Start + End populates L and R, C is empty", () => {
    const dict = new PartDictionary();
    const startId = dict.add(Kind.Start, "ru"); // proper prefix (L < n)
    const endId = dict.add(Kind.End, "nning"); // proper suffix
    augmentWithAtoms(dict);

    const sig = bpeEncode(dict, "running");
    expect(sig.whole).toBeNull();
    expect(sig.start).not.toBeNull();
    expect(sig.start.kind).toBe(Kind.Start);
    expect(sig.start.value).toBe("ru");
    expect(sig.start.id).toBe(startId);
    expect(sig.end).not.toBeNull();
    expect(sig.end.kind).toBe(Kind.End);
    expect(sig.end.value).toBe("nning");
    expect(sig.end.id).toBe(endId);
    expect(sig.mid).toEqual([]);
  });
});

describe("bpeEncode — boundary anchoring", () => {
  test("L and R are non-null when Start and End parts exist", () => {
    const dict = new PartDictionary();
    dict.add(Kind.Start, "un");
    dict.add(Kind.End, "ing");
    addBigramBackstop(dict); // covers the interior
    augmentWithAtoms(dict);

    const sig = bpeEncode(dict, "unboxing");
    expect(sig.start).not.toBeNull();
    expect(sig.start.kind).toBe(Kind.Start);
    expect(sig.start.value).toBe("un");
    expect(sig.end).not.toBeNull();
    expect(sig.end.kind).toBe(Kind.End);
    expect(sig.end.value).toBe("ing");
    // parts stay in position order, L first and R last.
    expect(sig.parts[0]).toBe(sig.start);
    expect(sig.parts[sig.parts.length - 1]).toBe(sig.end);
  });
});

describe("bpeEncode — track tiebreak (discriminative before generative)", () => {
  // Two overlapping, equal-length interior parts compete for the same span of
  // "abcdef": Mid "bcd" (pos 1) and Mid "cde" (pos 2). They cannot both be
  // chosen (they overlap on "cd"). With length and position order otherwise
  // deciding it, the track tag is the discriminator: the 'd'-tagged part wins
  // over the 'g'-tagged one. Flipping the tags flips the winner — proving the
  // tag, not the value, drives the tiebreak.
  const build = () => {
    const dict = new PartDictionary();
    dict.add(Kind.Mid, "bcd");
    dict.add(Kind.Mid, "cde");
    augmentWithAtoms(dict);
    return dict;
  };

  test("the 'd'-tagged Mid wins over the 'g'-tagged one of equal length", () => {
    const dict = build();
    const tracks = { [Kind.Mid + "|bcd"]: "d", [Kind.Mid + "|cde"]: "g" };
    const sig = bpeEncode(dict, "abcdef", tracks);
    const values = sig.mid.map((p) => p.value);
    expect(values).toContain("bcd");
    expect(values).not.toContain("cde");
  });

  test("flipping the tags flips the winner", () => {
    const dict = build();
    const tracks = { [Kind.Mid + "|bcd"]: "g", [Kind.Mid + "|cde"]: "d" };
    const sig = bpeEncode(dict, "abcdef", tracks);
    const values = sig.mid.map((p) => p.value);
    expect(values).toContain("cde");
    expect(values).not.toContain("bcd");
  });
});

describe("bpeEncode — odd-parity leftover", () => {
  test("an odd leftover position falls to a length-1 Start/Mid/End fragment", () => {
    const dict = new PartDictionary();
    // "cat" has n=3; no Start/End/Mid registered except bigrams, so the interior
    // tiles a length-2 bigram and one character is left over as a length-1 atom.
    addBigramBackstop(dict);
    augmentWithAtoms(dict);

    const sig = bpeEncode(dict, "cat");
    const fills = sig.parts.filter((p) => p.L === 1);
    expect(fills.length).toBeGreaterThanOrEqual(1);
    for (const l of fills) {
      expect([Kind.Start, Kind.Mid, Kind.End]).toContain(l.kind);
      expect(l.value.length).toBe(1); // bare char, no "##"
      expect(l.id).not.toBe(0xffffffff); // fragment atom resolves
    }
    // every position is covered exactly once
    const covered = sig.parts.reduce((sum, p) => sum + p.L, 0);
    expect(covered).toBe("cat".length);
  });
});
