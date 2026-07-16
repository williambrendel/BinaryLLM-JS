"use strict";

// ============================================================================
// __tests__/core/parts/normalization/canonicalize.test.js
//
// Mirrors src/core/parts/normalization/canonicalize.js — the confidence-gated
// resolver. Unlike the other two normalization tests, this one uses a REAL
// dictionary: a tiny corpus is run through generateParts + addBigramBackstop,
// each vocabulary word is fuzzyEncode'd, and buildInvertedIndex assembles the
// searchable index the canonicalizer queries.
// ============================================================================

import { generateParts } from "../../../../src/core/parts/generation/index.js";
import { wordFreq } from "../../../../src/utilities/corpus/wordFreq.js";
import { fuzzyEncode, addBigramBackstop } from "../../../../src/core/parts/encoding/index.js";
import { buildInvertedIndex } from "../../../../src/core/parts/normalization/invertedIndex.js";
import { canonicalize } from "../../../../src/core/parts/normalization/canonicalize.js";

// A small corpus with a few morphologically-related words so the NN search has
// real neighbours to score against.
const CORPUS =
  "running runner running jumping jumper jumped runs jumps " +
  "running runner jumping walking walker walked walks running";

let dict;
let index;
let words;

beforeAll(() => {
  const freq = wordFreq([new TextEncoder().encode(CORPUS)]);
  const gen = generateParts(freq);
  dict = gen.dict;
  addBigramBackstop(dict);

  words = [...freq.keys()];
  const signatures = words.map((w) => fuzzyEncode(dict, w));
  index = buildInvertedIndex(words, signatures);
});

describe("canonicalize — in-vocabulary words self-match", () => {
  test("vocabulary is non-empty and includes 'running'", () => {
    expect(words.length).toBeGreaterThan(0);
    expect(words).toContain("running");
  });

  test("an in-vocab word canonicalizes to itself, unchanged, at j ≈ 1", () => {
    const res = canonicalize("running", dict, index);
    expect(res.canonical).toBe("running");
    expect(res.changed).toBe(false);
    expect(res.j).toBeCloseTo(1, 5);
    expect(words[res.wi]).toBe("running");
  });

  test("every vocabulary word self-matches unchanged at j = 1", () => {
    for (const w of words) {
      const res = canonicalize(w, dict, index);
      expect(res.canonical).toBe(w);
      expect(res.changed).toBe(false);
      expect(res.j).toBeCloseTo(1, 5);
    }
  });
});

describe("canonicalize — a close typo snaps to the original", () => {
  // "runping" is "running" with one substitution (n → p). It is not itself a
  // vocabulary word, so it must be resolved via the fuzzy NN gate.
  const typo = "runping";

  test("the typo is not itself in the vocabulary", () => {
    expect(words).not.toContain(typo);
  });

  test("clears the gate: changes to the nearest known word with j ≥ theta", () => {
    const res = canonicalize(typo, dict, index);
    expect(res.changed).toBe(true);
    expect(res.j).toBeGreaterThanOrEqual(0.5); // default theta
    expect(words).toContain(res.canonical);
    expect(res.canonical).not.toBe(typo);
  });

  test("resolves to 'running' (its single-edit source)", () => {
    const res = canonicalize(typo, dict, index);
    expect(res.canonical).toBe("running");
  });

  test("a theta above the match Jaccard suppresses the snap (kept unchanged)", () => {
    const res = canonicalize(typo, dict, index, { theta: 1.01 });
    expect(res.changed).toBe(false);
    expect(res.canonical).toBe(typo);
  });
});

describe("canonicalize — a far/garbage string stays unchanged", () => {
  test("a string with no meaningful overlap is kept as-is (below theta)", () => {
    const res = canonicalize("xqzptvw", dict, index);
    expect(res.changed).toBe(false);
    expect(res.canonical).toBe("xqzptvw");
    expect(res.j).toBeLessThan(0.5);
  });
});
