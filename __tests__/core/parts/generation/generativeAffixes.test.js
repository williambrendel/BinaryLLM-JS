"use strict";

// ============================================================================
// __tests__/core/parts/generation/generativeAffixes.test.js
//
// Mirrors src/core/parts/generation/generativeAffixes.js — Track 2, the
// frequency-head affixes. Verifies that a clean, dominant suffix like "ing"
// gets added as an End part, that the addedSet receives the "kind|value" keys,
// and that Start/Mid/End respect their strict position rules.
// ============================================================================

import { generativeAffixes } from "../../../../src/core/parts/generation/generativeAffixes.js";
import { PartDictionary } from "../../../../src/core/parts/dictionary.js";
import { Kind } from "../../../../src/core/parts/kind.js";

// All words alive (nothing pre-filtered as a whole).
const allAlive = (words) => words.map((_, i) => i);

// A dominant "-ing" cluster plus several DISTINCT-suffix, low-frequency
// distractors. The distractors are needed so the length-3 End candidate map has
// ≥3 entries (the elbow criterion is a no-op below 3 points), letting the clean
// "ing" head stand out.
const ING_WORDS = [
  "running", "jumping", "singing", "reading", "walking", "talking", "cooking", "booking",
  "planets", "gardens", "markets", "tickets",
];
const ING_FREQ = Float64Array.from([50, 50, 50, 50, 50, 50, 50, 50, 1, 1, 1, 1]);

describe("generativeAffixes — frequency-head End suffix", () => {
  test('a dominant shared "ing" suffix is added as an End part', () => {
    const words = ING_WORDS;
    const freq = ING_FREQ;
    const dict = new PartDictionary();
    const added = new Set();

    const n = generativeAffixes(words, freq, allAlive(words), dict, added, { lMin: 3, lMax: 7 });

    expect(n).toBeGreaterThan(0);
    expect(dict.hasEnd("ing")).toBe(true);
    // addedSet carries the "kind|value" key for the End "ing".
    expect(added.has(Kind.End + "|ing")).toBe(true);
  });

  test("every key in addedSet corresponds to a real part in the dict", () => {
    const words = ING_WORDS;
    const freq = ING_FREQ;
    const dict = new PartDictionary();
    const added = new Set();

    generativeAffixes(words, freq, allAlive(words), dict, added, { lMin: 3, lMax: 7 });

    expect(added.size).toBeGreaterThan(0);
    for (const key of added) {
      const bar = key.indexOf("|");
      const kind = key.charCodeAt(0) - 48; // keys are `${kind}|value`, kind is 0..5
      const value = key.slice(bar + 1);
      expect(dict.lookup(kind, value)).not.toBe(0xffffffff);
    }
  });
});

describe("generativeAffixes — strict position rules", () => {
  test("Start parts require len > L (a length-L word yields no length-L Start)", () => {
    // Both words have length exactly 3; with L === 3, Start needs len > L, so
    // no length-3 Start candidate can be formed from them.
    const words = ["abc", "abd"];
    const freq = Float64Array.from([100, 100]);
    const dict = new PartDictionary();
    const added = new Set();

    generativeAffixes(words, freq, allAlive(words), dict, added, { lMin: 3, lMax: 3 });

    // No length-3 Start/End possible (len is not > 3). No length-3 Mid either
    // (Mid needs an interior slice: 1 + L <= len - 1 → 4 <= 2, false).
    expect(dict.hasStart("abc")).toBe(false);
    expect(dict.hasEnd("abc")).toBe(false);
    expect(dict.size()).toBe(0);
  });

  test("Mid parts are interior only (never the leading or trailing slice)", () => {
    // "abXYZcd" repeated; the interior length-3 slice "bXY".."XYc" are Mids.
    const words = ["abridge", "abridgy", "abridgz", "planet0"]; // len 7
    const freq = Float64Array.from([100, 100, 100, 1]);
    const dict = new PartDictionary();
    const added = new Set();

    generativeAffixes(words, freq, allAlive(words), dict, added, { lMin: 3, lMax: 3 });

    // Whatever Mids appear must be strictly interior slices: they cannot equal
    // the leading 3 chars ("abr") nor the trailing 3 chars ("dge") of a word,
    // because Mid iterates p in [1, len-1-L].
    for (const key of added) {
      if (key.charCodeAt(0) - 48 !== Kind.Mid) continue;
      const val = key.slice(key.indexOf("|") + 1);
      for (const w of words) {
        expect(val).not.toBe(w.slice(0, 3)); // not the Start slice
        expect(val).not.toBe(w.slice(w.length - 3)); // not the End slice
      }
    }
  });

  test("a flat candidate distribution adds nothing for that length/position", () => {
    // Distinct suffixes, all equal frequency → no elbow → no head taken.
    const words = ["catxxx", "dogyyy", "fishzz", "birdww"];
    const freq = Float64Array.from([1, 1, 1, 1]);
    const dict = new PartDictionary();
    const added = new Set();

    const n = generativeAffixes(words, freq, allAlive(words), dict, added, { lMin: 3, lMax: 3 });

    // With every candidate distinct and equal-weight, no clean head emerges.
    expect(n).toBe(0);
    expect(added.size).toBe(0);
  });
});
