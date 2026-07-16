"use strict";

// ============================================================================
// __tests__/core/parts/generation/prefilterWholes.test.js
//
// Mirrors src/core/parts/generation/prefilterWholes.js — Track 1, the
// whole-word atoms. Verifies the two length regimes: 1–2 letter words are all
// taken whole regardless of frequency, and 3+ letter buckets contribute only
// their clean frequency head (gated by the elbow criterion).
// ============================================================================

import { prefilterWholes } from "../../../../src/core/parts/generation/prefilterWholes.js";
import { PartDictionary } from "../../../../src/core/parts/dictionary.js";
import { Kind } from "../../../../src/core/parts/kind.js";

describe("prefilterWholes — short words (len ≤ shortWholeMax)", () => {
  test("all 1–2 letter words are taken whole regardless of frequency", () => {
    const words = ["a", "i", "of", "to", "an"];
    const freq = Float64Array.from([1, 999, 3, 2, 1]); // wildly uneven
    const dict = new PartDictionary();
    const removed = new Uint8Array(words.length);

    const added = prefilterWholes(words, freq, dict, removed);

    // Every short word is a whole, flagged in `removed`, present in the dict.
    for (let i = 0; i < words.length; i++) {
      expect(removed[i]).toBe(1);
      expect(dict.hasWhole(words[i])).toBe(true);
    }
    // Count returned matches the number added; all five are shorts here.
    expect(added).toBe(words.length);
    expect(dict.countOfKind(Kind.Whole)).toBe(words.length);
  });

  test("a low-frequency short word is still taken whole", () => {
    const words = ["xy"]; // frequency 1, but len 2 → always whole
    const freq = Float64Array.from([1]);
    const dict = new PartDictionary();
    const removed = new Uint8Array(words.length);

    const added = prefilterWholes(words, freq, dict, removed);

    expect(added).toBe(1);
    expect(removed[0]).toBe(1);
    expect(dict.hasWhole("xy")).toBe(true);
  });
});

describe("prefilterWholes — longer buckets (frequency head only)", () => {
  test("a bucket with a clear frequency head takes only the head", () => {
    // Six length-4 words: one dominant "the-like" head, a flat tail.
    const words = ["that", "moon", "star", "lamp", "desk", "boat"];
    const freq = Float64Array.from([1000, 1, 1, 1, 1, 1]);
    const dict = new PartDictionary();
    const removed = new Uint8Array(words.length);

    const added = prefilterWholes(words, freq, dict, removed);

    // The dominant head is taken; the flat tail is not atomized.
    expect(dict.hasWhole("that")).toBe(true);
    expect(removed[0]).toBe(1);

    // Fewer than the whole bucket were taken (head-only behaviour).
    expect(added).toBeGreaterThan(0);
    expect(added).toBeLessThan(words.length);

    // Tail words were left alive.
    for (let i = 1; i < words.length; i++) {
      if (!dict.hasWhole(words[i])) expect(removed[i]).toBe(0);
    }
  });

  test("a perfectly flat bucket contributes no wholes", () => {
    // No elbow → elbowEffectiveCount returns n → nothing cut → nothing taken.
    const words = ["moon", "star", "lamp", "desk", "boat"];
    const freq = Float64Array.from([1, 1, 1, 1, 1]);
    const dict = new PartDictionary();
    const removed = new Uint8Array(words.length);

    const added = prefilterWholes(words, freq, dict, removed);

    expect(added).toBe(0);
    expect(dict.countOfKind(Kind.Whole)).toBe(0);
    for (let i = 0; i < words.length; i++) expect(removed[i]).toBe(0);
  });

  test("return count equals the number of removed flags set", () => {
    const words = ["a", "of", "that", "moon", "star", "lamp", "desk", "boat"];
    const freq = Float64Array.from([5, 4, 1000, 1, 1, 1, 1, 1]);
    const dict = new PartDictionary();
    const removed = new Uint8Array(words.length);

    const added = prefilterWholes(words, freq, dict, removed);

    let flags = 0;
    for (let i = 0; i < removed.length; i++) flags += removed[i];
    expect(added).toBe(flags);
    // Both shorts are taken regardless of the head decision on the len-4 bucket.
    expect(dict.hasWhole("a")).toBe(true);
    expect(dict.hasWhole("of")).toBe(true);
  });
});
