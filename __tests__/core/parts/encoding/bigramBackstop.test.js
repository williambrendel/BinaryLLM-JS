"use strict";

// ============================================================================
// __tests__/core/parts/encoding/bigramBackstop.test.js
//
// Mirrors src/core/parts/encoding/bigramBackstop.js — seeds the fixed 676
// lowercase bigrams under all three affix positions (Start/Mid/End). Verifies
// the counts, a few known lookups, chaining, and idempotency.
// ============================================================================

import { Kind } from "../../../../src/core/parts/kind.js";
import { PartDictionary } from "../../../../src/core/parts/dictionary.js";
import { addBigramBackstop } from "../../../../src/core/parts/encoding/bigramBackstop.js";

describe("bigramBackstop — coverage", () => {
  test("adds 676 Start + 676 Mid + 676 End parts", () => {
    const dict = new PartDictionary();
    addBigramBackstop(dict);
    expect(dict.countOfKind(Kind.Start)).toBe(676);
    expect(dict.countOfKind(Kind.Mid)).toBe(676);
    expect(dict.countOfKind(Kind.End)).toBe(676);
    expect(dict.size()).toBe(676 * 3);
  });

  test("registers known bigrams under each position", () => {
    const dict = new PartDictionary();
    addBigramBackstop(dict);
    expect(dict.hasMid("th")).toBe(true);
    expect(dict.hasStart("qu")).toBe(true);
    expect(dict.hasEnd("zz")).toBe(true);
    expect(dict.hasStart("aa")).toBe(true);
  });
});

describe("bigramBackstop — chaining and idempotency", () => {
  test("returns the same dictionary for chaining", () => {
    const dict = new PartDictionary();
    expect(addBigramBackstop(dict)).toBe(dict);
  });

  test("is idempotent — size is unchanged on a second call", () => {
    const dict = new PartDictionary();
    addBigramBackstop(dict);
    const sizeAfterFirst = dict.size();
    addBigramBackstop(dict);
    expect(dict.size()).toBe(sizeAfterFirst);
  });
});
