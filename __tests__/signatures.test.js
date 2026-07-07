"use strict";

// ============================================================================
// __tests__/signatures.test.js
//
// Tests for the 3F signature encoder ported from core/signatures.
//
// The anchor values (e.g. the 3F bits for "peppers") were produced by the C++
// `debug_decompose --show=...` oracle on the committed fixture dictionary, so
// these assertions pin byte-parity without needing the C++ binary at run time.
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { tokenizeStream, StreamTokenType } from "../src/core/parts/tokenize.js";
import { loadDictText } from "../src/core/parts/dictionary.js";
import { encodeWord } from "../src/core/signatures/wordEncoder.js";
import {
  encode,
  encodeWindowed,
  sparseArrayType,
  toLCR,
  toLR,
  toLunionC,
  toLC,
} from "../src/core/signatures/encoder.js";

const FIX = path.join(path.dirname(fileURLToPath(import.meta.url)), "fixtures");
const dict = loadDictText(fs.readFileSync(path.join(FIX, "small_gold_dict.txt"), "latin1"));
const F = dict.size(); // 509 for this fixture

// First-occurrence signature for a word in a sentence.
const sigFor = (sentence, word) => {
  const toks = tokenizeStream(sentence);
  const sigs = encode(dict, toks);
  let w = -1;
  for (const t of toks) {
    if (t.type !== StreamTokenType.Word) continue;
    w++;
    if (t.value === word) return Array.from(sigs[w]);
  }
  return null;
};

describe("sparseArrayType", () => {
  test("Uint16 up to 65536 bit positions, Uint32 beyond", () => {
    expect(sparseArrayType(1527)).toBe(Uint16Array);
    expect(sparseArrayType(65536)).toBe(Uint16Array);
    expect(sparseArrayType(65537)).toBe(Uint32Array);
    expect(sparseArrayType(3 * 33104)).toBe(Uint32Array); // dict_combined-scale
  });

  test("encoder picks the element type from 3F", () => {
    expect(F).toBe(509);
    const sig = encode(dict, tokenizeStream("the peppers"))[0];
    expect(sig).toBeInstanceOf(Uint16Array); // 3F = 1527 <= 65536
  });
});

describe("encodeWord", () => {
  test("Option B bag is sorted and de-duplicated", () => {
    const ids = encodeWord(dict, "peppers");
    expect(ids).toEqual([...ids].sort((a, b) => a - b));
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe("encode (3F) — parity with debug_decompose", () => {
  test('"peppers" in "the peppers were pickled"', () => {
    // Oracle: L=the(0) | C=peppers(509+1) | R=were∪pickled(2F+7, 2F+9)
    expect(sigFor("the peppers were pickled", "peppers")).toEqual([0, 510, 1025, 1027]);
  });

  test("bands are well-formed: L<F, C in [F,2F) equals F+bag, R>=2F", () => {
    const toks = tokenizeStream("unhappily running players");
    const sigs = encode(dict, toks);
    const words = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => t.value);
    sigs.forEach((sig, i) => {
      const arr = Array.from(sig);
      // sorted ascending
      expect(arr).toEqual([...arr].sort((a, b) => a - b));
      const current = arr.filter((b) => b >= F && b < 2 * F).map((b) => b - F);
      expect(current).toEqual(encodeWord(dict, words[i])); // C band == Option B bag
      expect(arr.filter((b) => b >= 2 * F).every((b) => b < 3 * F)).toBe(true);
    });
  });

  test("first word has empty L, last word has empty R", () => {
    const toks = tokenizeStream("alpha beta gamma");
    const sigs = encode(dict, toks);
    expect(Array.from(sigs[0]).some((b) => b < F)).toBe(false); // no before band
    expect(Array.from(sigs[2]).some((b) => b >= 2 * F)).toBe(false); // no after band
  });
});

describe("encodeWindowed", () => {
  const sentence = "the peppers were pickled peppers games";
  const toks = tokenizeStream(sentence);

  test("radius >= N-1 equals whole-scope encode", () => {
    const wide = encodeWindowed(dict, toks, 99).map((s) => Array.from(s));
    const whole = encode(dict, toks).map((s) => Array.from(s));
    expect(wide).toEqual(whole);
  });

  test("radius 0 keeps only the current band", () => {
    const sigs = encodeWindowed(dict, toks, 0);
    const words = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => t.value);
    sigs.forEach((sig, i) => {
      expect(Array.from(sig)).toEqual(encodeWord(dict, words[i]).map((id) => F + id));
    });
  });

  test("radius 1 pools only immediate neighbors", () => {
    // For word index 2 ("were"), L should be just current[1] ("peppers"),
    // R should be just current[3] ("pickled").
    const sigs = encodeWindowed(dict, toks, 1);
    const arr = Array.from(sigs[2]);
    const L = arr.filter((b) => b < F);
    const R = arr.filter((b) => b >= 2 * F).map((b) => b - 2 * F);
    expect(L).toEqual(encodeWord(dict, "peppers"));
    expect(R).toEqual(encodeWord(dict, "pickled"));
  });
});

describe("band views over a 3F signature", () => {
  // peppers sig = [0, 510, 1025, 1027] with F=509 (2F=1018).
  const sig = Uint16Array.from([0, 510, 1025, 1027]);

  test("toLCR is an identity copy", () => {
    const c = toLCR(sig);
    expect(Array.from(c)).toEqual([0, 510, 1025, 1027]);
    expect(c).not.toBe(sig); // copy, not alias
  });

  test("toLunionC unions L and C into [0,F)", () => {
    expect(Array.from(toLunionC(sig, F))).toEqual([0, 1]); // 0 (L) ∪ 510-509 (C)
  });

  test("toLR keeps L in [0,F) and R in [F,2F)", () => {
    expect(Array.from(toLR(sig, F))).toEqual([0, 516, 518]); // 0 | 1025-509, 1027-509
  });

  test("toLC keeps L and C, drops R", () => {
    expect(Array.from(toLC(sig, F))).toEqual([0, 510]);
  });

  test("union can collapse a shared id", () => {
    // If L and C both carried id 3 (bits 3 and F+3), [L∪C] has it once.
    const s = Uint16Array.from([3, F + 3]);
    expect(Array.from(toLunionC(s, F))).toEqual([3]);
  });
});
