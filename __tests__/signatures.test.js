"use strict";

// ============================================================================
// __tests__/signatures.test.js
//
// Tests for the 3F signature encoder ported from core/signatures.
//
// A signature is a list of F-wide bands [L, C, R] (TypedArray[]), each holding
// local ids in [0,F). `flatten` concatenates bands into one global-indexed
// vector. The global-indexed anchor values (e.g. the 3F bits for "peppers")
// come from the C++ `debug_decompose --show=...` oracle on the committed
// fixture dictionary, so these pin parity without the C++ binary at run time.
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
  lcr,
  lr,
  lc,
  lUnionC,
  flatten,
} from "../src/core/signatures/encoder.js";

const FIX = path.join(path.dirname(fileURLToPath(import.meta.url)), "fixtures");
const dict = loadDictText(fs.readFileSync(path.join(FIX, "small_gold_dict.txt"), "latin1"));
const F = dict.size(); // 509 for this fixture

// First-occurrence signature (a [L, C, R] band list) for a word in a sentence.
const sigFor = (sentence, word) => {
  const toks = tokenizeStream(sentence);
  const sigs = encode(dict, toks);
  let w = -1;
  for (const t of toks) {
    if (t.type !== StreamTokenType.Word) continue;
    w++;
    if (t.value === word) return sigs[w];
  }
  return null;
};

describe("encoder band element type (keyed on F, see utilities/sparseArrayType)", () => {
  test("bands are Uint16Array for this fixture (F=509)", () => {
    expect(F).toBe(509);
    const sig = encode(dict, tokenizeStream("the peppers"))[1];
    expect(sig).toHaveLength(3);
    for (const band of sig) expect(band).toBeInstanceOf(Uint16Array);
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
  test('"peppers" bands in "the peppers were pickled"', () => {
    const [L, C, R] = sigFor("the peppers were pickled", "peppers");
    expect(Array.from(L)).toEqual([0]); // the(0)
    expect(Array.from(C)).toEqual([1]); // peppers(1)
    expect(Array.from(R)).toEqual([7, 9]); // were(9) ∪ pickled(7)
    // Global-indexed flat matches the oracle's 3F bits exactly.
    expect(Array.from(flatten(sigFor("the peppers were pickled", "peppers"), F))).toEqual([
      0, 510, 1025, 1027,
    ]);
  });

  test("bands hold local ids in [0,F); C band == Option B bag", () => {
    const toks = tokenizeStream("unhappily running players");
    const sigs = encode(dict, toks);
    const words = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => t.value);
    sigs.forEach((sig, i) => {
      for (const band of sig) {
        const arr = Array.from(band);
        expect(arr).toEqual([...arr].sort((a, b) => a - b)); // sorted
        expect(arr.every((id) => id >= 0 && id < F)).toBe(true); // local, in [0,F)
      }
      expect(Array.from(sig[1])).toEqual(encodeWord(dict, words[i])); // C == bag
    });
  });

  test("first word has empty L, last word has empty R", () => {
    const sigs = encode(dict, tokenizeStream("alpha beta gamma"));
    expect(sigs[0][0].length).toBe(0); // L
    expect(sigs[2][2].length).toBe(0); // R
  });
});

describe("encodeWindowed", () => {
  const sentence = "the peppers were pickled peppers games";
  const toks = tokenizeStream(sentence);
  const asArrays = (sigs) => sigs.map((s) => s.map((b) => Array.from(b)));

  test("radius >= N-1 equals whole-scope encode", () => {
    expect(asArrays(encodeWindowed(dict, toks, 99))).toEqual(asArrays(encode(dict, toks)));
  });

  test("radius 0 keeps only the current band", () => {
    const sigs = encodeWindowed(dict, toks, 0);
    const words = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => t.value);
    sigs.forEach(([L, C, R], i) => {
      expect(L.length).toBe(0);
      expect(R.length).toBe(0);
      expect(Array.from(C)).toEqual(encodeWord(dict, words[i]));
    });
  });

  test("radius 1 pools only immediate neighbors", () => {
    // word index 2 ("were"): L = current[1] ("peppers"), R = current[3] ("pickled").
    const [L, , R] = encodeWindowed(dict, toks, 1)[2];
    expect(Array.from(L)).toEqual(encodeWord(dict, "peppers"));
    expect(Array.from(R)).toEqual(encodeWord(dict, "pickled"));
  });
});

describe("band selectors + flatten", () => {
  // peppers: L={0}, C={1}, R={7,9}; F=509.
  const sig = [Uint16Array.from([0]), Uint16Array.from([1]), Uint16Array.from([7, 9])];

  test("selectors return banded signatures (TypedArray[])", () => {
    expect(lcr(sig).map((b) => Array.from(b))).toEqual([[0], [1], [7, 9]]);
    expect(lr(sig).map((b) => Array.from(b))).toEqual([[0], [7, 9]]);
    expect(lc(sig).map((b) => Array.from(b))).toEqual([[0], [1]]);
    expect(lUnionC(sig, F).map((b) => Array.from(b))).toEqual([[0, 1]]);
  });

  test("flatten concatenates with band k shifted by k*F", () => {
    expect(Array.from(flatten(lcr(sig), F))).toEqual([0, 510, 1025, 1027]); // [L|C|R]
    expect(Array.from(flatten(lr(sig), F))).toEqual([0, 516, 518]); // [L | R]
    expect(Array.from(flatten(lc(sig), F))).toEqual([0, 510]); // [L | C]
    expect(Array.from(flatten(lUnionC(sig, F), F))).toEqual([0, 1]); // [L ∪ C]
  });

  test("[L∪C] collapses an id shared by L and C", () => {
    const s = [Uint16Array.from([3]), Uint16Array.from([3]), Uint16Array.from([])];
    expect(Array.from(flatten(lUnionC(s, F), F))).toEqual([3]);
  });

  test("flatten picks the element type from nBands*F, not F", () => {
    const big = [Uint16Array.from([1]), Uint16Array.from([2]), Uint16Array.from([3])];
    // Bands are Uint16; only the flat total dimension decides the flat's type.
    expect(flatten(lcr(big), 40000)).toBeInstanceOf(Uint32Array); // 3F = 120000 > 65536
    expect(flatten(lUnionC(big, 40000), 40000)).toBeInstanceOf(Uint16Array); // F = 40000 <= 65536
  });
});
