"use strict";

// ============================================================================
// __tests__/parts.test.js
//
// Tests for the token part-discovery pipeline ported from BinaryLLM (C++):
// tokenize, byte-string handling, adaptive threshold, dictionary IO,
// decomposer, and the PartExtractor.
//
// The headline test is byte-exact parity: the JS extractor's trained-only
// text dictionary must equal the one produced by the C++ `phase1_extract`
// oracle on the same corpus (fixtures/small_gold_dict.txt).
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { tokenizeStream, asciiLowercase, StreamTokenType } from "../src/core/parts/tokenize.js";
import { toByteString, fromByteString } from "../src/core/parts/byteString.js";
import { Kind, kInvalidPartId } from "../src/core/parts/kind.js";
import {
  PartDictionary,
  saveDictText,
  loadDictText,
  saveDictBinary,
  loadDictBinary,
  augmentWithAtoms,
} from "../src/core/parts/dictionary.js";
import { decomposeWord, decomposeDelimiter, findSingletonRuns } from "../src/core/parts/decomposer.js";
import { PartExtractor } from "../src/core/parts/extractor.js";
import {
  adaptivePruneCount,
  entropyEffectiveCount,
  scaledMmEffectiveCount,
} from "../src/core/math/adaptiveThreshold.js";

const FIX = path.join(path.dirname(fileURLToPath(import.meta.url)), "fixtures");

// Convenience: word/delim values from a UTF-8 JS string.
const words = (raw) =>
  tokenizeStream(raw).filter((t) => t.type === StreamTokenType.Word).map((t) => t.value);
const delims = (raw) =>
  tokenizeStream(raw).filter((t) => t.type === StreamTokenType.Delimiter).map((t) => t.value);

describe("tokenize", () => {
  test("splits words and single-space delimiters", () => {
    const toks = tokenizeStream("hello world");
    expect(toks.map((t) => [t.type, t.value])).toEqual([
      [StreamTokenType.Word, "hello"],
      [StreamTokenType.Delimiter, " "],
      [StreamTokenType.Word, "world"],
    ]);
  });

  test("lowercases words", () => {
    expect(words("Hello WORLD")).toEqual(["hello", "world"]);
    expect(asciiLowercase("AbC123")).toBe("abc123");
  });

  test("keeps in-word connectors", () => {
    expect(words("don't stop")).toEqual(["don't", "stop"]);
    expect(words("rock'n'roll")).toEqual(["rock'n'roll"]);
    expect(words("2,4-d")).toEqual(["2,4-d"]);
  });

  test("decimal vs operator", () => {
    expect(words("3.14")).toEqual(["3.14"]);
    expect(tokenizeStream("5+5").map((t) => t.value)).toEqual(["5", "+", "5"]);
  });

  test("acronyms keep interior dots", () => {
    expect(words("U.S.A")).toEqual(["u.s.a"]);
  });

  test("number prefixes: sign and currency", () => {
    expect(words("$5")).toEqual(["$5"]);
    expect(words("-$5")).toEqual(["-$5"]);
  });

  test("em-dash and ellipsis runs normalize", () => {
    expect(tokenizeStream("a--b").map((t) => t.value)).toEqual(["a", "--", "b"]);
    expect(delims("x... y")).toEqual(["...", " "]);
  });

  test("whitespace run classes", () => {
    expect(delims("a\n\nb")).toEqual(["\n\n"]);
    expect(delims("a\nb")).toEqual(["\n"]);
    expect(delims("a  b")).toEqual(["\t"]); // 2 spaces -> indent class
  });

  test("trailing-apostrophe absorption depends on quote state", () => {
    // Outside a quote, mothers' keeps the trailing apostrophe.
    expect(words("mothers' day")).toEqual(["mothers'", "day"]);
  });
});

describe("byteString", () => {
  test("round-trips UTF-8", () => {
    expect(fromByteString(toByteString("héllo — señor"))).toBe("héllo — señor");
  });

  test("multi-byte chars expand to their UTF-8 bytes", () => {
    expect(toByteString("é").length).toBe(2); // C3 A9
    expect(toByteString("abc").length).toBe(3);
  });

  test("non-ASCII punctuation tokenizes byte-wise", () => {
    // A curly em-dash "—" is 3 UTF-8 bytes, none of them word chars, so it
    // forms a delimiter run kept literally (as its bytes).
    const toks = tokenizeStream("a—b");
    expect(toks.map((t) => t.type)).toEqual([
      StreamTokenType.Word,
      StreamTokenType.Delimiter,
      StreamTokenType.Word,
    ]);
    expect(fromByteString(toks[1].value)).toBe("—");
  });
});

describe("adaptiveThreshold", () => {
  test("entropy: a delta returns 1", () => {
    expect(entropyEffectiveCount([10, 0, 0, 0])).toBe(1);
  });

  test("scaled_mm saturates to n on a flat distribution", () => {
    expect(scaledMmEffectiveCount([1, 1, 1, 1])).toBe(4);
  });

  test("adaptive_prune finds a sharp cliff via the elbow", () => {
    expect(adaptivePruneCount([100, 100, 1, 1, 1, 1, 1, 1, 1, 1])).toBe(2);
  });

  test("adaptive_prune keeps all of a flat distribution", () => {
    expect(adaptivePruneCount([1, 1, 1, 1])).toBe(4);
  });
});

describe("PartDictionary", () => {
  test("add dedups and returns a stable id", () => {
    const d = new PartDictionary();
    const a = d.add(Kind.Whole, "the");
    const b = d.add(Kind.Whole, "the");
    expect(a).toBe(b);
    expect(d.size()).toBe(1);
    // (kind, value) identity: same value under a different kind is distinct.
    expect(d.add(Kind.Start, "the")).not.toBe(a);
    expect(d.size()).toBe(2);
  });

  test("lookup and has* predicates", () => {
    const d = new PartDictionary();
    d.add(Kind.Whole, "the");
    expect(d.hasWhole("the")).toBe(true);
    expect(d.hasStart("the")).toBe(false);
    expect(d.lookup(Kind.Whole, "nope")).toBe(kInvalidPartId);
  });

  test("augmentWithAtoms regenerates singletons/specials", () => {
    const d = new PartDictionary();
    augmentWithAtoms(d);
    expect(d.lookup(Kind.Letter, "a##")).not.toBe(kInvalidPartId);
    expect(d.lookup(Kind.Letter, "##a##")).not.toBe(kInvalidPartId);
    expect(d.lookup(Kind.Letter, "##a")).not.toBe(kInvalidPartId);
    expect(d.hasWhole("a")).toBe(true);
    expect(d.hasDelimiter("-")).toBe(true);
    // 26+10+6 single-char wholes, 42*3 letters, 6 connector delims.
    expect(d.countOfKind(Kind.Whole)).toBe(42);
    expect(d.countOfKind(Kind.Letter)).toBe(126);
    expect(d.countOfKind(Kind.Delimiter)).toBe(6);
  });
});

describe("decomposer", () => {
  test("whole fast path emits a single id", () => {
    const d = new PartDictionary();
    const id = d.add(Kind.Whole, "the");
    augmentWithAtoms(d);
    expect(decomposeWord(d, "the")).toEqual([id]);
  });

  test("letter fill covers every uncovered position", () => {
    const d = new PartDictionary();
    augmentWithAtoms(d); // only singletons available
    const ids = decomposeWord(d, "xyz");
    expect(ids).toHaveLength(3);
    expect(ids.every((i) => i !== kInvalidPartId)).toBe(true);
    // Positional encoding: x## (start), ##y## (mid), ##z (end).
    expect(ids).toEqual([
      d.lookup(Kind.Letter, "x##"),
      d.lookup(Kind.Letter, "##y##"),
      d.lookup(Kind.Letter, "##z"),
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

describe("PartExtractor — parity with the C++ oracle", () => {
  const corpus = new Uint8Array(fs.readFileSync(path.join(FIX, "small_corpus.txt")));
  const gold = fs.readFileSync(path.join(FIX, "small_gold_dict.txt"), "latin1");

  const build = () => {
    const ex = new PartExtractor();
    for (const tok of tokenizeStream(corpus)) {
      if (tok.type === StreamTokenType.Word) ex.addWord(tok.value);
      else ex.addDelimiter(tok.value);
    }
    return { ex, dict: ex.finalize() };
  };

  test("trained-only text dictionary matches the oracle byte-for-byte", () => {
    const { dict } = build();
    expect(saveDictText(dict)).toBe(gold);
  });

  test("dictionary size and per-kind counts match the oracle", () => {
    const { dict } = build();
    expect(dict.size()).toBe(509);
    expect(dict.countOfKind(Kind.Whole)).toBe(52);
    expect(dict.countOfKind(Kind.Start)).toBe(63);
    expect(dict.countOfKind(Kind.Mid)).toBe(166);
    expect(dict.countOfKind(Kind.End)).toBe(89);
    expect(dict.countOfKind(Kind.Letter)).toBe(126);
    expect(dict.countOfKind(Kind.Delimiter)).toBe(13);
  });

  test("extraction is deterministic", () => {
    expect(saveDictText(build().dict)).toBe(saveDictText(build().dict));
  });

  test("every training word decomposes to all-valid part ids", () => {
    const { dict } = build();
    const seen = new Set();
    for (const tok of tokenizeStream(corpus)) {
      if (tok.type !== StreamTokenType.Word || seen.has(tok.value)) continue;
      seen.add(tok.value);
      const ids = decomposeWord(dict, tok.value);
      expect(ids.length).toBeGreaterThan(0);
      expect(ids.every((i) => i !== kInvalidPartId)).toBe(true);
    }
  });
});

describe("dictionary IO round-trips", () => {
  const buildDict = () => {
    const ex = new PartExtractor();
    for (const tok of tokenizeStream(fs.readFileSync(path.join(FIX, "small_corpus.txt")))) {
      if (tok.type === StreamTokenType.Word) ex.addWord(tok.value);
      else ex.addDelimiter(tok.value);
    }
    return ex.finalize();
  };

  test("text save -> load -> save is stable", () => {
    const dict = buildDict();
    const text = saveDictText(dict);
    const reloaded = loadDictText(text);
    expect(saveDictText(reloaded)).toBe(text);
  });

  test("binary save -> load reproduces the same trained parts", () => {
    const dict = buildDict();
    const bin = saveDictBinary(dict);
    const reloaded = loadDictBinary(bin);
    // Binary and text share the trained-only contract, so their text forms match.
    expect(saveDictText(reloaded)).toBe(saveDictText(dict));
  });

  test("loadDictBinary rejects bad magic", () => {
    expect(() => loadDictBinary(Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]))).toThrow(/magic/);
  });
});
