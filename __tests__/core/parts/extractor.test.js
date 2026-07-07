"use strict";

// ============================================================================
// __tests__/core/parts/extractor.test.js
//
// Mirrors src/core/parts/extractor.js — the PartExtractor that trains a part
// dictionary from a token stream. The headline test is byte-exact parity:
// the JS extractor's trained-only text dictionary must equal the one produced
// by the C++ `phase1_extract` oracle on the same corpus
// (fixtures/small_gold_dict.txt).
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { PartExtractor } from "../../../src/core/parts/extractor.js";
import { tokenizeStream, StreamTokenType } from "../../../src/core/parts/tokenize.js";
import { saveDictText } from "../../../src/core/parts/dictionary.js";
import { decomposeWord } from "../../../src/core/parts/decomposer.js";
import { Kind, kInvalidPartId } from "../../../src/core/parts/kind.js";

const FIX = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "fixtures");

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
