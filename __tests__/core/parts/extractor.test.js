"use strict";

// ============================================================================
// __tests__/core/parts/extractor.test.js
//
// Mirrors src/core/parts/extractor.js — the PartExtractor that trains a part
// dictionary from a token stream. The headline test is a REGRESSION SNAPSHOT:
// the JS extractor's text dictionary must equal the checked-in fixture
// (__tests__/fixtures/small_gold_dict.txt), regenerated from the JS behavior.
// The file now includes the single-char atoms (only Delimiters are regenerated
// at load) so part ids are stable across save → load.
//
// NOTE: this fixture diverged from the C++ `phase1_extract` oracle when the JS
// tokenizer dropped the `'`/`&` in-word connectors and switched to a fixed
// manual delimiter set (and, later, when atoms began being serialized). Port
// those changes to the C++ side to re-establish
// cross-implementation parity.
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

describe("PartExtractor — regression snapshot (JS; diverged from C++ oracle)", () => {
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

  test("text dictionary (atoms included) matches the snapshot byte-for-byte", () => {
    const { dict } = build();
    expect(saveDictText(dict)).toBe(gold);
  });

  test("dictionary size and per-kind counts match the snapshot", () => {
    const { dict } = build();
    expect(dict.size()).toBe(519);
    expect(dict.countOfKind(Kind.Whole)).toBe(50);
    // Single chars are now length-1 Start/Mid/End atoms (40 chars each), folded
    // into the learned counts: Start 61+40, Mid 163+40, End 87+40. Total unchanged.
    expect(dict.countOfKind(Kind.Start)).toBe(101);
    expect(dict.countOfKind(Kind.Mid)).toBe(203);
    expect(dict.countOfKind(Kind.End)).toBe(127);
    expect(dict.countOfKind(Kind.Delimiter)).toBe(38);
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
