"use strict";

// ============================================================================
// __tests__/core/parts/dictionary.test.js
//
// Mirrors src/core/parts/dictionary.js — the PartDictionary (dedup add,
// lookup, per-kind counts), atom augmentation, and the text/binary IO
// round-trips. The IO tests build a real dictionary from the fixture corpus
// via the extractor and assert the serialized forms are stable.
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  PartDictionary,
  saveDictText,
  loadDictText,
  saveDictBinary,
  loadDictBinary,
  augmentWithAtoms,
} from "../../../src/core/parts/dictionary.js";
import { Kind, kInvalidPartId } from "../../../src/core/parts/kind.js";
import { PartExtractor } from "../../../src/core/parts/extractor.js";
import { tokenizeStream, StreamTokenType } from "../../../src/core/parts/tokenize.js";

const FIX = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "fixtures");

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
