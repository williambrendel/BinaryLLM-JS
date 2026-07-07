"use strict";

// ============================================================================
// __tests__/core/parts/tokenize.test.js
//
// Mirrors src/core/parts/tokenize.js — the streaming tokenizer that splits
// raw text into Word / Delimiter tokens, lowercasing words and normalizing
// whitespace runs into delimiter classes.
// ============================================================================

import { tokenizeStream, asciiLowercase, StreamTokenType } from "../../../src/core/parts/tokenize.js";

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
