"use strict";

// ============================================================================
// __tests__/utilities/corpus/wordFreq.test.js
//
// Mirrors src/utilities/corpus/wordFreq.js — the reusable corpus helper that
// tokenizes one or more byte sources into a word -> token-count Map. Only Word
// tokens are counted (Delimiters ignored); values are the (lowercased)
// byte-string produced by tokenizeStream, and counts are summed across sources.
// ============================================================================

import { wordFreq } from "../../../src/utilities/corpus/wordFreq.js";

const enc = (s) => new TextEncoder().encode(s);

describe("wordFreq — basic counting", () => {
  test('counts word occurrences: "the cat the dog"', () => {
    const freq = wordFreq([enc("the cat the dog")]);
    expect(freq.get("the")).toBe(2);
    expect(freq.get("cat")).toBe(1);
    expect(freq.get("dog")).toBe(1);
  });

  test("returns a Map with exactly the distinct words", () => {
    const freq = wordFreq([enc("the cat the dog")]);
    expect(freq).toBeInstanceOf(Map);
    expect(freq.size).toBe(3);
    expect([...freq.keys()].sort()).toEqual(["cat", "dog", "the"]);
  });

  test("lowercases words so 'The' and 'the' collapse", () => {
    const freq = wordFreq([enc("The the THE")]);
    expect(freq.get("the")).toBe(3);
    expect(freq.has("The")).toBe(false);
  });
});

describe("wordFreq — delimiters", () => {
  test("delimiters (punctuation and whitespace) are not counted", () => {
    const freq = wordFreq([enc("the, cat. the!  dog")]);
    // Only word tokens counted; ",", ".", "!", spaces are all delimiters.
    expect(freq.get("the")).toBe(2);
    expect(freq.get("cat")).toBe(1);
    expect(freq.get("dog")).toBe(1);
    expect(freq.size).toBe(3);
    // No delimiter value ever ends up as a key.
    for (const k of freq.keys()) expect(k).toMatch(/^[a-z0-9$,.\\-]+$/);
  });

  test("newlines and tabs separate words but are not counted", () => {
    const freq = wordFreq([enc("the\tcat\nthe\ndog")]);
    expect(freq.get("the")).toBe(2);
    expect(freq.get("cat")).toBe(1);
    expect(freq.get("dog")).toBe(1);
    expect(freq.size).toBe(3);
  });
});

describe("wordFreq — multiple sources", () => {
  test("counts from separate sources are summed", () => {
    const freq = wordFreq([enc("the cat"), enc("the dog"), enc("the the")]);
    expect(freq.get("the")).toBe(4);
    expect(freq.get("cat")).toBe(1);
    expect(freq.get("dog")).toBe(1);
  });

  test("accepts string sources as well as byte arrays", () => {
    const freq = wordFreq(["the cat", enc("the dog")]);
    expect(freq.get("the")).toBe(2);
    expect(freq.get("cat")).toBe(1);
    expect(freq.get("dog")).toBe(1);
  });
});

describe("wordFreq — edge cases", () => {
  test("no sources yields an empty Map", () => {
    const freq = wordFreq([]);
    expect(freq).toBeInstanceOf(Map);
    expect(freq.size).toBe(0);
  });

  test("a source of only delimiters yields an empty Map", () => {
    const freq = wordFreq([enc("  ,. \n\t ")]);
    expect(freq).toBeInstanceOf(Map);
    expect(freq.size).toBe(0);
  });
});
