/**
 * @file protectDots.test.js
 * @description Unit tests for the protectDots module.
 *
 * protectDots(text) → { protectedText, dictionary }
 * restore(text, dictionary) → text
 *
 * Tokens use NULL control character delimiters: \x00TOK_<n>\x00
 * Patterns are applied in order — order matters for correctness.
 */
"use strict";

import { protectDots, restore } from "../../../src/utilities/textSegmentation/protectDots";

/**
 * Helper: runs protectDots, then verifies that restoring the protected text
 * yields the original input exactly. This is the fundamental round-trip
 * invariant the module must uphold.
 */
const expectRoundTrip = (input) => {
  const { protectedText, dictionary } = protectDots(input);
  expect(restore(protectedText, dictionary)).toBe(input);
  return { protectedText, dictionary };
};

/**
 * Helper: counts how many "." characters remain in the protected text.
 * Useful for asserting that protected patterns no longer expose dots.
 */
const countDots = (s) => (s.match(/\./g) || []).length;

describe("protectDots", () => {
  describe("basic behavior", () => {
    test("returns an object with protectedText and dictionary", () => {
      const result = protectDots("hello world");
      expect(result).toHaveProperty("protectedText");
      expect(result).toHaveProperty("dictionary");
      expect(typeof result.protectedText).toBe("string");
      expect(typeof result.dictionary).toBe("object");
    });

    test("returns input unchanged when no protected patterns exist", () => {
      const input = "This is a plain sentence. So is this one!";
      const { protectedText, dictionary } = protectDots(input);
      expect(protectedText).toBe(input);
      expect(Object.keys(dictionary)).toHaveLength(0);
    });

    test("handles empty string", () => {
      const { protectedText, dictionary } = protectDots("");
      expect(protectedText).toBe("");
      expect(dictionary).toEqual({});
    });

    test("handles whitespace-only string", () => {
      expectRoundTrip("   \n\n   ");
    });
  });

  describe("decimal numbers", () => {
    test("protects simple decimals", () => {
      const { protectedText, dictionary } = protectDots("pi is 3.14 approximately");
      expect(protectedText).not.toContain("3.14");
      expect(Object.values(dictionary)).toContain("3.14");
    });

    test("protects multiple decimals", () => {
      const { dictionary } = protectDots("Values: 1.5, 2.7, and 3.14159.");
      expect(Object.values(dictionary)).toEqual(
        expect.arrayContaining(["1.5", "2.7", "3.14159"])
      );
    });

    test("protects decimals with thousands separators", () => {
      const { dictionary } = protectDots("Total: 1,234.56 dollars.");
      expect(Object.values(dictionary)).toContain("1,234.56");
    });

    test("protects leading-dot decimals like .5", () => {
      const { dictionary } = protectDots("The value was .5 meters.");
      expect(Object.values(dictionary)).toContain(".5");
    });

    test("does not affect integers", () => {
      const input = "There are 42 items.";
      const { protectedText } = protectDots(input);
      expect(protectedText).toContain("42");
    });
  });

  describe("acronyms and initials", () => {
    test("protects multi-letter acronyms", () => {
      const { dictionary } = protectDots("She lives in the U.S.A. now.");
      expect(Object.values(dictionary)).toContain("U.S.A.");
    });

    test("protects two-letter initials", () => {
      const { dictionary } = protectDots("J.K. Rowling wrote the book.");
      expect(Object.values(dictionary).some((v) => v.includes("J.K."))).toBe(true);
    });

    test("protects e.g. and i.e.", () => {
      const { dictionary } = protectDots("Citrus fruits, e.g. oranges and lemons.");
      const values = Object.values(dictionary);
      expect(values.some((v) => v.includes("e.g"))).toBe(true);
    });
  });

  describe("honorifics and abbreviations", () => {
    test.each([
      ["Dr. Smith arrived.", "Dr."],
      ["Mr. Jones called.", "Mr."],
      ["Mrs. Davis agreed.", "Mrs."],
      ["Ms. Lee responded.", "Ms."],
      ["Prof. Allen taught.", "Prof."],
      ["See Fig. 3 below.", "Fig."],
      ["Refer to Eq. 12.", "Eq."],
      ["Vol. 5 of the journal.", "Vol."],
    ])("protects honorific in: %s", (input, expected) => {
      const { dictionary } = protectDots(input);
      expect(Object.values(dictionary)).toContain(expected);
    });

    test("protects et al.", () => {
      const { dictionary } = protectDots("Smith et al. demonstrated the effect.");
      const values = Object.values(dictionary);
      expect(values.some((v) => v.includes("et"))).toBe(true);
    });
  });

  describe("outline numbering", () => {
    test("protects single-token line-start headers (digit)", () => {
      const input = "1. First item\n2. Second item";
      const { protectedText, dictionary } = protectDots(input);
      expect(Object.values(dictionary)).toEqual(
        expect.arrayContaining(["1.", "2."])
      );
      expect(protectedText).not.toMatch(/^\s*1\./m);
    });

    test("protects single-letter outline headers (A., a.)", () => {
      const input = "A. First\nb. second\n";
      const { dictionary } = protectDots(input);
      expect(Object.values(dictionary)).toEqual(
        expect.arrayContaining(["A.", "b."])
      );
    });

    test("protects roman numeral outline headers (vii., IV.)", () => {
      const input = "vii. seventh item\nIV. fourth item\n";
      const { dictionary } = protectDots(input);
      expect(Object.values(dictionary)).toEqual(
        expect.arrayContaining(["vii.", "IV."])
      );
    });

    test("protects multi-level numbering (1.2.3)", () => {
      const { dictionary } = protectDots("See section 1.2.3 for details.");
      expect(Object.values(dictionary)).toContain("1.2.3");
    });

    test("protects mixed multi-level numbering (A.1.b)", () => {
      const { dictionary } = protectDots("Refer to A.1.b in the appendix.");
      expect(Object.values(dictionary)).toContain("A.1.b");
    });
  });

  describe("URLs and emails", () => {
    test("protects http URLs", () => {
      const { dictionary } = protectDots("Visit http://example.com for info.");
      expect(Object.values(dictionary)).toContain("http://example.com");
    });

    test("protects https URLs", () => {
      const { dictionary } = protectDots("See https://example.com/path?q=1 here.");
      expect(Object.values(dictionary).some((v) => v.startsWith("https://"))).toBe(true);
    });

    test("protects www URLs without scheme", () => {
      const { dictionary } = protectDots("Go to www.example.com today.");
      expect(Object.values(dictionary).some((v) => v.startsWith("www."))).toBe(true);
    });

    test("protects email addresses", () => {
      const { dictionary } = protectDots("Email me at user@example.com please.");
      expect(Object.values(dictionary)).toContain("user@example.com");
    });

    test("protects emails with dots in local part", () => {
      const { dictionary } = protectDots("Contact john.doe@company.co.uk now.");
      expect(Object.values(dictionary).some((v) => v.includes("john.doe"))).toBe(true);
    });
  });

  describe("filenames", () => {
    test("protects simple filenames", () => {
      const { dictionary } = protectDots("Open the file.txt now.");
      expect(Object.values(dictionary)).toContain("file.txt");
    });

    test("protects compound extensions", () => {
      const { dictionary } = protectDots("Extract archive.tar.gz here.");
      expect(Object.values(dictionary)).toContain("archive.tar.gz");
    });
  });

  describe("segmentation enables clean splitting", () => {
    test("real sentence boundaries remain after protection", () => {
      const input = "Dr. Smith said pi is 3.14. The next sentence follows.";
      const { protectedText, dictionary } = protectDots(input);
      const remainingDots = countDots(protectedText);
      expect(remainingDots).toBe(2); // one after "3.14", one after "follows"

      const segments = protectedText.split(/(?<=[.!?])\s+/);
      expect(segments).toHaveLength(2);

      const restored = segments.map((s) => restore(s, dictionary));
      expect(restored[0]).toContain("Dr. Smith");
      expect(restored[0]).toContain("3.14");
      expect(restored[1]).toContain("next sentence");
    });

    test("blank-line splitting still works", () => {
      const input = "First paragraph with Dr. Smith.\n\nSecond paragraph at 3.14.";
      const { protectedText, dictionary } = protectDots(input);
      const paragraphs = protectedText.split(/\n\s*\n/);
      expect(paragraphs).toHaveLength(2);
      const restored = paragraphs.map((p) => restore(p, dictionary));
      expect(restored[0]).toContain("Dr. Smith");
      expect(restored[1]).toContain("3.14");
    });

    test("question marks and exclamation points are unaffected", () => {
      const input = "Is pi 3.14? Yes! Dr. Smith confirmed it.";
      const { protectedText } = protectDots(input);
      expect(protectedText).toContain("?");
      expect(protectedText).toContain("!");
    });
  });

  describe("output safe for character-code segmentation", () => {
    test("placeholder tokens contain no '.' character", () => {
      const { protectedText } = protectDots("Dr. Smith found pi = 3.14");
      const tokenMatches = protectedText.match(/\x00TOK_\d+\x00/g) || [];
      for (const tok of tokenMatches) {
        expect(tok).not.toContain(".");
      }
    });

    test("protected text has fewer dots than original when patterns match", () => {
      const input = "Dr. Smith found pi = 3.14. End";
      const { protectedText } = protectDots(input);
      const originalDots = (input.match(/\./g) || []).length;
      const protectedDots = (protectedText.match(/\./g) || []).length;
      // We protect "Dr." (1 dot) and "3.14" (1 dot), leaving 1 sentence-ending dot
      expect(protectedDots).toBeLessThan(originalDots);
      expect(protectedDots).toBe(1);
    });

    test("non-string input does not crash", () => {
      expect(() => protectDots(null)).not.toThrow();
      expect(() => protectDots(undefined)).not.toThrow();
      expect(() => protectDots(42)).not.toThrow();
    });
  });

  describe("pattern ordering correctness", () => {
    test("decimal does not get partially eaten by multi-level outline", () => {
      // Regression: multi-level "1.2.3" pattern was matching "234.56" as two-segment outline
      const { dictionary } = protectDots("Total: 1,234.56 dollars");
      const values = Object.values(dictionary);
      expect(values).toContain("1,234.56");
      expect(values).not.toContain("234.56");
    });

    test("acronym U.S.A. wins over multi-level eating U.S.A as outline", () => {
      // Regression: multi-level was matching "U.S.A" without trailing dot
      const { dictionary } = protectDots("She lives in the U.S.A. now");
      const values = Object.values(dictionary);
      expect(values).toContain("U.S.A.");
      expect(values).not.toContain("U.S.A");
    });

    test("J.K. — two-letter acronym not stolen by multi-level", () => {
      const { dictionary } = protectDots("J.K. Rowling");
      expect(Object.values(dictionary)).toContain("J.K.");
    });
  });

  describe("round-trip integrity", () => {
    test.each([
      "Dr. Smith found pi = 3.14. See Fig. 2 for details.",
      "1. First item\n2. Second item\n3. Third item",
      "Visit https://example.com or email user@test.org for info.",
      "Refer to section 1.2.3 and Appendix A.1.b.",
      "She visited the U.S.A. and saw the U.N. building.",
      "The values are .5, 1.5, and 1,234.56 respectively.",
      "Open file.txt or archive.tar.gz to begin.",
      "",
      "No dots here just words",
      "...",
      "Multiple sentences. Each ending. With a dot.",
    ])("restores exactly: %j", (input) => {
      expectRoundTrip(input);
    });
  });

  describe("token format", () => {
    test("tokens use NULL control character delimiters", () => {
      const { dictionary } = protectDots("Dr. Smith");
      const tokens = Object.keys(dictionary);
      expect(tokens.length).toBeGreaterThan(0);
      tokens.forEach((tok) => {
        expect(tok).toMatch(/^\x00TOK_\d+\x00$/);
      });
    });

    test("tokens are unique within a single call", () => {
      const { dictionary } = protectDots(
        "Dr. Smith and Dr. Jones met at 3.14 with Mr. Lee."
      );
      const tokens = Object.keys(dictionary);
      expect(new Set(tokens).size).toBe(tokens.length);
    });

    test("counter resets between separate calls", () => {
      const { dictionary: d1 } = protectDots("Dr. Smith");
      const { dictionary: d2 } = protectDots("Mr. Jones");
      expect(Object.keys(d1)).toContain("\x00TOK_0\x00");
      expect(Object.keys(d2)).toContain("\x00TOK_0\x00");
    });
  });

  describe("edge cases", () => {
    test("does not protect a lone period", () => {
      const input = "End of sentence.";
      const { protectedText, dictionary } = protectDots(input);
      expect(protectedText).toBe(input);
      expect(Object.keys(dictionary)).toHaveLength(0);
    });

    test("handles consecutive abbreviations", () => {
      const input = "Dr. Mr. Mrs. all attended.";
      const { protectedText, dictionary } = protectDots(input);
      expect(Object.values(dictionary)).toEqual(
        expect.arrayContaining(["Dr.", "Mr.", "Mrs."])
      );
      expectRoundTrip(input);
    });

    test("handles dot at very end of string", () => {
      expectRoundTrip("This ends with Dr.");
    });

    test("handles repeated identical patterns", () => {
      const input = "Dr. Smith and Dr. Jones and Dr. Lee.";
      const { dictionary } = protectDots(input);
      const drTokens = Object.entries(dictionary).filter(([, v]) => v === "Dr.");
      expect(drTokens.length).toBe(3);
      expectRoundTrip(input);
    });

    test("multiline input", () => {
      const input = [
        "1. Introduction",
        "Dr. Smith reports pi = 3.14.",
        "",
        "2. Methods",
        "See Fig. 1 and Eq. 2.",
      ].join("\n");
      expectRoundTrip(input);
    });
  });
});

describe("restore", () => {
  test("returns text unchanged when dictionary is empty", () => {
    expect(restore("hello world", {})).toBe("hello world");
  });

  test("replaces a single token", () => {
    const dict = { "\x00TOK_0\x00": "Dr." };
    expect(restore("\x00TOK_0\x00 Smith", dict)).toBe("Dr. Smith");
  });

  test("replaces multiple tokens", () => {
    const dict = {
      "\x00TOK_0\x00": "Dr.",
      "\x00TOK_1\x00": "3.14",
    };
    const input = "\x00TOK_0\x00 Smith found \x00TOK_1\x00.";
    expect(restore(input, dict)).toBe("Dr. Smith found 3.14.");
  });

  test("replaces all occurrences of the same token", () => {
    const dict = { "\x00TOK_0\x00": "Dr." };
    const input = "\x00TOK_0\x00 A and \x00TOK_0\x00 B";
    expect(restore(input, dict)).toBe("Dr. A and Dr. B");
  });

  test("leaves unknown tokens alone", () => {
    const dict = { "\x00TOK_0\x00": "Dr." };
    const input = "\x00TOK_0\x00 saw \x00TOK_99\x00.";
    expect(restore(input, dict)).toBe("Dr. saw \x00TOK_99\x00.");
  });
});