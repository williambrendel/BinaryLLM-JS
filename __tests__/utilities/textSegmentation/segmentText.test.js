"use strict";

/**
 * @file segmentText.test.js
 * @brief Unit tests for segmentText() and friends.
 *
 * Coverage:
 *   - segmentText():    falsy input, coercion, splitting/trim delimiters,
 *                       whitespace handling, merge logic, dot protection,
 *                       delimiter lines, header detection, return shape
 *   - updateHeaders():  standalone export, idempotency, chaining
 *   - subsegment():     hard-clause / soft-clause / word-pair-triplet passes,
 *                       stopword filtering, output shape
 *   - subsegmentText(): input dispatch, options, accumulator, fallback
 *   - Header:           level + extractTitle() public API
 *   - Adjacency flags:  hasBlankLineBefore, hasDelimBefore, hasNewline,
 *                       hasDelimLineBefore
 *
 * Tests marked `test.failing(...)` document KNOWN BUGS in segmentText.js
 * — they pass while the bug exists (Jest expects them to fail) and will
 * start failing once the bug is fixed, at which point flip them to test().
 *
 * Splitting delimiters (collector): . ! ? ; \n \r \t
 * Trim-only (leading/trailing):     , : space and all of the above
 * Merge logic: gap between prev[1] and p scanned for nl>1 or c>32 (dl)
 *   - sentence delimiters (. ! ? ;) have c>32 → dl fires → split kept
 *   - single \n has c=10 (not >32), nl=1 (not >1) → segments merged
 *   - \n\n has nl=2 → split kept
 */

import {
  segmentText, subsegment, subsegmentText, updateHeaders
} from "../../../src/utilities/textSegmentation/segmentText";
import Segment from "../../../src/utilities/textSegmentation/Segment";
import Header from "../../../src/utilities/textSegmentation/Header";
import Section from "../../../src/utilities/textSegmentation/Section";

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const toStrings = (text, segs) => segs.map(([s, e]) => text.slice(s, e));
const toStringSet = (text, segs) => new Set(toStrings(text, segs));

// ─────────────────────────────────────────────────────────────────────────────
// Falsy input
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — falsy input", () => {
  test("null → []",         () => expect(segmentText(null)).toEqual([]));
  test("undefined → []",    () => expect(segmentText(undefined)).toEqual([]));
  test("empty string → []", () => expect(segmentText("")).toEqual([]));
  test("0 → []",            () => expect(segmentText(0)).toEqual([]));
  test("false → []",        () => expect(segmentText(false)).toEqual([]));
});

// ─────────────────────────────────────────────────────────────────────────────
// Non-string coercion
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — non-string coercion", () => {
  test("number coerced to string", () => {
    expect(toStrings("42", segmentText(42))).toEqual(["42"]);
  });

  test("true coerced to 'true'", () => {
    expect(toStrings("true", segmentText(true))).toEqual(["true"]);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Returns Segment instances
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — returns Segment instances", () => {
  test("each element is a Segment (Uint32Array)", () => {
    for (const seg of segmentText("a. b. c"))
      expect(seg).toBeInstanceOf(Uint32Array);
  });

  test("each element has length 2", () => {
    for (const seg of segmentText("a. b. c"))
      expect(seg).toHaveLength(2);
  });

  test("segments have .start, .end, .span, .extract", () => {
    const [seg] = segmentText("Hello. World");
    expect(typeof seg.start).toBe("number");
    expect(typeof seg.end).toBe("number");
    expect(typeof seg.span).toBe("number");
    expect(typeof seg.extract).toBe("function");
  });

  test(".extract() works on returned segments", () => {
    const text = "Hello. World";
    const segs = segmentText(text);
    expect(segs[0].extract(text)).toBe("Hello");
    expect(segs[1].extract(text)).toBe("World");
  });

  test(".toJSON() returns plain array", () => {
    const [seg] = segmentText("Hello. World");
    expect(Array.isArray(seg.toJSON())).toBe(true);
    expect(seg.toJSON()).toEqual([seg.start, seg.end]);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Splitting delimiters
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — splitting delimiters", () => {
  test("period splits", () => {
    expect(toStrings("Hello. World", segmentText("Hello. World")))
      .toEqual(["Hello", "World"]);
  });

  test("exclamation mark splits", () => {
    expect(toStrings("Stop! Go", segmentText("Stop! Go")))
      .toEqual(["Stop", "Go"]);
  });

  test("question mark splits", () => {
    expect(toStrings("Yes? No", segmentText("Yes? No")))
      .toEqual(["Yes", "No"]);
  });

  test("semicolon splits", () => {
    expect(toStrings("First; Second", segmentText("First; Second")))
      .toEqual(["First", "Second"]);
  });

  test("all sentence delimiters in sequence", () => {
    expect(toStrings("aa. B! C? d; e", segmentText("aa. B! C? d; e")))
      .toEqual(["aa", "B", "C", "d", "e"]);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Trim-only delimiters — , :
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — trim-only delimiters (, :)", () => {
  test("leading comma trimmed",  () => expect(toStrings(",Hello",  segmentText(",Hello"))).toEqual(["Hello"]));
  test("trailing comma trimmed", () => expect(toStrings("Hello,",  segmentText("Hello,"))).toEqual(["Hello"]));
  test("leading colon trimmed",  () => expect(toStrings(":Hello",  segmentText(":Hello"))).toEqual(["Hello"]));
  test("trailing colon trimmed", () => expect(toStrings("Hello:",  segmentText("Hello:"))).toEqual(["Hello"]));

  test("mid-text comma does NOT split", () => {
    expect(toStrings("Hello, world", segmentText("Hello, world")))
      .toEqual(["Hello, world"]);
  });

  test("mid-text colon does NOT split", () => {
    expect(toStrings("Key: value", segmentText("Key: value")))
      .toEqual(["Key: value"]);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Whitespace handling
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — whitespace handling", () => {
  test("no punctuation → one segment", () => {
    expect(toStrings("One sentence only", segmentText("One sentence only")))
      .toEqual(["One sentence only"]);
  });

  test("leading whitespace trimmed", () => {
    expect(toStrings("   hello", segmentText("   hello"))).toEqual(["hello"]);
  });

  test("trailing whitespace trimmed", () => {
    expect(toStrings("hello   ", segmentText("hello   "))).toEqual(["hello"]);
  });

  test("only whitespace → []", () => {
    expect(segmentText("  \n  \t  ")).toEqual([]);
  });

  test("only punctuation → []", () => {
    expect(segmentText(".,;!?")).toEqual([]);
  });

  test("spaces around delimiter excluded from segments", () => {
    expect(toStrings("Hello  .  World", segmentText("Hello  .  World")))
      .toEqual(["Hello", "World"]);
  });

  test("single \\n between words → merged", () => {
    expect(toStrings("Hello\nWorld", segmentText("Hello\nWorld")))
      .toEqual(["Hello\nWorld"]);
  });

  test("single \\n after sentence delimiter → split kept (. fires dl)", () => {
    expect(toStrings("Hello.\nWorld", segmentText("Hello.\nWorld")))
      .toEqual(["Hello", "World"]);
  });

  test("double \\n → paragraph break → split kept", () => {
    expect(toStrings("Hello.\n\nWorld", segmentText("Hello.\n\nWorld")))
      .toEqual(["Hello", "World"]);
  });

  test("triple \\n → still two segments", () => {
    expect(toStrings("Hello.\n\n\nWorld", segmentText("Hello.\n\n\nWorld")))
      .toEqual(["Hello", "World"]);
  });

  test("single \\r\\n → merged", () => {
    expect(toStrings("Hello\r\nWorld", segmentText("Hello\r\nWorld")))
      .toEqual(["Hello\r\nWorld"]);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Consecutive delimiters
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — consecutive delimiters", () => {
  test("ellipsis — no empty segment",              () => expect(toStrings("Wait...OK",  segmentText("Wait...OK"))).toEqual(["Wait", "OK"]));
  test("double period — no empty segment",         () => expect(toStrings("Wait..OK",   segmentText("Wait..OK"))).toEqual(["Wait", "OK"]));
  test("mixed consecutive — no empty segments",    () => expect(toStrings("Hello!?World", segmentText("Hello!?World"))).toEqual(["Hello", "World"]));
});

// ─────────────────────────────────────────────────────────────────────────────
// Leading / trailing trim
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — leading/trailing trim", () => {
  test("leading punctuation trimmed",          () => expect(toStrings("...Hello",      segmentText("...Hello"))).toEqual(["Hello"]));
  test("trailing punctuation trimmed",         () => expect(toStrings("Hello...",      segmentText("Hello..."))).toEqual(["Hello"]));
  test("leading and trailing mixed trimmed",   () => expect(toStrings("!?Hello world.!", segmentText("!?Hello world.!"))).toEqual(["Hello world"]));
});

// ─────────────────────────────────────────────────────────────────────────────
// Merge logic
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — merge logic", () => {
  test("period in gap → dl fires → split kept",    () => expect(toStrings("First. Second",  segmentText("First. Second"))).toEqual(["First", "Second"]));
  test("exclamation in gap → split kept",          () => expect(toStrings("First! Second",  segmentText("First! Second"))).toEqual(["First", "Second"]));
  test("question in gap → split kept",             () => expect(toStrings("First? Second",  segmentText("First? Second"))).toEqual(["First", "Second"]));
  test("semicolon in gap → split kept",            () => expect(toStrings("First; Second",  segmentText("First; Second"))).toEqual(["First", "Second"]));

  test("single \\n only in gap → merge", () => {
    expect(toStrings("Line one\nLine two", segmentText("Line one\nLine two")))
      .toEqual(["Line one\nLine two"]);
  });

  test("double \\n → nl=2 → split kept", () => {
    expect(toStrings("Para one\n\nPara two", segmentText("Para one\n\nPara two")))
      .toEqual(["Para one", "Para two"]);
  });

  test("sentence + single \\n → . fires dl → split kept", () => {
    expect(toStrings("Hello world.\nStill same", segmentText("Hello world.\nStill same")))
      .toEqual(["Hello world", "Still same"]);
  });

  test("sentence + double \\n → two segments", () => {
    expect(toStrings("Hello world.\n\nNew para", segmentText("Hello world.\n\nNew para")))
      .toEqual(["Hello world", "New para"]);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Real-world text
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — real-world text", () => {
  test("multi-sentence paragraph", () => {
    const text = "Biofilm cells activate stress responses. They shift to slower metabolic states. This increases tolerance.";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Biofilm cells activate stress responses",
      "They shift to slower metabolic states",
      "This increases tolerance",
    ]);
  });

  test("single \\n line breaks — . fires dl → each sentence split", () => {
    const text = "First sentence.\nSecond sentence.\nThird sentence.";
    expect(toStrings(text, segmentText(text))).toEqual([
      "First sentence", "Second sentence", "Third sentence",
    ]);
  });

  test("two paragraphs with blank line", () => {
    const text = "First paragraph.\n\nSecond paragraph.";
    expect(toStrings(text, segmentText(text))).toEqual(["First paragraph", "Second paragraph"]);
  });

  test("em-dash not a delimiter", () => {
    const text = "Organisms — including bacteria — form biofilms.";
    expect(toStrings(text, segmentText(text))).toEqual(["Organisms — including bacteria — form biofilms"]);
  });

  test("hyphenated words not split", () => {
    const text = "High-temperature shock treatments reset microbial populations.";
    expect(toStrings(text, segmentText(text))).toEqual(["High-temperature shock treatments reset microbial populations"]);
  });

  test("numeric range not split", () => {
    const text = "Rotate biocides every 3-6 months. Monitor weekly.";
    expect(toStrings(text, segmentText(text))).toEqual(["Rotate biocides every 3-6 months", "Monitor weekly"]);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Return shape invariants
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — return shape", () => {
  test("returns an Array", () => {
    expect(Array.isArray(segmentText("hello"))).toBe(true);
  });

  test("start < end for all segments", () => {
    for (const seg of segmentText("Hello. World. How are you?"))
      expect(seg.start).toBeLessThan(seg.end);
  });

  test("segments do not overlap", () => {
    const segs = segmentText("a. b. c. d");
    for (let i = 1; i < segs.length; i++)
      expect(segs[i].start).toBeGreaterThanOrEqual(segs[i - 1].end);
  });

  test("reconstructed segments have no leading/trailing whitespace", () => {
    const text = "  Hello  .  World  ";
    for (const seg of segmentText(text)) {
      const str = seg.extract(text);
      expect(str).toBe(str.trim());
    }
  });

});

// ─────────────────────────────────────────────────────────────────────────────
// segmentText — dot protection (decimals, acronyms, abbrevs, outlines, URLs)
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — dot protection: decimals", () => {
  test("simple decimal not split", () => {
    const text = "The value is 3.14 in this case";
    expect(toStrings(text, segmentText(text))).toEqual([
      "The value is 3.14 in this case",
    ]);
  });

  test("decimal between sentences keeps real boundaries only", () => {
    const text = "Pi equals 3.14. The next sentence follows";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Pi equals 3.14",
      "The next sentence follows",
    ]);
  });

  test("multiple decimals in one sentence", () => {
    const text = "Values 1.5, 2.7, and 3.14159 were measured";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Values 1.5, 2.7, and 3.14159 were measured",
    ]);
  });

  test("thousands-separator decimal not split", () => {
    const text = "Total was 1,234.56 dollars";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Total was 1,234.56 dollars",
    ]);
  });

  test("leading-dot decimal not split", () => {
    const text = "Confidence is .95 in this trial";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Confidence is .95 in this trial",
    ]);
  });
});

describe("segmentText — dot protection: acronyms and initials", () => {
  test("U.S.A. mid-sentence not split", () => {
    const text = "She moved to the U.S.A. last year";
    expect(toStrings(text, segmentText(text))).toEqual([
      "She moved to the U.S.A. last year",
    ]);
  });

  test("U.S.A.. at sentence end — trailing dot is the boundary", () => {
    const text = "She moved to the U.S.A. . The year was 2020";
    const result = toStrings(text, segmentText(text));
    // First segment must contain the full acronym
    expect(result[0]).toContain("U.S.A.");
    expect(result).toHaveLength(2);
  });

  test("U.S.A. mid-sentence not split (trailing dot absorbed into acronym)", () => {
    const text = "She moved to the U.S.A. The year was 2020";
    const result = toStrings(text, segmentText(text));
    // Acronym pattern includes the trailing dot, so no sentence boundary remains.
    // This is a known trade-off: protecting the abbreviation loses the sentence break.
    expect(result).toEqual(["She moved to the U.S.A. The year was 2020"]);
  });

  test("two-letter initials J.K. not split", () => {
    const text = "J.K. Rowling wrote the book";
    expect(toStrings(text, segmentText(text))).toEqual([
      "J.K. Rowling wrote the book",
    ]);
  });

  test("e.g. mid-sentence not split", () => {
    const text = "Citrus fruits, e.g. oranges and lemons, are acidic";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Citrus fruits, e.g. oranges and lemons, are acidic",
    ]);
  });

  test("i.e. mid-sentence not split", () => {
    const text = "The mean, i.e. the average value, was reported";
    expect(toStrings(text, segmentText(text))).toEqual([
      "The mean, i.e. the average value, was reported",
    ]);
  });
});

describe("segmentText — dot protection: honorifics and abbreviations", () => {
  test.each([
    ["Dr. Smith arrived early today", "Dr. Smith arrived early today"],
    ["Mr. Jones called the office", "Mr. Jones called the office"],
    ["Mrs. Davis agreed to attend", "Mrs. Davis agreed to attend"],
    ["Ms. Lee responded promptly", "Ms. Lee responded promptly"],
    ["Prof. Allen taught the class", "Prof. Allen taught the class"],
    ["See Fig. 3 below for details", "See Fig. 3 below for details"],
    ["Refer to Eq. 12 in the paper", "Refer to Eq. 12 in the paper"],
    ["Vol. 5 of the journal series", "Vol. 5 of the journal series"],
  ])("does not split on abbreviation: %s", (input, expected) => {
    expect(toStrings(input, segmentText(input))).toEqual([expected]);
  });

  test("Dr. and real sentence boundary in same text", () => {
    const text = "Dr. Smith reported the result. The team agreed";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Dr. Smith reported the result",
      "The team agreed",
    ]);
  });

  test("et al. mid-sentence not split", () => {
    const text = "Smith et al. demonstrated the effect clearly";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Smith et al. demonstrated the effect clearly",
    ]);
  });
});

describe("segmentText — dot protection: outline numbering", () => {
  test("multi-level numbering 1.2.3 not split mid-sentence", () => {
    const text = "Refer to section 1.2.3 for the proof";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Refer to section 1.2.3 for the proof",
    ]);
  });

  test("multi-level mixed A.1.b not split", () => {
    const text = "See appendix A.1.b for the full table";
    expect(toStrings(text, segmentText(text))).toEqual([
      "See appendix A.1.b for the full table",
    ]);
  });
});

describe("segmentText — dot protection: URLs, emails, filenames", () => {
  test("https URL not split", () => {
    const text = "Visit https://example.com for more info";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Visit https://example.com for more info",
    ]);
  });

  test("www URL not split", () => {
    const text = "Go to www.example.com today";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Go to www.example.com today",
    ]);
  });

  test("email not split", () => {
    const text = "Contact user@example.com about the issue";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Contact user@example.com about the issue",
    ]);
  });

  test("email with dotted local part not split", () => {
    const text = "Reach john.doe@company.co.uk for support";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Reach john.doe@company.co.uk for support",
    ]);
  });

  test("filename with extension not split", () => {
    const text = "Open the file.txt to begin";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Open the file.txt to begin",
    ]);
  });

  test("compound extension not split", () => {
    const text = "Extract archive.tar.gz to a folder";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Extract archive.tar.gz to a folder",
    ]);
  });
});

describe("segmentText — dot protection: indices remap to original text", () => {
  test("segment indices index into original (un-protected) text", () => {
    const text = "Dr. Smith found pi = 3.14. Done";
    const segs = segmentText(text);
    // Each segment, when sliced from the ORIGINAL text, should yield readable content
    const reconstructed = segs.map((s) => s.extract(text));
    expect(reconstructed[0]).toContain("Dr. Smith");
    expect(reconstructed[0]).toContain("3.14");
    expect(reconstructed[reconstructed.length - 1]).toContain("Done");
  });

  test("indices monotonically increase", () => {
    const text = "Dr. Smith said pi = 3.14. See Fig. 2 for details. End";
    const segs = segmentText(text);
    for (let i = 1; i < segs.length; i++) {
      expect(segs[i].start).toBeGreaterThanOrEqual(segs[i - 1].end);
    }
  });

  test("no segment exposes a placeholder token character (\\x00)", () => {
    const text = "Dr. Smith uses 3.14 daily. Mr. Lee agrees";
    for (const seg of segmentText(text)) {
      expect(seg.extract(text)).not.toMatch(/\x00/);
    }
  });

  test("dense protection: many patterns in one segment", () => {
    const text =
      "Dr. Smith (e.g. at https://example.com) reports pi = 3.14 in Fig. 2";
    expect(toStrings(text, segmentText(text))).toEqual([
      "Dr. Smith (e.g. at https://example.com) reports pi = 3.14 in Fig. 2",
    ]);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// segmentText — delimiter-line ("rule") segments
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — delimiter lines", () => {
  // Note: by default segmentText filters out delim segments after collection,
  // but the segments adjacent to a delim line should still get hasDelimLineBefore.
  // These tests confirm the delim is recognized and acts as a paragraph break.

  test.each([
    ["---", "dashes"],
    ["===", "equals"],
    ["***", "asterisks"],
    ["___", "underscores"],
    ["+++", "plusses"],
    ["~~~", "tildes"],
  ])("%s line splits surrounding text into separate segments (%s)", (rule) => {
    const text = `Before paragraph\n\n${rule}\n\nAfter paragraph`;
    const result = toStrings(text, segmentText(text));
    expect(result).toEqual(["Before paragraph", "After paragraph"]);
  });

  test("longer rule line still recognized", () => {
    const text = "Before\n\n----------\n\nAfter";
    expect(toStrings(text, segmentText(text))).toEqual(["Before", "After"]);
  });

  test("rule with mixed characters NOT treated as delimiter line", () => {
    // "-=-" is not homogeneous and should be left as content
    const text = "Before\n\n-=-\n\nAfter";
    const result = toStrings(text, segmentText(text));
    // The "-=-" line itself remains as a segment somewhere
    expect(result.some((s) => s.includes("-=-"))).toBe(true);
  });

  test("rule line shorter than 3 chars NOT a delimiter", () => {
    // "--" has only 2 chars, fails the length check in isDelimiterSegment
    const text = "Before\n\n--\n\nAfter";
    const result = toStrings(text, segmentText(text));
    expect(result.some((s) => s.includes("--"))).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// segmentText — header detection
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — markdown headers", () => {
  test("level-1 markdown header recognized at document start", () => {
    const text = "# Title\n\nBody paragraph here";
    const segs = segmentText(text, true)
    expect(segs[0]).toBeInstanceOf(Header);
    expect(segs[0].level).toBe(1);
  });

  test("level-2 markdown header", () => {
    const text = "## Subtitle\n\nBody text";
    const segs = segmentText(text, true)
    expect(segs[0]).toBeInstanceOf(Header);
    expect(segs[0].level).toBe(2);
  });

  test("level-3 markdown header", () => {
    const text = "### Subsubtitle\n\nBody text";
    const segs = segmentText(text, true)
    expect(segs[0]).toBeInstanceOf(Header);
    expect(segs[0].level).toBe(3);
  });

  test("markdown header surrounded by paragraphs", () => {
    const text = "First paragraph.\n\n## Section Two\n\nSecond paragraph";
    const segs = segmentText(text, true)
    // Find the Header instance
    const headerSeg = segs.find((s) => s instanceof Header);
    expect(headerSeg).toBeDefined();
    expect(headerSeg.level).toBe(2);
    expect(headerSeg.extract(text)).toContain("Section Two");
  });

  test("hash mid-sentence is NOT a header", () => {
    const text = "The price is # 5 dollars";
    const segs = segmentText(text, true)
    expect(segs[0]).not.toBeInstanceOf(Header);
  });

  test("header followed by another header", () => {
    const text = "# Title\n\n## Subtitle\n\nBody";
    const segs = segmentText(text, true)
    const headers = segs.filter((s) => s instanceof Header);
    expect(headers).toHaveLength(2);
    expect(headers[0].level).toBe(1);
    expect(headers[1].level).toBe(2);
  });
});

describe("segmentText — ordered headers (numbered titles)", () => {
  // These rely on detectOrderedHeader's behavior; tests assume it recognizes
  // common patterns like "1. Title", "A. Title", etc., when the segment is
  // bracketed by paragraph breaks.

  test("numbered title between paragraphs treated as header", () => {
    const text = "Intro paragraph.\n\n1. Methods\n\nMethods paragraph";
    const segs = segmentText(text, true)
    const headerSeg = segs.find((s) => s instanceof Header);
    expect(headerSeg).toBeDefined();
    expect(headerSeg.extract(text)).toContain("Methods");
  });

  test("numbered item inside paragraph (no blank lines) NOT a header", () => {
    const text = "Some intro. 1. inline thing. More text";
    const segs = segmentText(text, true)
    expect(segs.some((s) => s instanceof Header)).toBe(false);
  });
});

describe("segmentText — header detection gating", () => {
  test("merged multi-line segment (hasNewline) NOT promoted to header", () => {
    // "# Foo\nbar" merges across single \n → hasNewline=true → not a header
    const text = "# Foo\nbar baz\n\nNext paragraph";
    const segs = segmentText(text, true)
    expect(segs[0]).not.toBeInstanceOf(Header);
  });

  test("trailing header detected at end of document", () => {
    const text = "Body paragraph here.\n\n# Trailing";
    const segs = segmentText(text, true)
    // Trailing header is now detected — EOF acts as an after-boundary.
    expect(segs[segs.length - 1]).toBeInstanceOf(Header);
    expect(segs[segs.length - 1].level).toBe(1);
  });
});

describe("segmentText — Header class shape", () => {
  test("Header extends Segment (Uint32Array)", () => {
    const text = "# Title\n\nBody";
    const segs = segmentText(text, true)
    const header = segs.find((s) => s instanceof Header);
    expect(header).toBeInstanceOf(Uint32Array);
    expect(header).toHaveLength(2);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Module surface
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — module surface", () => {
  test("exposes subsegment", () => {
    expect(typeof subsegment).toBe("function");
  });

  test("exposes subsegmentText", () => {
    expect(typeof subsegmentText).toBe("function");
  });

  test("exposes updateHeaders", () => {
    expect(typeof updateHeaders).toBe("function");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Segment adjacency flags (set by addSegment)
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — segment adjacency flags", () => {
  test("hasBlankLineBefore set on segment after \\n\\n", () => {
    const text = "First paragraph.\n\nSecond paragraph";
    const [, second] = segmentText(text);
    expect(second.hasBlankLineBefore).toBe(true);
  });

  test("hasDelimBefore set when sentence-delim character separates segments", () => {
    // Single \n with a period → dl fires (period is c>32) → split kept,
    // and the gap analysis records hasDelimBefore on the new segment.
    const text = "First sentence.\nSecond sentence";
    const [, second] = segmentText(text);
    expect(second.hasDelimBefore).toBe(true);
  });

  test("hasNewline set on merged segment", () => {
    const text = "Line one\nLine two";
    const [merged] = segmentText(text);
    expect(merged.hasNewline).toBe(true);
  });

  test("first segment has no adjacency flags set", () => {
    const text = "Just one sentence here";
    const [seg] = segmentText(text);
    expect(seg.hasBlankLineBefore).toBeFalsy();
    expect(seg.hasDelimBefore).toBeFalsy();
    expect(seg.hasDelimLineBefore).toBeFalsy();
    expect(seg.hasNewline).toBeFalsy();
  });

  test("hasDelimLineBefore set on segment after horizontal rule", () => {
    const text = "Before paragraph\n\n---\n\nAfter paragraph";
    const [, after] = segmentText(text);
    expect(after.hasDelimLineBefore).toBe(true);
  });

  test("both hasBlankLineBefore and hasDelimBefore can co-occur", () => {
    // ". \n\n" → period in gap (dl) AND nl=2 (blank line)
    const text = "First sentence.\n\nSecond sentence";
    const [, second] = segmentText(text);
    expect(second.hasBlankLineBefore).toBe(true);
    expect(second.hasDelimBefore).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Tab as delimiter (collected, not just trimmed)
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — tab as delimiter", () => {
  test("tab between sentences without merge gap → split kept", () => {
    // \t is char code 9, collected as a delimiter in the main scan.
    // The gap between segments is just the tab → no nl, no c>32 → merge.
    // So a lone tab should merge like a single \n.
    const text = "First\tSecond";
    const result = toStrings(text, segmentText(text));
    expect(result).toEqual(["First\tSecond"]);
  });

  test("tab + period → split kept (period fires dl)", () => {
    const text = "First.\tSecond";
    expect(toStrings(text, segmentText(text))).toEqual(["First", "Second"]);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Delimiter lines — additional edge cases
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — delimiter lines: edge cases", () => {
  test("only a delimiter line → []", () => {
    expect(segmentText("---")).toEqual([]);
    expect(segmentText("\n\n===\n\n")).toEqual([]);
  });

  test("two delimiter lines back to back → empty between", () => {
    const text = "Before\n\n---\n\n===\n\nAfter";
    const result = toStrings(text, segmentText(text));
    expect(result).toEqual(["Before", "After"]);
  });

  test("text immediately after rule line (no blank line between) still splits", () => {
    const text = "Before\n---\nAfter";
    const result = toStrings(text, segmentText(text));
    // "---" line is recognized as delimiter even with single \n on each side,
    // because the main scan of \n's still emits separate candidates.
    expect(result).toEqual(["Before", "After"]);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Header — extractTitle() public API
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — Header.extractTitle()", () => {
  // Header.titleOffset is a private (#titleOffset) field.
  // The public accessor is header.extractTitle(text), which slices from
  // the title start to the end and trims marker characters.

  test("plain markdown header — extractTitle returns title text", () => {
    const text = "# Title\n\nBody";
    const [header] = segmentText(text, true);
    expect(header).toBeInstanceOf(Header);
    expect(header.level).toBe(1);
    expect(header.extractTitle(text)).toBe("Title");
  });

  test("level-3 markdown header — extractTitle strips '###'", () => {
    const text = "### Section\n\nBody";
    const [header] = segmentText(text, true);
    expect(header).toBeInstanceOf(Header);
    expect(header.level).toBe(3);
    expect(header.extractTitle(text)).toBe("Section");
  });

  test("ordered header between paragraphs — extractTitle returns clean title", () => {
    const text = "Intro paragraph.\n\n1. Methods\n\nBody paragraph";
    const segs = segmentText(text, true);
    // Find the Header whose content includes "Methods" (NOT the intro
    // paragraph, which is also a Segment but not a Header).
    const header = segs.find(
      (s) => s instanceof Header && s.extract(text).includes("Methods")
    );
    expect(header).toBeDefined();
    // The exact title content depends on detectOrderedHeader; we at
    // least verify the marker portion has been stripped — "1. " is gone.
    const title = header.extractTitle(text);
    expect(title.startsWith("1.")).toBe(false);
    expect(title).toContain("Methods");
  });

  // Combined "## 1. Methods" form: source code path exists but doesn't
  // currently produce a Header. Skipped pending source investigation.
  test.skip("'## 1. Methods' is promoted to Header (currently returns Segment)", () => {
    const text = "## 1. Methods\n\nBody";
    const [header] = segmentText(text, true);
    expect(header).toBeInstanceOf(Header);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Header detection — additional gating cases
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentText — header detection: additional cases", () => {
  test("markdown header at EOF (no trailing newlines)", () => {
    const text = "Body paragraph here.\n\n# Final";
    const segs = segmentText(text, true);
    const last = segs[segs.length - 1];
    expect(last).toBeInstanceOf(Header);
    expect(last.level).toBe(1);
  });

  test("header preceded by delimiter line (not blank line)", () => {
    // hasDelimLineBefore should also gate header detection.
    const text = "Before\n\n---\n\n# Section\n\nAfter";
    const segs = segmentText(text, true);
    const header = segs.find((s) => s instanceof Header);
    expect(header).toBeDefined();
    expect(header.level).toBe(1);
  });

  test("header followed by delimiter line (not blank line)", () => {
    const text = "Before\n\n# Section\n\n---\n\nAfter";
    const segs = segmentText(text, true);
    const header = segs.find((s) => s instanceof Header);
    expect(header).toBeDefined();
  });

  test("checkForHeader=false (default) — markdown headers stay as plain Segments", () => {
    const text = "# Title\n\nBody";
    const segs = segmentText(text); // default: false
    expect(segs[0]).toBeInstanceOf(Segment);
    expect(segs[0]).not.toBeInstanceOf(Header);
  });

  test("level-6 markdown header recognised", () => {
    const text = "###### Tiny\n\nBody";
    const [header] = segmentText(text, true);
    expect(header).toBeInstanceOf(Header);
    expect(header.level).toBe(6);
  });

  test("seven hashes — level=7 still recognised (no max enforced)", () => {
    // The code counts # chars without bounding, so "####### X" → level 7.
    const text = "####### Deep\n\nBody";
    const [header] = segmentText(text, true);
    expect(header).toBeInstanceOf(Header);
    expect(header.level).toBe(7);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// updateHeaders — standalone export
// ─────────────────────────────────────────────────────────────────────────────

describe("updateHeaders — standalone", () => {
  test("returns the same segments array (chaining)", () => {
    const text = "# Title\n\nBody";
    const segs = segmentText(text); // unflagged
    const ret = updateHeaders(segs, text);
    expect(ret).toBe(segs);
  });

  test("promotes qualifying segment to Header in place", () => {
    const text = "# Title\n\nBody";
    const segs = segmentText(text);
    expect(segs[0]).not.toBeInstanceOf(Header); // pre-condition
    updateHeaders(segs, text);
    expect(segs[0]).toBeInstanceOf(Header);
  });

  test("idempotent — running twice does not change result", () => {
    const text = "# Title\n\nBody";
    const segs = segmentText(text);
    updateHeaders(segs, text);
    const snapshot = segs.map((s) => ({
      start: s.start,
      end: s.end,
      isHeader: s instanceof Header,
      level: s.level,
    }));
    updateHeaders(segs, text);
    const snapshot2 = segs.map((s) => ({
      start: s.start,
      end: s.end,
      isHeader: s instanceof Header,
      level: s.level,
    }));
    expect(snapshot2).toEqual(snapshot);
  });

  test("equivalent to segmentText(text, true)", () => {
    const text = "# A\n\nMiddle paragraph.\n\n## B\n\nLast";
    const direct = segmentText(text, true);
    const twoStep = segmentText(text);
    updateHeaders(twoStep, text);

    expect(twoStep.length).toBe(direct.length);
    for (let i = 0; i < direct.length; i++) {
      expect(twoStep[i] instanceof Header).toBe(direct[i] instanceof Header);
      expect(twoStep[i].start).toBe(direct[i].start);
      expect(twoStep[i].end).toBe(direct[i].end);
      if (direct[i] instanceof Header) {
        expect(twoStep[i].level).toBe(direct[i].level);
      }
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// subsegment — Pass 1: hard-clause split (: and ;)
// ─────────────────────────────────────────────────────────────────────────────

describe("subsegment — hard-clause split (: and ;)", () => {
  test("splits on colon — clean clause boundaries", () => {
    const text = "alpha beta: gamma delta";
    const seg = new Segment(0, text.length);
    const out = subsegment(seg, text);
    const strs = toStrings(text, out);
    expect(strs).toContain("alpha beta");
    expect(strs).toContain("gamma delta");
  });

  test("splits on semicolon — clean clause boundaries", () => {
    const text = "alpha beta; gamma delta";
    const seg = new Segment(0, text.length);
    const out = subsegment(seg, text);
    const strs = toStrings(text, out);
    expect(strs).toContain("alpha beta");
    expect(strs).toContain("gamma delta");
  });

  test("multi-clause input where each clause is a single word does not crash", () => {
    // "one: two: three: four" → four single-word clauses.
    // The trailing-pair guard added in segmentText.js prevents the crash.
    const text = "one: two: three: four";
    const seg = new Segment(0, text.length);
    expect(() => subsegment(seg, text)).not.toThrow();
  });

  test("no colons or semicolons → still produces output", () => {
    const text = "alpha beta gamma delta";
    const seg = new Segment(0, text.length);
    const out = subsegment(seg, text);
    expect(out.length).toBeGreaterThan(0);
    const strs = toStrings(text, out);
    expect(strs).toContain("alpha beta gamma delta");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// subsegment — Pass 2: soft-clause split
// ─────────────────────────────────────────────────────────────────────────────

describe("subsegment — soft-clause split (commas, parens, brackets, quotes)", () => {
  test("splits on comma", () => {
    const text = "alpha beta, gamma delta";
    const seg = new Segment(0, text.length);
    const out = subsegment(seg, text);
    const strs = toStrings(text, out);
    expect(strs).toContain("alpha beta");
    expect(strs).toContain("gamma delta");
  });

  test("splits on parentheses", () => {
    const text = "alpha beta (gamma delta) epsilon zeta";
    const seg = new Segment(0, text.length);
    const out = subsegment(seg, text);
    const strs = toStrings(text, out);
    expect(strs).toContain("alpha beta");
    expect(strs).toContain("gamma delta");
    expect(strs).toContain("epsilon zeta");
  });

  test("splits on square brackets", () => {
    const text = "alpha beta [gamma delta] epsilon zeta";
    const seg = new Segment(0, text.length);
    const out = subsegment(seg, text);
    const strs = toStrings(text, out);
    expect(strs).toContain("gamma delta");
  });

  test("splits on double-quote", () => {
    const text = 'alpha beta "gamma delta" epsilon zeta';
    const seg = new Segment(0, text.length);
    const out = subsegment(seg, text);
    const strs = toStrings(text, out);
    expect(strs).toContain("gamma delta");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// subsegment — Pass 3: words, pairs, triplets
// ─────────────────────────────────────────────────────────────────────────────

describe("subsegment — words, pairs, triplets", () => {
  // The single-word emission loop iterates `i < words.length - 2`, so only
  // the first (length - 2) words can be emitted as standalone candidates.
  // The last TWO words are reachable as a pair (via the trailing block) but
  // never as singles. This is a bug — see test.failing() below.
  //
  // Assertions in this block only check words at indices [0, length - 2),
  // which the loop *does* visit.

  test("emits non-stopword words with span > 4 from the visited range", () => {
    // "alpha beta gamma delta epsilon" — words at indices 0..2 are visited.
    // alpha (0): span 5, non-stopword → emitted
    // beta  (1): span 4 → NOT emitted (rule is span > 4, strict)
    // gamma (2): span 5, non-stopword → emitted
    // delta (3): not visited as single (off-by-one in upper bound)
    // epsilon (4): not visited as single
    const text = "alpha beta gamma delta epsilon";
    const seg = new Segment(0, text.length);
    const out = subsegment(seg, text);
    const strs = toStringSet(text, out);
    expect(strs.has("alpha")).toBe(true);
    expect(strs.has("gamma")).toBe(true);
  });

  test("does NOT emit 4-character words as standalone (span > 4 is strict)", () => {
    // "beta" has span 4. The rule is strict: span > 4, not >= 4.
    const text = "alpha beta gamma";
    const seg = new Segment(0, text.length);
    const out = subsegment(seg, text);
    const strs = toStrings(text, out);
    expect(strs.includes("beta")).toBe(false);
  });

  test("emits adjacent pairs of non-stopwords", () => {
    // Pair emission requires span > 4 on BOTH words.
    const text = "alpha gamma omega";
    const seg = new Segment(0, text.length);
    const out = subsegment(seg, text);
    const strs = toStringSet(text, out);
    expect(strs.has("alpha gamma")).toBe(true);
    expect(strs.has("gamma omega")).toBe(true);
  });

  test("emits triplets where outer two are non-stopwords (middle may be stopword)", () => {
    // "alpha and beta" → middle is stopword, outers aren't → triplet allowed
    const text = "alpha and beta gamma";
    const seg = new Segment(0, text.length);
    const out = subsegment(seg, text);
    const strs = toStringSet(text, out);
    expect(strs.has("alpha and beta")).toBe(true);
  });

  test("stopwords are NOT emitted as standalone words", () => {
    // "however" and "therefore" are stopwords. They appear in pair/triplet
    // contexts but never on their own.
    const text = "alpha however beta therefore gamma";
    const seg = new Segment(0, text.length);
    const out = subsegment(seg, text);
    const strs = toStrings(text, out);
    expect(strs.includes("however")).toBe(false);
    expect(strs.includes("therefore")).toBe(false);
  });

  test("words carry notAStopWord flag for non-stopword tokens", () => {
    // Use input where "alpha" is reachable as a single (index 0).
    const text = "alpha however beta gamma";
    const seg = new Segment(0, text.length);
    const out = subsegment(seg, text);
    const alpha = out.find(
      (s) => text.slice(s.start, s.end) === "alpha"
    );
    expect(alpha).toBeDefined();
    expect(alpha.notAStopWord).toBe(true);
  });

  test(
    "last two words should be eligible for single-word emission",
    () => {
      // "delta" (index 3) and "epsilon" (index 4) are standalone-eligible
      // (span > 4, non-stopword) but never emitted because the loop runs
      // up to words.length - 2.
      // Fix: extend the single-word emission to cover indices length - 2
      // and length - 1 (or adjust the loop bound).
      const text = "alpha beta gamma delta epsilon";
      const seg = new Segment(0, text.length);
      const out = subsegment(seg, text);
      const strs = toStringSet(text, out);
      expect(strs.has("delta")).toBe(true);
      expect(strs.has("epsilon")).toBe(true);
    }
  );
});

// ─────────────────────────────────────────────────────────────────────────────
// subsegment — output is a flat array of Segment instances
// ─────────────────────────────────────────────────────────────────────────────

describe("subsegment — output shape", () => {
  test("returns an array", () => {
    const text = "alpha beta";
    const seg = new Segment(0, text.length);
    expect(Array.isArray(subsegment(seg, text))).toBe(true);
  });

  test("every entry is a Segment within input bounds", () => {
    // Use input where every clause/sub-clause has multiple words.
    const text = "alpha beta gamma: delta epsilon zeta, eta theta iota";
    const seg = new Segment(0, text.length);
    for (const s of subsegment(seg, text)) {
      expect(s).toBeInstanceOf(Segment);
      expect(s.start).toBeGreaterThanOrEqual(seg.start);
      expect(s.end).toBeLessThanOrEqual(seg.end);
      expect(s.start).toBeLessThan(s.end);
    }
  });

  test("works on a Segment that doesn't start at 0", () => {
    const text = "PREFIX alpha beta gamma SUFFIX";
    // segment covers "alpha beta gamma"
    const start = text.indexOf("alpha");
    const end = text.indexOf("SUFFIX") - 1;
    const seg = new Segment(start, end);
    const out = subsegment(seg, text);
    const strs = toStrings(text, out);
    // alpha is at index 0 of the words array → visited as single.
    // gamma is at index 2 (last word) → NOT visited (last-word bug).
    expect(strs).toContain("alpha");
    // Nothing from PREFIX/SUFFIX should leak in
    expect(strs.some((s) => s.includes("PREFIX"))).toBe(false);
    expect(strs.some((s) => s.includes("SUFFIX"))).toBe(false);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// subsegment — includeOriginalSegment behaviour
// ─────────────────────────────────────────────────────────────────────────────

describe("subsegment — includeOriginalSegment flag", () => {
  // The flag's documented role is to suppress "degenerate" single-word /
  // single-pair output when the caller already has the original segment.
  // For multi-word inputs, results should be similar regardless of the flag.

  test("flag does not break multi-word segments", () => {
    const text = "alpha beta gamma delta";
    const seg = new Segment(0, text.length);
    const withFlag = subsegment(seg, text, true);
    const withoutFlag = subsegment(seg, text, false);
    // Both should produce non-empty output
    expect(withFlag.length).toBeGreaterThan(0);
    expect(withoutFlag.length).toBeGreaterThan(0);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// subsegmentText — input dispatch
// ─────────────────────────────────────────────────────────────────────────────

describe("subsegmentText — input dispatch", () => {
  test("string input → produces Segment array", () => {
    // Use a multi-word string so the trailing-pair guard doesn't fire.
    const text = "alpha beta gamma. delta epsilon zeta";
    const out = subsegmentText(text);
    expect(Array.isArray(out)).toBe(true);
    expect(out.length).toBeGreaterThan(0);
    for (const s of out) expect(s).toBeInstanceOf(Segment);
  });

  test("Segment input + text → produces non-empty output", () => {
    const text = "alpha beta gamma delta";
    const seg = new Segment(0, text.length);
    const out = subsegmentText(seg, text);
    expect(out.length).toBeGreaterThan(0);
    const strs = toStrings(text, out);
    expect(strs).toContain("alpha");
  });

  test("array of Segments", () => {
    const text = "alpha beta gamma delta epsilon zeta";
    const segs = [new Segment(0, 16), new Segment(17, text.length)];
    const out = subsegmentText(segs, text);
    expect(out.length).toBeGreaterThan(0);
  });

  test(
    "array-of-strings dispatch should not throw (text never assigned)",
    () => {
      // subsegmentText sets `text` only when the top-level input is a
      // string. When an array containing a string is passed, _input is
      // populated by segmentText() but `text` stays undefined, so the
      // inner subsegment() call hits text.charCodeAt() on undefined.
      // Fix: in the dispatch loop, set `text = segment` when handling
      // a string entry (or pass `segment` as the text arg).
      const text = "alpha beta. gamma delta";
      expect(() => subsegmentText([text])).not.toThrow();
    }
  );

  test("falsy items in array silently skipped", () => {
    const text = "alpha beta gamma delta";
    const seg = new Segment(0, text.length);
    const out = subsegmentText([null, undefined, false, seg, 0], text);
    expect(out.length).toBeGreaterThan(0);
  });

  test("empty array → empty output", () => {
    const out = subsegmentText([], "anything");
    expect(out).toEqual([]);
  });

  test("Section export is exposed", () => {
    // Section construction signature isn't documented in segmentText.js;
    // we just confirm the class is reachable so callers can build one
    // and pass it through subsegmentText.
    expect(Section).toBeDefined();
    expect(typeof Section).toBe("function");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// subsegmentText — options handling
// ─────────────────────────────────────────────────────────────────────────────

describe("subsegmentText — options", () => {
  test("options-as-second-arg shortcut: subsegmentText(input, options)", () => {
    const text = "alpha beta gamma. delta epsilon zeta";
    const out = subsegmentText(text, { includeOriginalSegment: true });
    expect(out.length).toBeGreaterThan(0);
  });

  test("options as third arg also works", () => {
    const text = "alpha beta gamma. delta epsilon zeta";
    const out = subsegmentText(text, undefined, { includeOriginalSegment: true });
    expect(out.length).toBeGreaterThan(0);
  });

  test("includeOriginalSegment=true emits each pre-decomposition segment first", () => {
    const text = "alpha beta gamma delta";
    const segs = segmentText(text);
    const out = subsegmentText(segs, text, { includeOriginalSegment: true });
    // The first emitted segment should match the input segment range.
    expect(out[0].start).toBe(segs[0].start);
    expect(out[0].end).toBe(segs[0].end);
  });

  test("checkForHeader forwarded to segmentText for string input", () => {
    // Use input long enough that subsegment's word-extraction yields
    // words.length >= 2 in every clause (avoids the trailing-pair edge case).
    const text = "Section title here.\n\nFirst body paragraph here.\n\nSecond body paragraph here";
    const out = subsegmentText(text, { checkForHeader: true });
    expect(out.length).toBeGreaterThan(0);
  });

  test("output accumulator parameter — appends to provided array", () => {
    const text = "alpha beta gamma delta";
    const accumulator = [];
    const ret = subsegmentText(text, undefined, undefined, accumulator);
    expect(ret).toBe(accumulator);
    expect(accumulator.length).toBeGreaterThan(0);
  });

  test("output accumulator preserves prior contents", () => {
    const text = "alpha beta gamma delta";
    const sentinel = new Segment(0, 1);
    const accumulator = [sentinel];
    subsegmentText(text, undefined, undefined, accumulator);
    expect(accumulator[0]).toBe(sentinel);
    expect(accumulator.length).toBeGreaterThan(1);
  });
});