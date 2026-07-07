"use strict";

/**
 * @file segmentTextSections.test.js
 * @brief Unit tests for segmentTextSections().
 *
 * Returns a Section tree where:
 *  - Header sections (Section.header is a Header) hold body content and
 *    can nest deeper Sections.
 *  - Paragraph sections (Section.header is undefined) hold body Segments;
 *    created on every blank-line break.
 *  - The root is itself a Section. When the document has exactly one
 *    top-level child, that child is returned directly (single-child unwrap).
 *  - Empty input returns an empty Section (length 0).
 */

import segmentTextSections from "../../../src/utilities/textSegmentation/segmentTextSections";
import segmentText        from "../../../src/utilities/textSegmentation/segmentText";
import Segment            from "../../../src/utilities/textSegmentation/Segment";
import Section            from "../../../src/utilities/textSegmentation/Section";
import Header             from "../../../src/utilities/textSegmentation/Header";

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const toStrings = (text, segs) => segs.map((s) => text.slice(s[0], s[1]));

/** True if a Section is a header section (has a Header as first child). */
const isHeaderSection = (s) => s instanceof Section && s.header !== undefined;

/** True if a Section is a paragraph section (no header). */
const isParagraphSection = (s) => s instanceof Section && s.header === undefined;

/** Collect all body Segments (no Header, no Section) from a subtree. */
const allBodySegments = (root) =>
  root.flatten().filter((x) => x instanceof Segment && !(x instanceof Header) && !(x instanceof Section));

/** Collect all header sections from a subtree (excluding paragraph sections). */
const allHeaderSections = (root) =>
  root.flatten().filter(isHeaderSection);

// ─────────────────────────────────────────────────────────────────────────────
// Falsy input
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentTextSections — falsy input", () => {
  test("null → empty Section", () => {
    const r = segmentTextSections(null);
    expect(r).toBeInstanceOf(Section);
    expect(r).toHaveLength(0);
  });

  test("undefined → empty Section", () => {
    const r = segmentTextSections(undefined);
    expect(r).toBeInstanceOf(Section);
    expect(r).toHaveLength(0);
  });

  test("empty string → empty Section", () => {
    const r = segmentTextSections("");
    expect(r).toBeInstanceOf(Section);
    expect(r).toHaveLength(0);
  });

  test("only delimiters → empty Section", () => {
    const r = segmentTextSections(".,!?");
    expect(r).toBeInstanceOf(Section);
    expect(r).toHaveLength(0);
  });

  test("only whitespace → empty Section", () => {
    const r = segmentTextSections("  \n  ");
    expect(r).toBeInstanceOf(Section);
    expect(r).toHaveLength(0);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Result is always a Section
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentTextSections — return type", () => {
  test("single-section input returns a Section (unwrapped)", () => {
    const r = segmentTextSections("Hello. World.");
    expect(r).toBeInstanceOf(Section);
    // Single-child unwrap: r is the paragraph section itself, not a wrapper
    expect(isParagraphSection(r)).toBe(true);
  });

  test("multi-section input returns a Section (root wrapper)", () => {
    const r = segmentTextSections("Hello.\n\nWorld.");
    expect(r).toBeInstanceOf(Section);
    // Root has multiple paragraph children
    expect(r.length).toBeGreaterThan(1);
  });

  test("constructor.name is Section", () => {
    const r = segmentTextSections("Hello. World.");
    expect(r.constructor.name).toBe("Section");
  });

  test(".start and .end are numeric", () => {
    const r = segmentTextSections("Hello. World.");
    expect(typeof r.start).toBe("number");
    expect(typeof r.end).toBe("number");
  });

  test(".extract() returns text spanning the result", () => {
    const text = "Hello.\n\nHow are you?";
    const r = segmentTextSections(text);
    expect(r.extract(text)).toBe(text.slice(r.start, r.end));
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Header-free input — single section
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentTextSections — header-free single section (unwrapped)", () => {
  test("single word → unwrapped paragraph with one segment", () => {
    const text = "Hello";
    const r = segmentTextSections(text);
    expect(isParagraphSection(r)).toBe(true);
    expect(r).toHaveLength(1);
    expect(r[0].extract(text)).toBe("Hello");
  });

  test("multi-sentence, no blank line → one paragraph with multiple segments", () => {
    const text = "Hello. World. How are you?";
    const r = segmentTextSections(text);
    expect(isParagraphSection(r)).toBe(true);
    expect(toStrings(text, r)).toEqual(["Hello", "World", "How are you"]);
  });

  test("single \\n between words → merged → one paragraph, one segment", () => {
    const r = segmentTextSections("Hello\nWorld");
    expect(isParagraphSection(r)).toBe(true);
    expect(r).toHaveLength(1);
  });

  test("CRLF single line ending → same paragraph", () => {
    const r = segmentTextSections("Hello.\r\nWorld");
    expect(isParagraphSection(r)).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Header-free input — multiple sections
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentTextSections — header-free multiple sections (root wrapped)", () => {
  test("two paragraphs separated by \\n\\n", () => {
    const text = "Hello. World.\n\nHow are you?";
    const r = segmentTextSections(text);
    expect(r.length).toBe(2);
    expect(toStrings(text, r[0])).toEqual(["Hello", "World"]);
    expect(toStrings(text, r[1])).toEqual(["How are you"]);
  });

  test("three paragraphs", () => {
    const text = "First.\n\nSecond.\n\nThird";
    const r = segmentTextSections(text);
    expect(r.length).toBe(3);
    expect(toStrings(text, r[0])).toEqual(["First"]);
    expect(toStrings(text, r[1])).toEqual(["Second"]);
    expect(toStrings(text, r[2])).toEqual(["Third"]);
  });

  test("Windows CRLF blank line (\\r\\n\\r\\n) → two paragraphs", () => {
    expect(segmentTextSections("Hello.\r\n\r\nWorld").length).toBe(2);
  });

  test("three or more newlines → still one section break", () => {
    expect(segmentTextSections("Para one.\n\n\n\nPara two").length).toBe(2);
  });

  test("multiple segments per paragraph", () => {
    const text = "Hello. Nice day.\n\nHow are you? Fine.";
    const r = segmentTextSections(text);
    expect(r.length).toBe(2);
    expect(toStrings(text, r[0])).toEqual(["Hello", "Nice day"]);
    expect(toStrings(text, r[1])).toEqual(["How are you", "Fine"]);
  });

  test("each top-level child is a non-empty paragraph section", () => {
    const r = segmentTextSections("A.\n\nB.\n\nC");
    for (const child of r) {
      expect(isParagraphSection(child)).toBe(true);
      expect(child.length).toBeGreaterThan(0);
    }
  });

  test("paragraph with internal single \\n → one merged segment in paragraph", () => {
    const text = "Line one\nLine two\n\nNew paragraph.";
    const r = segmentTextSections(text);
    expect(r.length).toBe(2);
    expect(r[0]).toHaveLength(1);
    expect(toStrings(text, r[1])).toEqual(["New paragraph"]);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Section geometry
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentTextSections — section geometry", () => {
  test(".start of a paragraph is start of its first segment", () => {
    const text = "Hello. World.\n\nHow are you?";
    const r = segmentTextSections(text);
    expect(r[0].start).toBe(0);
    expect(r[1].start).toBe(15);
  });

  test(".end of a paragraph is end of its last segment", () => {
    const text = "Hello. World.\n\nHow are you?";
    const r = segmentTextSections(text);
    expect(r[0].end).toBe(12);
    expect(r[1].end).toBe(26);
  });

  test(".span equals end - start", () => {
    const text = "Hello. World.\n\nHow are you?";
    const r = segmentTextSections(text);
    expect(r[0].span).toBe(r[0].end - r[0].start);
    expect(r[1].span).toBe(r[1].end - r[1].start);
  });

  test(".extract() returns full paragraph including inter-segment gap", () => {
    const text = "Hello. World.\n\nHow are you?";
    const r = segmentTextSections(text);
    expect(r[0].extract(text)).toBe("Hello. World");
  });

  test(".intersectsWith, .isWithin, .contains are functions", () => {
    const r = segmentTextSections("Hello. World.");
    expect(typeof r.intersectsWith).toBe("function");
    expect(typeof r.isWithin).toBe("function");
    expect(typeof r.contains).toBe("function");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// protectDots integration
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentTextSections — protectDots integration", () => {
  test("'Dr.' does not split a section", () => {
    const text = "Dr. Smith reported the result.\n\nThe team agreed";
    const r = segmentTextSections(text);
    expect(r.length).toBe(2);
    expect(r[0][0].extract(text)).toContain("Dr. Smith");
  });

  test("decimal does not split a paragraph's segments", () => {
    const text = "The value is 3.14 in the formula.\n\nNext paragraph";
    const r = segmentTextSections(text);
    expect(r[0]).toHaveLength(1);
    expect(r[0][0].extract(text)).toContain("3.14");
  });

  test("URL inside paragraph does not split", () => {
    const text = "Visit https://example.com today.\n\nNext paragraph";
    const r = segmentTextSections(text);
    expect(r.length).toBe(2);
    expect(r[0][0].extract(text)).toContain("https://example.com");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Single header — unwrapped
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentTextSections — single header (unwrapped)", () => {
  test("header at document start returns the header section directly", () => {
    const text = "# Title\n\nBody paragraph.";
    const r = segmentTextSections(text);
    expect(isHeaderSection(r)).toBe(true);
    expect(r.header).toBeInstanceOf(Header);
    expect(r.header.level).toBe(1);
  });

  test("header section contains a Header followed by a paragraph child", () => {
    const text = "# Title\n\nBody paragraph.";
    const r = segmentTextSections(text);
    // r[0] is the Header; r[1] is a paragraph Section wrapping the body Segment
    expect(r[0]).toBeInstanceOf(Header);
    expect(isParagraphSection(r[1])).toBe(true);
    expect(r[1][0]).toBeInstanceOf(Segment);
    expect(r[1][0]).not.toBeInstanceOf(Header);
  });

  test("header.extractTitle returns the title text", () => {
    const text = "## Methods\n\nWe used X.";
    const r = segmentTextSections(text);
    expect(r.header.extractTitle(text)).toBe("Methods");
  });

  test("section.level mirrors header.level", () => {
    const text = "### Deep\n\nbody.";
    const r = segmentTextSections(text);
    expect(r.level).toBe(3);
  });

  test("section.content excludes the header", () => {
    const text = "# Title\n\nFirst sentence. Second sentence.";
    const r = segmentTextSections(text);
    // Body is one paragraph section with two segments
    expect(r.content).toHaveLength(1);
    expect(r.content[0]).not.toBeInstanceOf(Header);
    expect(isParagraphSection(r.content[0])).toBe(true);
    expect(r.content[0]).toHaveLength(2);
  });

  test("section.content for two body paragraphs returns two paragraph sections", () => {
    const text = "# Title\n\nFirst paragraph.\n\nSecond paragraph.";
    const r = segmentTextSections(text);
    expect(r.content).toHaveLength(2);
    expect(isParagraphSection(r.content[0])).toBe(true);
    expect(isParagraphSection(r.content[1])).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Same-level headers (siblings)
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentTextSections — same-level headers as siblings", () => {
  test("two level-2 headers produce two top-level children of root", () => {
    const text = "## A\n\nBody A.\n\n## B\n\nBody B.";
    const r = segmentTextSections(text);
    expect(r.length).toBe(2);
    expect(isHeaderSection(r[0])).toBe(true);
    expect(isHeaderSection(r[1])).toBe(true);
    expect(r[0].header.level).toBe(2);
    expect(r[1].header.level).toBe(2);
  });

  test("siblings do not nest as header children", () => {
    const text = "# A\n\nbody.\n\n# B\n\nbody B.";
    const r = segmentTextSections(text);
    expect(r.length).toBe(2);
    // Neither sibling should contain another header section
    for (const sib of r) {
      const childHeaderSections = sib.filter(isHeaderSection);
      expect(childHeaderSections).toHaveLength(0);
    }
  });

  test("each sibling header section owns its body", () => {
    const text = "## A\n\nBody A.\n\n## B\n\nBody B.";
    const r = segmentTextSections(text);
    // Each sibling has [Header, paragraphSection(bodySegment)]
    expect(r[0][1][0].extract(text)).toContain("Body A");
    expect(r[1][1][0].extract(text)).toContain("Body B");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Nested headers
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentTextSections — nested headers", () => {
  test("level-2 inside level-1 nests as child header section", () => {
    const text = "# Top\n\nIntro.\n\n## Sub\n\nSub body.";
    const r = segmentTextSections(text);
    expect(isHeaderSection(r)).toBe(true);
    expect(r.header.level).toBe(1);
    const child = r.find(isHeaderSection);
    expect(child).toBeDefined();
    expect(child.header.level).toBe(2);
  });

  test("intro body lives in the parent as a paragraph section", () => {
    const text = "# Top\n\nIntro paragraph.\n\n## Sub\n\nSub body.";
    const r = segmentTextSections(text);
    // Top-level paragraph children of `r` (excluding child header sections)
    const paragraphs = r.filter(isParagraphSection);
    expect(paragraphs).toHaveLength(1);
    expect(paragraphs[0][0].extract(text)).toContain("Intro paragraph");
  });

  test("multiple level-2 children of one level-1", () => {
    const text = "# Top\n\n## A\n\nbody a.\n\n## B\n\nbody b.";
    const r = segmentTextSections(text);
    const children = r.filter(isHeaderSection);
    expect(children).toHaveLength(2);
    expect(children[0].header.level).toBe(2);
    expect(children[1].header.level).toBe(2);
  });

  test("three-level nesting — # → ## → ###", () => {
    const text = "# A\n\n## B\n\n### C\n\ndeep body.";
    const r = segmentTextSections(text);
    expect(r.header.level).toBe(1);
    const b = r.find(isHeaderSection);
    expect(b).toBeDefined();
    expect(b.header.level).toBe(2);
    const c = b.find(isHeaderSection);
    expect(c).toBeDefined();
    expect(c.header.level).toBe(3);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Header level pop-back
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentTextSections — header level pop-back", () => {
  test("higher-level header (smaller number) closes deeper sections", () => {
    const text = "## A\n\n### B\n\n# C\n\nC body.";
    const r = segmentTextSections(text);
    expect(r.length).toBe(2);
    expect(r[0].header.level).toBe(2);
    expect(r[1].header.level).toBe(1);
  });

  test("level-1 after level-3 pops back to root", () => {
    const text = "# A\n\n### Deep\n\n# B\n\nB body.";
    const r = segmentTextSections(text);
    expect(r.length).toBe(2);
    expect(r[0].header.level).toBe(1);
    expect(r[1].header.level).toBe(1);
  });

  test("partial pop — ### then ## attaches as sibling of parent", () => {
    // # A contains ### B; then ## C should be a child of # A.
    const text = "# A\n\n### B\n\n## C\n\nC body.";
    const r = segmentTextSections(text);
    // r is the unwrapped # A section
    const children = r.filter(isHeaderSection);
    expect(children).toHaveLength(2);
    expect(children[0].header.level).toBe(3);
    expect(children[1].header.level).toBe(2);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Body before any header
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentTextSections — body before first header", () => {
  test("preserved as top-level paragraph section", () => {
    const text = "Intro paragraph.\n\n# Title\n\nBody.";
    const r = segmentTextSections(text);
    expect(r.length).toBe(2);
    expect(isParagraphSection(r[0])).toBe(true);
    expect(isHeaderSection(r[1])).toBe(true);
    expect(r[1].header.level).toBe(1);
  });

  test("multiple pre-header paragraphs split on blank lines", () => {
    const text = "First intro.\n\nSecond intro.\n\n# Title\n\nBody.";
    const r = segmentTextSections(text);
    expect(r.length).toBe(3);
    expect(isParagraphSection(r[0])).toBe(true);
    expect(isParagraphSection(r[1])).toBe(true);
    expect(isHeaderSection(r[2])).toBe(true);
  });

  test("pre-header prose with multiple sentences in one paragraph", () => {
    const text = "Sentence one. Sentence two.\n\n# Title\n\nBody.";
    const r = segmentTextSections(text);
    expect(r[0]).toHaveLength(2); // two sentence segments in one paragraph
    expect(isParagraphSection(r[0])).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// flatten() depth-first traversal
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentTextSections — flatten() document order", () => {
  test("flatten visits self, header, then paragraph and its segment", () => {
    const text = "# Title\n\nBody.";
    const r = segmentTextSections(text);
    const flat = r.flatten();
    expect(flat[0]).toBe(r);
    expect(flat[1]).toBeInstanceOf(Header);
    // Paragraph section wrapping the body
    expect(isParagraphSection(flat[2])).toBe(true);
    // Body segment inside the paragraph
    expect(flat[3]).toBeInstanceOf(Segment);
    expect(flat[3]).not.toBeInstanceOf(Header);
  });

  test("flatten descends into nested header sections", () => {
    const text = "# Top\n\nIntro.\n\n## Sub\n\nSub body.";
    const r = segmentTextSections(text);
    const flat = r.flatten();
    expect(flat[0]).toBe(r);
    const headers = flat.filter((x) => x instanceof Header);
    expect(headers).toHaveLength(2);
    expect(headers[0].level).toBe(1);
    expect(headers[1].level).toBe(2);
  });

  test("body segments from all levels recoverable via flatten", () => {
    const text = "# A\n\nA body.\n\n## B\n\nB body.";
    const r = segmentTextSections(text);
    const bodies = allBodySegments(r);
    expect(bodies).toHaveLength(2);
    expect(bodies[0].extract(text)).toContain("A body");
    expect(bodies[1].extract(text)).toContain("B body");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Consistency with segmentText
// ─────────────────────────────────────────────────────────────────────────────

describe("segmentTextSections — consistency with segmentText (deep)", () => {
  test("deeply flattened bodies match segmentText (header-free input)", () => {
    const text = "Hello. World.\n\nHow are you?";
    const r = segmentTextSections(text);
    const flat = r.flatten().filter(
      (x) => x instanceof Segment && !(x instanceof Header) && !(x instanceof Section)
    );
    const segs = segmentText(text);
    expect(flat.length).toBe(segs.length);
    flat.forEach((seg, i) => {
      expect(seg[0]).toBe(segs[i][0]);
      expect(seg[1]).toBe(segs[i][1]);
    });
  });

  test("deeply flattened output matches segmentText (with headers)", () => {
    const text = "# Title\n\nBody one. Body two.\n\n## Sub\n\nSub body.";
    const r = segmentTextSections(text);
    const flat = r.flatten().filter(
      (x) => (x instanceof Segment || x instanceof Header) && !(x instanceof Section)
    );
    const segs = segmentText(text);
    expect(flat.length).toBe(segs.length);
    flat.forEach((seg, i) => {
      expect(seg[0]).toBe(segs[i][0]);
      expect(seg[1]).toBe(segs[i][1]);
    });
  });
});

