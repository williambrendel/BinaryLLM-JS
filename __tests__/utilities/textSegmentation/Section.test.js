"use strict";

/**
 * @file Section.test.js
 * @brief Unit tests for the Section class.
 *
 * Section extends Array — each element is a Segment, Header, or nested Section.
 * Construction: new Section() then .push(segment).
 *
 * Adds to Array:
 *   .start, .end, .span — derived from first/last element
 *   .first, .last       — convenience accessors
 *   .header             — first element if it is a Header (constructor.isHeader)
 *                         OR has an own "level" property (duck typing)
 *   .level              — header level if .header is defined
 *   .content            — elements after header (or all if no header), with
 *                         own .start/.end getters
 *   .extract(text)      — text from .start to .end
 *   .intersectsWith, .isWithin, .contains — geometric tests
 *   .flatten()          — depth-first flat array starting with [this, ...]
 *
 * Section.create(...args) — factory equivalent to new Section(...args).
 * Section.interval re-exports the interval utility.
 */

import Section from "../../../src/utilities/textSegmentation/Section";
import Segment from "../../../src/utilities/textSegmentation/Segment";
import Header  from "../../../src/utilities/textSegmentation/Header";
import segmentTextSections from "../../../src/utilities/textSegmentation/segmentTextSections";

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

describe("Section — construction", () => {
  test("empty section has length 0", () => {
    expect(new Section()).toHaveLength(0);
  });

  test("push segments increases length", () => {
    const s = new Section();
    s.push(new Segment(0, 5));
    s.push(new Segment(7, 12));
    expect(s).toHaveLength(2);
  });

  test("Section.create factory", () => {
    const s = Section.create();
    expect(s).toBeInstanceOf(Section);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Class shape
// ─────────────────────────────────────────────────────────────────────────────

describe("Section — class shape", () => {
  test("extends Array", () => {
    expect(new Section()).toBeInstanceOf(Array);
  });

  test("constructor.name is Section", () => {
    expect(new Section().constructor.name).toBe("Section");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// .start, .end, .span, .first, .last
// ─────────────────────────────────────────────────────────────────────────────

describe("Section — start/end/span/first/last", () => {
  let s;
  beforeEach(() => {
    s = new Section();
    s.push(new Segment(0, 5));
    s.push(new Segment(7, 12));
    s.push(new Segment(15, 20));
  });

  test(".start equals first segment start", () => {
    expect(s.start).toBe(0);
  });

  test(".end equals last segment end", () => {
    expect(s.end).toBe(20);
  });

  test(".span equals end - start", () => {
    expect(s.span).toBe(20);
  });

  test(".first returns first element", () => {
    expect(s.first).toBe(s[0]);
  });

  test(".last returns last element", () => {
    expect(s.last).toBe(s[2]);
  });

  test("empty section .start is undefined", () => {
    expect(new Section().start).toBeUndefined();
  });

  test("empty section .end is undefined", () => {
    expect(new Section().end).toBeUndefined();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Nested-section start/end fallback
// ─────────────────────────────────────────────────────────────────────────────

describe("Section — nested children: start/end fallback", () => {
  test(".start reads from nested Section .start (not [0])", () => {
    const parent = new Section();
    const child = new Section();
    child.push(new Segment(10, 20));
    parent.push(child);
    expect(parent.start).toBe(10);
  });

  test(".end reads from nested Section .end (not [1])", () => {
    const parent = new Section();
    const child = new Section();
    child.push(new Segment(10, 20));
    parent.push(child);
    expect(parent.end).toBe(20);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// .header / .level detection
// ─────────────────────────────────────────────────────────────────────────────

describe("Section — header detection", () => {
  test("plain section has no header", () => {
    const s = new Section();
    s.push(new Segment(0, 5));
    expect(s.header).toBeUndefined();
    expect(s.level).toBeUndefined();
  });

  test("section with Header as first element exposes .header", () => {
    const s = new Section();
    const h = new Header(0, 8, 2, 3);
    s.push(h);
    s.push(new Segment(10, 20));
    expect(s.header).toBe(h);
    expect(s.level).toBe(2);
  });

  test(".header detection via constructor.isHeader (Header instance)", () => {
    const s = new Section();
    s.push(new Header(0, 8, 2, 3));
    expect(s.header).toBeDefined();
  });

  test(".header detection via own 'level' property (duck-typed)", () => {
    // The impl checks hasOwnProperty("level") OR constructor.isHeader.
    const s = new Section();
    const fakeHeader = new Segment(0, 8);
    Object.defineProperty(fakeHeader, "level", { value: 2, enumerable: true });
    s.push(fakeHeader);
    expect(s.header).toBe(fakeHeader);
  });

  test("empty section .header is undefined", () => {
    expect(new Section().header).toBeUndefined();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// .content
// ─────────────────────────────────────────────────────────────────────────────

describe("Section — .content", () => {
  test(".content excludes the header", () => {
    const s = new Section();
    s.push(new Header(0, 8, 1, 2));
    s.push(new Segment(10, 20));
    s.push(new Segment(22, 30));
    expect(s.content).toHaveLength(2);
    expect(s.content[0].start).toBe(10);
    expect(s.content[1].start).toBe(22);
  });

  test(".content on plain section returns all elements", () => {
    const s = new Section();
    s.push(new Segment(0, 5));
    s.push(new Segment(7, 12));
    expect(s.content).toHaveLength(2);
  });

  test(".content has working .start and .end getters", () => {
    const s = new Section();
    s.push(new Header(0, 8, 1, 2));
    s.push(new Segment(10, 20));
    s.push(new Segment(22, 30));
    expect(s.content.start).toBe(10);
    expect(s.content.end).toBe(30);
  });

  test(".content on header-only section returns empty array with undefined start/end", () => {
    const s = new Section();
    s.push(new Header(0, 8, 1, 2));
    expect(s.content).toHaveLength(0);
    expect(s.content.start).toBeUndefined();
    expect(s.content.end).toBeUndefined();
  });

  test(".content is a section", () => {
    const s = new Section();
    s.push(new Segment(0, 5));
    expect(Array.isArray(s.content)).toBe(true);
    expect(s.content).toBeInstanceOf(Section);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// .extract
// ─────────────────────────────────────────────────────────────────────────────

describe("Section — .extract(text)", () => {
  test("returns full paragraph text", () => {
    const text = "Hello, World! How are you?";
    const s = new Section();
    s.push(new Segment(0, 5));
    s.push(new Segment(7, 12));
    expect(s.extract(text)).toBe("Hello, World");
  });

  test("includes inter-segment gap content", () => {
    const text = "First. Second";
    const s = new Section();
    s.push(new Segment(0, 5));
    s.push(new Segment(7, 13));
    expect(s.extract(text)).toBe("First. Second");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Geometric methods
// ─────────────────────────────────────────────────────────────────────────────

describe("Section — geometric methods", () => {
  let s;
  beforeEach(() => {
    s = new Section();
    s.push(new Segment(5, 10));
    s.push(new Segment(12, 15));
  });

  test("intersectsWith — overlap → true", () => {
    expect(s.intersectsWith(10, 20)).toBe(true);
  });

  test("intersectsWith — disjoint → false", () => {
    expect(s.intersectsWith(20, 30)).toBe(false);
  });

  test("isWithin — contained → true", () => {
    expect(s.isWithin(0, 20)).toBe(true);
  });

  test("isWithin — exact match → true", () => {
    expect(s.isWithin(5, 15)).toBe(true);
  });

  test("isWithin — not contained → false", () => {
    expect(s.isWithin(0, 10)).toBe(false);
  });

  test("contains — smaller range → true", () => {
    expect(s.contains(7, 14)).toBe(true);
  });

  test("contains — extending range → false", () => {
    expect(s.contains(0, 20)).toBe(false);
  });

  test("accepts Segment as input", () => {
    expect(s.intersectsWith(new Segment(10, 20))).toBe(true);
  });

  test("accepts [start, end] array as input", () => {
    expect(s.contains([7, 14])).toBe(true);
  });

  test("accepts { start, end } object as input", () => {
    expect(s.intersectsWith({ start: 10, end: 20 })).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// .flatten
// ─────────────────────────────────────────────────────────────────────────────

describe("Section — .flatten()", () => {
  test("flat section: returns [self, ...segments]", () => {
    const s = new Section();
    const a = new Segment(0, 5);
    const b = new Segment(7, 12);
    s.push(a);
    s.push(b);
    const flat = s.flatten();
    expect(flat[0]).toBe(s);
    expect(flat[1]).toBe(a);
    expect(flat[2]).toBe(b);
    expect(flat).toHaveLength(3);
  });

  test("nested sections: depth-first order", () => {
    const parent = new Section();
    const child = new Section();
    const seg1 = new Segment(0, 5);
    const seg2 = new Segment(10, 20);
    child.push(seg2);
    parent.push(seg1);
    parent.push(child);
    const flat = parent.flatten();
    expect(flat[0]).toBe(parent);
    expect(flat[1]).toBe(seg1);
    expect(flat[2]).toBe(child);
    expect(flat[3]).toBe(seg2);
  });

  test("empty section flattens to [self]", () => {
    const s = new Section();
    expect(s.flatten()).toEqual([s]);
  });

  test("plain [s,e] array child gets wrapped in Segment", () => {
    const s = new Section();
    s.push(new Segment(0, 5));
    // Manually push a non-Segment array to test the wrapping branch
    const plainArr = [7, 12];
    s.push(plainArr);
    const flat = s.flatten();
    const wrapped = flat[2];
    expect(wrapped).toBeInstanceOf(Segment);
    expect(wrapped.start).toBe(7);
    expect(wrapped.end).toBe(12);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// .contentSections — ancestor chaining across nested headers
//
// These tests verify the documented behavior: ancestors should chain
// through ALL parent headers, even when a parent header section has no
// direct body content (structural-only). The structural-only chunk is
// filtered from the output, but its header should still appear in the
// ancestor chain of deeper chunks.
//
// If any of these tests fail, the docstring promise is wrong and we
// have a real bug in the walker — or my reading of the docstring is
// off. Either outcome is useful information.
// ─────────────────────────────────────────────────────────────────────────────

describe("Section — .contentSections() ancestor chaining", () => {
  test("H1 → H2 → H3 with body only at H3: ancestors chain through all levels", () => {
    // Sub has no direct body (structural-only); Deep has body.
    // Expected: ONE chunk for Deep's body, with ancestors=[Top, Sub], header=Deep.
    const md = `# Top

Content under top.

## Sub

### Deep

Some deep content here.`;

    const tree = segmentTextSections(md);
    const chunks = tree.contentSections();

    // Find the chunk whose body text mentions "deep content".
    const deep = chunks.find(c =>
      md.slice(c.content.start, c.content.end).includes("deep content")
    );
    expect(deep).toBeDefined();

    // Its own header should be "Deep".
    expect(deep.header).toBeDefined();
    expect(deep.header.extractTitle(md)).toBe("Deep");

    // Its ancestors should be [Top, Sub] — full chain even though Sub
    // had no body and didn't emit its own chunk.
    const ancestorTitles = (deep.ancestors || []).map(h => h.extractTitle(md));
    expect(ancestorTitles).toEqual(["Top", "Sub"]);
  });

  test("H1 → H2 → H3 with body at every level: each chunk has correct ancestors", () => {
    const md = `# Top

Content under top.

## Sub

Content under sub.

### Deep

Some deep content here.`;

    const tree = segmentTextSections(md);
    const chunks = tree.contentSections();

    // Find chunk for each body level.
    const findByBody = needle => chunks.find(c =>
      md.slice(c.content.start, c.content.end).includes(needle)
    );

    const topChunk  = findByBody("Content under top");
    const subChunk  = findByBody("Content under sub");
    const deepChunk = findByBody("deep content here");

    expect(topChunk).toBeDefined();
    expect(subChunk).toBeDefined();
    expect(deepChunk).toBeDefined();

    // Top's own header is Top; its ancestors are empty.
    expect(topChunk.header?.extractTitle(md)).toBe("Top");
    expect((topChunk.ancestors || []).length).toBe(0);

    // Sub's own header is Sub; its ancestors are [Top].
    expect(subChunk.header?.extractTitle(md)).toBe("Sub");
    expect((subChunk.ancestors || []).map(h => h.extractTitle(md))).toEqual(["Top"]);

    // Deep's own header is Deep; its ancestors are [Top, Sub].
    expect(deepChunk.header?.extractTitle(md)).toBe("Deep");
    expect((deepChunk.ancestors || []).map(h => h.extractTitle(md))).toEqual(["Top", "Sub"]);
  });

  test("H2 with no body but with H3 descendant: H2 still appears in H3's body chunk's ancestors", () => {
    // Minimal repro: H2 has zero body, H3 has body. H2 should NOT be
    // dropped from the ancestor chain just because it has no direct
    // content.
    const md = `## EmptyMiddle

### Leaf

Leaf body content.`;

    const tree = segmentTextSections(md);
    const chunks = tree.contentSections();

    // The Leaf body chunk should have EmptyMiddle in its ancestors.
    const leaf = chunks.find(c =>
      md.slice(c.content.start, c.content.end).includes("Leaf body content")
    );
    expect(leaf).toBeDefined();

    const ancestorTitles = (leaf.ancestors || []).map(h => h.extractTitle(md));
    expect(ancestorTitles).toContain("EmptyMiddle");
  });
});