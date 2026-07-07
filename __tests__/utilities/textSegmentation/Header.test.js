"use strict";

/**
 * @file Header.test.js
 * @brief Unit tests for the Header class.
 *
 * Header extends Segment extends Uint32Array(2).
 * Constructor: new Header(start, end, level, titleOffset)
 * Title is derived at query time via .extractTitle(text).
 *
 * Adds to Segment:
 *   .level          — heading depth (1–6), enumerable own property; undefined if omitted
 *   #titleOffset    — private; clamped to [0, span]
 *   .extractTitle(text) — strips leading/trailing # and whitespace from extract(text)
 *   .toJSON()       — overrides to return clean [start, end], excluding level
 *
 * Header.isHeader (static class property) — always true; used by Section
 * for duck-typed header detection.
 */

import Header  from "../../../src/utilities/textSegmentation/Header";
import Segment from "../../../src/utilities/textSegmentation/Segment";

// ─────────────────────────────────────────────────────────────────────────────
// Class shape
// ─────────────────────────────────────────────────────────────────────────────

describe("Header — class shape", () => {
  test("extends Segment",      () => expect(new Header(0, 7, 1)).toBeInstanceOf(Segment));
  test("extends Uint32Array",  () => expect(new Header(0, 7, 1)).toBeInstanceOf(Uint32Array));
  test("has length 2",         () => expect(new Header(0, 7, 1)).toHaveLength(2));
  test("constructor.name is Header", () => {
    expect(new Header(0, 7, 1).constructor.name).toBe("Header");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

describe("Header — construction", () => {
  test("[0] and [1] hold start and end", () => {
    const h = new Header(3, 15, 2);
    expect(h[0]).toBe(3);
    expect(h[1]).toBe(15);
  });

  test(".start and .end match [0] and [1]", () => {
    const h = new Header(3, 15, 2);
    expect(h.start).toBe(3);
    expect(h.end).toBe(15);
  });

  test(".level set from constructor arg", () => {
    expect(new Header(0, 5, 1).level).toBe(1);
    expect(new Header(0, 5, 3).level).toBe(3);
    expect(new Header(0, 5, 6).level).toBe(6);
  });

  test(".level is enumerable own property", () => {
    expect(Object.keys(new Header(0, 5, 2))).toContain("level");
  });

  test(".level is undefined when not provided", () => {
    expect(new Header(0, 5).level).toBeUndefined();
  });

  test(".level is non-writable", () => {
    const h = new Header(0, 5, 2);
    expect(() => { h.level = 99; }).toThrow();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// titleOffset (4th constructor arg)
// ─────────────────────────────────────────────────────────────────────────────

describe("Header — titleOffset (4th constructor arg)", () => {
  test("constructor accepts titleOffset as 4th argument", () => {
    expect(() => new Header(0, 10, 1, 2)).not.toThrow();
  });

  test("titleOffset is private — not exposed as own property", () => {
    const h = new Header(0, 10, 1, 2);
    expect(Object.keys(h)).not.toContain("titleOffset");
    expect(h.titleOffset).toBeUndefined();
  });

  test("titleOffset clamped to span when too large", () => {
    // span = 5; titleOffset 100 should clamp to 5 (no error, no over-read)
    const text = "##abc"; // span = 5
    const h = new Header(0, 5, 2, 100);
    expect(() => h.extractTitle(text)).not.toThrow();
    // After clamping to span=5, extractTitle slices from start+5 to end=5 → empty
    expect(h.extractTitle(text)).toBe("");
  });

  test("titleOffset of 0 — extracts full segment minus markers", () => {
    const text = "## Title";
    const h = new Header(0, 8, 2, 0);
    expect(h.extractTitle(text)).toBe("Title");
  });

  test("titleOffset skips markers — content starts after offset", () => {
    const text = "## Title";
    // titleOffset 3 puts the title at "Title" directly (after "## ")
    const h = new Header(0, 8, 2, 3);
    expect(h.extractTitle(text)).toBe("Title");
  });

  test("missing titleOffset defaults to 0", () => {
    const text = "## Title";
    const h = new Header(0, 8, 2);
    expect(() => h.extractTitle(text)).not.toThrow();
  });

  test("negative titleOffset clamped to 0 (not allowed to go below start)", () => {
    // NOTE: this requires fixing Math.max(titleOffset || 0) → Math.max(0, titleOffset || 0)
    // in Header.js — currently, negative values may survive.
    const text = "# Title";
    const h = new Header(0, text.length, 1, -5);
    expect(() => h.extractTitle(text)).not.toThrow();
    // Slice from start + 0 to end → "# Title" with markers stripped → "Title"
    expect(h.extractTitle(text)).toBe("Title");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// .isHeader
// ─────────────────────────────────────────────────────────────────────────────

describe("Header — .isHeader", () => {
  test("Header.isHeader is true (static class property)", () => {
    expect(Header.isHeader).toBe(true);
  });

  test("instance.constructor.isHeader is true", () => {
    expect(new Header(0, 5, 1).constructor.isHeader).toBe(true);
  });

  test("plain Segment.isHeader is undefined", () => {
    expect(Segment.isHeader).toBeUndefined();
  });

  test("plain Segment instance.constructor.isHeader is undefined", () => {
    expect(new Segment(0, 5).constructor.isHeader).toBeUndefined();
  });

  test("instanceof Header distinguishes from plain Segment", () => {
    expect(new Header(0, 5, 1) instanceof Header).toBe(true);
    expect(new Segment(0, 5) instanceof Header).toBe(false);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// .extractTitle
// ─────────────────────────────────────────────────────────────────────────────

describe("Header — .extractTitle(text)", () => {
  test("strips leading # and space", () => {
    const text = "# My Title\nBody.";
    expect(new Header(0, 10, 1).extractTitle(text)).toBe("My Title");
  });

  test("strips ## and space for level 2", () => {
    const text = "## Background\nBody.";
    expect(new Header(0, 13, 2).extractTitle(text)).toBe("Background");
  });

  test("strips trailing # chars", () => {
    const text = "# Title ##\nBody.";
    expect(new Header(0, 10, 1).extractTitle(text)).toBe("Title");
  });

  test("title with spaces preserved", () => {
    const text = "# What causes biological resistance?\nBody.";
    expect(new Header(0, 36, 1).extractTitle(text))
      .toBe("What causes biological resistance?");
  });

  test("result matches .extract(text) stripped of markers", () => {
    const text = "### Deep Section\nBody.";
    const h    = new Header(0, 16, 3);
    expect(h.extractTitle(text)).toBe(h.extract(text).replace(/^\s*#+\s*/, "").replace(/\s*#+\s*$/, "").trim());
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// .extractTitle — edge cases
// ─────────────────────────────────────────────────────────────────────────────

describe("Header — extractTitle edge cases", () => {
  test("extractTitle on header with no content after markers", () => {
    const text = "##";
    const h = new Header(0, 2, 2, 2);
    expect(h.extractTitle(text)).toBe("");
  });

  test("extractTitle preserves internal whitespace", () => {
    const text = "# Two  spaces  inside";
    const h = new Header(0, text.length, 1, 2);
    // Internal double spaces should not be collapsed (regex only trims edges)
    expect(h.extractTitle(text)).toBe("Two  spaces  inside");
  });

  test("extractTitle on plain (non-marker) header — used for ordered titles", () => {
    // detectOrderedHeader produces Headers without leading # markers.
    // titleOffset points past the "1. " prefix to the actual title.
    const text = "1. Methods";
    const h = new Header(0, text.length, 1, 3);
    expect(h.extractTitle(text)).toBe("Methods");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Inherited Segment methods
// ─────────────────────────────────────────────────────────────────────────────

describe("Header — inherited Segment methods", () => {
  test(".extract() returns full heading line text", () => {
    const text = "# My Title\nBody text.";
    expect(new Header(0, 10, 1).extract(text)).toBe("# My Title");
  });

  test(".span equals end - start", () => {
    expect(new Header(0, 10, 1).span).toBe(10);
  });

  test("destructuring [s, e] works", () => {
    const [s, e] = new Header(3, 12, 2);
    expect(s).toBe(3);
    expect(e).toBe(12);
  });

  test(".intersectsWith works", () => {
    const h = new Header(0, 10, 1);
    expect(h.intersectsWith(5, 15)).toBe(true);
    expect(h.intersectsWith(10, 20)).toBe(false);
  });

  test(".isWithin works", () => {
    expect(new Header(5, 10, 1).isWithin(0, 50)).toBe(true);
  });

  test(".contains works", () => {
    expect(new Header(0, 20, 1).contains(5, 15)).toBe(true);
  });

  test(".getIntersection returns a Segment", () => {
    const i = new Header(0, 15, 1).getIntersection(5, 20);
    expect(i).toBeInstanceOf(Uint32Array);
    expect(i.start).toBe(5);
    expect(i.end).toBe(15);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// toJSON — clean serialization
// ─────────────────────────────────────────────────────────────────────────────

describe("Header — toJSON", () => {
  test("toJSON() returns plain [start, end]", () => {
    expect(new Header(0, 10, 2).toJSON()).toEqual([0, 10]);
  });

  test("JSON.stringify produces [start, end] — level excluded", () => {
    const json = JSON.parse(JSON.stringify(new Header(0, 10, 2)));
    expect(json).toEqual([0, 10]);
    expect(json.level).toBeUndefined();
  });
});

