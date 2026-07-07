"use strict";

/**
 * @file Segment.test.js
 * @brief Unit tests for the Segment class.
 *
 * Segment extends Uint32Array(2). Constructor accepts any form that interval()
 * accepts — the full input is forwarded to interval() for normalization:
 *   new Segment(start, end)       — two numbers
 *   new Segment([start, end])     — flat array or typed array
 *   new Segment({ start, end })   — duck-typed object
 *   new Segment(otherSegment)     — copies via duck typing
 *
 * Negative or out-of-range values are clamped to 0.
 * When end < start after clamping, end is set to start.
 *
 * .start → this[0], .end → this[1]
 * .span  → end - start
 * .extract(text), .toJSON() → [start, end]
 * .getIntersection() → Segment | null
 * .intersectsWith(), .isWithin(), .contains() → boolean
 *
 * Segment.interval re-exports the interval utility.
 * Segment.create is a factory equivalent to new Segment(...args).
 */

import Segment  from "../../../src/utilities/textSegmentation/Segment";

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

describe("Segment — construction", () => {
  test("new Segment(start, end) — two numbers", () => {
    const seg = new Segment(3, 10);
    expect(seg[0]).toBe(3);
    expect(seg[1]).toBe(10);
  });

  test("new Segment([start, end]) — flat array", () => {
    const seg = new Segment([5, 15]);
    expect(seg[0]).toBe(5);
    expect(seg[1]).toBe(15);
  });

  test("new Segment(uint32array) — copies values", () => {
    const src = new Uint32Array([7, 20]);
    const seg  = new Segment(src);
    expect(seg[0]).toBe(7);
    expect(seg[1]).toBe(20);
  });

  test("new Segment(uint16array) — copies values", () => {
    const src = new Uint16Array([2, 9]);
    const seg  = new Segment(src);
    expect(seg[0]).toBe(2);
    expect(seg[1]).toBe(9);
  });

  test("new Segment({ start, end }) — duck-typed object", () => {
    const seg = new Segment({ start: 4, end: 11 });
    expect(seg[0]).toBe(4);
    expect(seg[1]).toBe(11);
  });

  test("new Segment(otherSegment) — copies via duck typing", () => {
    const src = new Segment(6, 14);
    const seg  = new Segment(src);
    expect(seg[0]).toBe(6);
    expect(seg[1]).toBe(14);
  });

  test("new Segment(otherSegment) — values are copied, not aliased", () => {
    const src = new Segment(6, 14);
    const copy = new Segment(src);
    src[0] = 999; // would corrupt copy if aliased
    expect(copy[0]).toBe(6);
  });

  test("start = 0 stays 0", () => {
    expect(new Segment(0, 5)[0]).toBe(0);
  });

  test("negative start clamped to 0", () => {
    expect(new Segment(-5, 10)[0]).toBe(0);
  });

  test("negative end clamped to start (end < start → end = start)", () => {
    const seg = new Segment(3, -1);
    expect(seg[1]).toBe(seg[0]); // end clamped to start
  });

  test("clamping: end exactly 0 with start 0 → both 0", () => {
    const seg = new Segment(0, 0);
    expect(seg[0]).toBe(0);
    expect(seg[1]).toBe(0);
  });

  test("no args — both slots are 0", () => {
    const seg = new Segment();
    expect(seg[0]).toBe(0);
    expect(seg[1]).toBe(0);
  });

  test("is an instance of Uint32Array", () => {
    expect(new Segment(0, 5)).toBeInstanceOf(Uint32Array);
  });

  test("has length 2", () => {
    expect(new Segment(0, 5)).toHaveLength(2);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Getters
// ─────────────────────────────────────────────────────────────────────────────

describe("Segment — getters", () => {
  test(".start equals [0]", () => {
    const seg = new Segment(4, 11);
    expect(seg.start).toBe(seg[0]);
    expect(seg.start).toBe(4);
  });

  test(".end equals [1]", () => {
    const seg = new Segment(4, 11);
    expect(seg.end).toBe(seg[1]);
    expect(seg.end).toBe(11);
  });

  test(".span equals end - start", () => {
    expect(new Segment(4, 11).span).toBe(7);
  });

  test(".span is 0 when start equals end", () => {
    expect(new Segment(5, 5).span).toBe(0);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// extract and toJSON
// ─────────────────────────────────────────────────────────────────────────────

describe("Segment — extract and toJSON", () => {
  test(".extract() returns correct substring", () => {
    expect(new Segment(6, 11).extract("Hello world")).toBe("world");
  });

  test(".extract() on full string", () => {
    expect(new Segment(0, 5).extract("Hello")).toBe("Hello");
  });

  test(".toJSON() returns plain array", () => {
    const json = new Segment(3, 9).toJSON();
    expect(Array.isArray(json)).toBe(true);
    expect(json).toEqual([3, 9]);
  });

  test("JSON.stringify serializes as [start, end]", () => {
    expect(JSON.parse(JSON.stringify(new Segment(3, 9)))).toEqual([3, 9]);
  });

  test("destructuring [s, e] works", () => {
    const [s, e] = new Segment(2, 7);
    expect(s).toBe(2);
    expect(e).toBe(7);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Geometric methods
// ─────────────────────────────────────────────────────────────────────────────

describe("Segment — geometric methods", () => {
  // seg = [5, 15]
  let seg;
  beforeAll(() => { seg = new Segment(5, 15); });

  // intersectsWith
  test(".intersectsWith — overlapping → true",          () => expect(seg.intersectsWith(10, 20)).toBe(true));
  test(".intersectsWith — touching end → false",        () => expect(seg.intersectsWith(15, 25)).toBe(false));
  test(".intersectsWith — touching start → false",      () => expect(seg.intersectsWith(0, 5)).toBe(false));
  test(".intersectsWith — fully inside → true",         () => expect(seg.intersectsWith(7, 12)).toBe(true));
  test(".intersectsWith — non-overlapping → false",     () => expect(seg.intersectsWith(20, 30)).toBe(false));

  // isWithin
  test(".isWithin — inside larger range → true",        () => expect(seg.isWithin(0, 50)).toBe(true));
  test(".isWithin — exact same range → true",           () => expect(seg.isWithin(5, 15)).toBe(true));
  test(".isWithin — smaller range → false",             () => expect(seg.isWithin(7, 12)).toBe(false));
  test(".isWithin — partial overlap → false",           () => expect(seg.isWithin(8, 20)).toBe(false));

  // contains
  test(".contains — smaller range → true",              () => expect(seg.contains(7, 12)).toBe(true));
  test(".contains — exact same range → true",           () => expect(seg.contains(5, 15)).toBe(true));
  test(".contains — larger range → false",              () => expect(seg.contains(0, 50)).toBe(false));
  test(".contains — partial overlap → false",           () => expect(seg.contains(3, 10)).toBe(false));

  // getIntersection
  test(".getIntersection — overlap → new Segment", () => {
    const i = seg.getIntersection(10, 20);
    expect(i).not.toBeNull();
    expect(i).toBeInstanceOf(Uint32Array);
    expect(i.start).toBe(10);
    expect(i.end).toBe(15);
  });

  test(".getIntersection — touching end → null",        () => expect(seg.getIntersection(15, 25)).toBeNull());
  test(".getIntersection — no overlap → null",          () => expect(seg.getIntersection(20, 30)).toBeNull());

  test(".getIntersection — fully contained → returns contained range", () => {
    const i = seg.getIntersection(7, 12);
    expect(i.start).toBe(7);
    expect(i.end).toBe(12);
  });

  // Input forms via interval()
  test("geometric methods accept [start, end] array", () => {
    expect(seg.intersectsWith([10, 20])).toBe(true);
    expect(seg.isWithin([0, 50])).toBe(true);
    expect(seg.contains([7, 12])).toBe(true);
  });

  test("geometric methods accept { start, end } object", () => {
    expect(seg.intersectsWith({ start: 10, end: 20 })).toBe(true);
  });

  test("geometric methods accept another Segment", () => {
    expect(seg.intersectsWith(new Segment(10, 20))).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Module export
// ─────────────────────────────────────────────────────────────────────────────

describe("Segment — module export", () => {
  test("Segment.create produces a Segment instance", () => {
    const seg = Segment.create(3, 9);
    expect(seg).toBeInstanceOf(Uint32Array);
    expect(seg[0]).toBe(3);
    expect(seg[1]).toBe(9);
  });

  test("Segment.create accepts all interval input forms", () => {
    expect(Segment.create([3, 9])[0]).toBe(3);
    expect(Segment.create({ start: 3, end: 9 })[0]).toBe(3);
  });

  test("Segment.create equals new Segment for all input forms", () => {
    const a = new Segment(3, 9);
    const b = Segment.create(3, 9);
    expect(b[0]).toBe(a[0]);
    expect(b[1]).toBe(a[1]);
    expect(b).toBeInstanceOf(Segment);
  });
});