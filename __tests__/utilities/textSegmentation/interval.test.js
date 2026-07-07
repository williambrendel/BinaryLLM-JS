"use strict";

/**
 * @file interval.test.js
 * @brief Unit tests for the interval utility and its geometric helpers.
 *
 * interval() normalizes any supported range form into { start, end }.
 * Accepted forms:
 *   (start, end)          — two numbers
 *   { start, end }        — duck-typed object
 *   [start, end]          — flat array or typed array
 *   [[s,e], [s,e], ...]   — array of pairs → span of first to last
 *
 * interval.intersect(a, b)  → { start, end } | null
 * interval.intersects(a, b) → boolean
 * interval.contains(big, small) → boolean
 */

import interval, { intersect, intersects, contains } from "../../../src/utilities/textSegmentation/interval";
import Segment from "../../../src/utilities/textSegmentation/Segment";

// ─────────────────────────────────────────────────────────────────────────────
// interval() — normalization
// ─────────────────────────────────────────────────────────────────────────────

describe("interval — normalization", () => {
  test("two numbers → { start, end }", () => {
    expect(interval(3, 9)).toEqual({ start: 3, end: 9 });
  });

  test("start === end → { start, end } with equal values", () => {
    expect(interval(5, 5)).toEqual({ start: 5, end: 5 });
  });

  test("single number with no end → end defaults to start", () => {
    expect(interval(7)).toEqual({ start: 7, end: 7 });
  });

  test("{ start, end } object passes through unchanged", () => {
    const obj = { start: 2, end: 8 };
    expect(interval(obj)).toBe(obj);
  });

  test("object with extra properties passes through", () => {
    const obj = { start: 1, end: 4, level: 2 };
    expect(interval(obj)).toBe(obj);
  });

  test("flat [start, end] array", () => {
    expect(interval([3, 9])).toEqual({ start: 3, end: 9 });
  });

  test("Uint32Array [start, end]", () => {
    const arr = new Uint32Array([5, 12]);
    const result = interval(arr);
    expect(result.start).toBe(5);
    expect(result.end).toBe(12);
  });

  test("Uint16Array [start, end]", () => {
    const arr = new Uint16Array([2, 7]);
    const result = interval(arr);
    expect(result.start).toBe(2);
    expect(result.end).toBe(7);
  });

  test("array of pairs [[s,e],[s,e]] → span from first start to last end", () => {
    const result = interval([[0, 5], [7, 12]]);
    expect(result.start).toBe(0);
    expect(result.end).toBe(12);
  });

  test("three-element array — uses first and last", () => {
    // interval([0, 5, 12]) → interval(0, 12)
    expect(interval([0, 5, 12])).toEqual({ start: 0, end: 12 });
  });

  test("Segment-like object (duck typing)", () => {
    const seg = { start: 10, end: 20, span: 10, extract: () => "" };
    expect(interval(seg)).toBe(seg);
  });

  test("Segment instance (Uint32Array with .start/.end getters)", () => {
    const seg = new Segment(3, 9);
    const result = interval(seg);
    expect(result.start).toBe(3);
    expect(result.end).toBe(9);
  });

  test("empty array → falls through gracefully", () => {
    expect(() => interval([])).not.toThrow();
  });

  test("array with single element [n] → start and end both n", () => {
    expect(interval([5])).toEqual({ start: 5, end: 5 });
  });

  test("nested empty arrays [[]] — does not throw", () => {
    expect(() => interval([[]])).not.toThrow();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// interval — variadic with extra args
// ─────────────────────────────────────────────────────────────────────────────

describe("interval — variadic with extra args", () => {
  test("third+ args ignored when first is a number — last positional wins", () => {
    expect(interval(1, 2, 99, 100)).toEqual({ start: 1, end: 100 });
  });

  test("first arg is array — extra positional args do not crash", () => {
    const result = interval([0, 5], [10, 20]);
    expect(result).toBeDefined();
    expect(typeof result.start).toBe("number");
    expect(typeof result.end).toBe("number");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// intersect()
// ─────────────────────────────────────────────────────────────────────────────

describe("intersect", () => {
  test("overlapping ranges → intersection range", () => {
    expect(intersect([0, 10], [5, 15])).toEqual({ start: 5, end: 10 });
  });

  test("fully contained → returns inner range", () => {
    expect(intersect([0, 20], [5, 15])).toEqual({ start: 5, end: 15 });
  });

  test("exact match → returns same range", () => {
    expect(intersect([3, 9], [3, 9])).toEqual({ start: 3, end: 9 });
  });

  test("touching (end === start) → null", () => {
    expect(intersect([0, 5], [5, 10])).toBeNull();
  });

  test("disjoint → null", () => {
    expect(intersect([0, 5], [6, 10])).toBeNull();
  });

  test("reversed disjoint → null", () => {
    expect(intersect([6, 10], [0, 5])).toBeNull();
  });

  test("accepts { start, end } objects", () => {
    expect(intersect({ start: 0, end: 10 }, { start: 5, end: 15 }))
      .toEqual({ start: 5, end: 10 });
  });

  test("accepts mixed input forms", () => {
    expect(intersect([0, 10], { start: 5, end: 15 }))
      .toEqual({ start: 5, end: 10 });
  });

  test("writes into provided output object", () => {
    const out = {};
    const result = intersect([0, 10], [5, 15], out);
    expect(result).toBe(out);
    expect(out).toEqual({ start: 5, end: 10 });
  });

  test("returns null without mutating output when no intersection", () => {
    const out = { start: 99, end: 99 };
    intersect([0, 5], [6, 10], out);
    // start gets written (max), end gets written (min) — but result is null
    expect(intersect([0, 5], [6, 10])).toBeNull();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// intersects()
// ─────────────────────────────────────────────────────────────────────────────

describe("intersects", () => {
  test("overlapping → true", () => {
    expect(intersects([0, 10], [5, 15])).toBe(true);
  });

  test("fully contained → true", () => {
    expect(intersects([0, 20], [5, 15])).toBe(true);
  });

  test("reverse contained → true", () => {
    expect(intersects([5, 15], [0, 20])).toBe(true);
  });

  test("exact match → true", () => {
    expect(intersects([3, 9], [3, 9])).toBe(true);
  });

  test("touching end → false", () => {
    expect(intersects([0, 5], [5, 10])).toBe(false);
  });

  test("touching start → false", () => {
    expect(intersects([5, 10], [0, 5])).toBe(false);
  });

  test("disjoint → false", () => {
    expect(intersects([0, 5], [6, 10])).toBe(false);
  });

  test("accepts { start, end } objects", () => {
    expect(intersects({ start: 0, end: 10 }, { start: 5, end: 15 })).toBe(true);
  });

  test("accepts two-number form", () => {
    expect(intersects(interval(0, 10), interval(5, 15))).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// contains()
// ─────────────────────────────────────────────────────────────────────────────

describe("contains", () => {
  test("big fully contains small → true", () => {
    expect(contains([0, 20], [5, 15])).toBe(true);
  });

  test("exact match → true", () => {
    expect(contains([0, 20], [0, 20])).toBe(true);
  });

  test("small extends beyond big end → false", () => {
    expect(contains([0, 20], [15, 25])).toBe(false);
  });

  test("small starts before big → false", () => {
    expect(contains([5, 20], [0, 15])).toBe(false);
  });

  test("big is smaller than small → false", () => {
    expect(contains([5, 15], [0, 20])).toBe(false);
  });

  test("touching start — small starts exactly at big start → true", () => {
    expect(contains([5, 20], [5, 15])).toBe(true);
  });

  test("touching end — small ends exactly at big end → true", () => {
    expect(contains([0, 20], [5, 20])).toBe(true);
  });

  test("accepts { start, end } objects", () => {
    expect(contains({ start: 0, end: 20 }, { start: 5, end: 15 })).toBe(true);
  });

  test("accepts mixed input forms", () => {
    expect(contains([0, 20], { start: 5, end: 15 })).toBe(true);
  });
});