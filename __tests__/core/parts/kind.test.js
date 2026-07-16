"use strict";

// ============================================================================
// __tests__/core/parts/kind.test.js
//
// Mirrors src/core/parts/kind.js — the Kind enum (position role of a part),
// its string (de)serialization, the shared length bounds, and the invalid-id
// sentinel. Numeric values must match the C++ `enum class Kind`.
// ============================================================================

import {
  Kind,
  kindToString,
  parseKind,
  positionalKind,
  kMinPartLength,
  kMaxPartLength,
  kInvalidPartId,
} from "../../../src/core/parts/kind.js";

describe("kind — Kind enum", () => {
  test("four content kinds plus the delimiter exception (no separate Letter)", () => {
    expect(Kind).toEqual({
      Start: 0,
      End: 1,
      Mid: 2,
      Whole: 3,
      Delimiter: 4,
    });
    expect(Kind.Letter).toBeUndefined();
  });

  test("the enum object is frozen", () => {
    expect(Object.isFrozen(Kind)).toBe(true);
  });
});

describe("kind — positionalKind", () => {
  test("maps boundary position to the length-1 fragment kind (start-wins for a lone char)", () => {
    expect(positionalKind(true, false)).toBe(Kind.Start);
    expect(positionalKind(false, false)).toBe(Kind.Mid);
    expect(positionalKind(false, true)).toBe(Kind.End);
    expect(positionalKind(true, true)).toBe(Kind.Start);
  });
});

describe("kind — kindToString / parseKind", () => {
  const pairs = [
    [Kind.Start, "start"],
    [Kind.End, "end"],
    [Kind.Mid, "mid"],
    [Kind.Whole, "whole"],
    [Kind.Delimiter, "delim"],
  ];

  test.each(pairs)("kindToString(%i) -> %s", (kind, str) => {
    expect(kindToString(kind)).toBe(str);
  });

  test.each(pairs)("parseKind round-trips %i (%s)", (kind, str) => {
    expect(parseKind(str)).toBe(kind);
    expect(parseKind(kindToString(kind))).toBe(kind);
  });

  test("kindToString of an unknown kind is '?'", () => {
    expect(kindToString(99)).toBe("?");
    expect(kindToString(-1)).toBe("?");
  });

  test("parseKind of an unrecognized string is undefined", () => {
    expect(parseKind("nope")).toBeUndefined();
    expect(parseKind("")).toBeUndefined();
  });
});

describe("kind — bounds and sentinels", () => {
  test("length bounds", () => {
    expect(kMinPartLength).toBe(2);
    expect(kMaxPartLength).toBe(7);
    expect(kMinPartLength).toBeLessThan(kMaxPartLength);
  });

  test("invalid part id matches C++ static_cast<uint32_t>(-1)", () => {
    expect(kInvalidPartId).toBe(0xffffffff);
  });
});
