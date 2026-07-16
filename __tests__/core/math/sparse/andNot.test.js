"use strict";

// ============================================================================
// __tests__/core/math/sparse/andNot.test.js
//
// Mirrors src/core/math/sparse/andNot.js — relative complement (A ∖ B) of two
// sorted, strictly increasing index sets. Returns a (typed) array view of the
// elements of `a` absent from `b`.
// ============================================================================

import andNot, { andNot as andNotNamed } from "../../../../src/core/math/sparse/andNot.js";

const arr = (x) => Array.from(x);

describe("sparse/andNot", () => {
  test("keeps values of a absent from b, drops shared ones", () => {
    expect(arr(andNot([1, 2, 3, 5], [2, 3, 4]))).toEqual([1, 5]);
  });

  test("disjoint b leaves a unchanged", () => {
    expect(arr(andNot([1, 3, 5], [2, 4, 6]))).toEqual([1, 3, 5]);
  });

  test("a ⊆ b yields empty", () => {
    expect(arr(andNot([1, 2, 3], [1, 2, 3, 4]))).toEqual([]);
  });

  test("identical sets yield empty", () => {
    expect(arr(andNot([1, 2, 3], [1, 2, 3]))).toEqual([]);
  });

  test("b superset-tail beyond a is ignored", () => {
    expect(arr(andNot([1, 2], [2, 3, 4, 5]))).toEqual([1]);
  });

  test("empty a yields empty; empty b returns all of a", () => {
    expect(arr(andNot([], [1, 2]))).toEqual([]);
    expect(arr(andNot([1, 2], []))).toEqual([1, 2]);
  });

  test("value 0 (falsy part id) survives when absent from b", () => {
    expect(arr(andNot([0, 1, 2], [1]))).toEqual([0, 2]);
    expect(arr(andNot([0, 1, 2], [0]))).toEqual([1, 2]);
  });

  test("Uint32Array input produces a Uint32Array result", () => {
    const out = andNot(Uint32Array.from([1, 2, 3, 5]), Uint32Array.from([2, 3]));
    expect(out).toBeInstanceOf(Uint32Array);
    expect(arr(out)).toEqual([1, 5]);
  });

  test("writes into a provided output buffer", () => {
    const out = new Uint32Array(8);
    const res = andNot([1, 2, 3, 5], [2, 3, 4], out);
    expect(res.buffer).toBe(out.buffer); // view onto the same buffer
    expect(arr(res)).toEqual([1, 5]);
  });

  test("default export equals the named export", () => {
    expect(andNot).toBe(andNotNamed);
  });
});
