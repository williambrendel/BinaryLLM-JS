"use strict";

// ============================================================================
// __tests__/core/discovery/heaps.test.js
//
// Mirrors src/core/discovery/heaps.js — the log-log power-law fit K ≈ c·|A|^β
// and the deterministic prefix subsampler.
// ============================================================================

import heapsFit, { heapsFit as fitNamed, subsamplePrefix } from "../../../src/core/discovery/heaps.js";

describe("discovery/heaps", () => {
  test("recovers a planted exponent β on exact power-law points", () => {
    // K = 3 · n^0.5 sampled at several n
    const points = [10, 20, 40, 80, 160, 320].map((n) => ({ n, K: 3 * Math.pow(n, 0.5) }));
    const { beta, c, r2 } = heapsFit(points);
    expect(beta).toBeCloseTo(0.5, 6);
    expect(c).toBeCloseTo(3, 6);
    expect(r2).toBeCloseTo(1, 6);
  });

  test("flat K yields β ≈ 0 (sense-collapse signature)", () => {
    const points = [10, 50, 100, 500].map((n) => ({ n, K: 4 }));
    expect(heapsFit(points).beta).toBeCloseTo(0, 6);
  });

  test("linear growth K = n yields β ≈ 1", () => {
    const points = [5, 10, 20, 40].map((n) => ({ n, K: n }));
    expect(heapsFit(points).beta).toBeCloseTo(1, 6);
  });

  test("ignores zero-K points and needs ≥2 to fit", () => {
    expect(heapsFit([{ n: 10, K: 0 }, { n: 20, K: 3 }]).n).toBe(1);
    expect(Number.isNaN(heapsFit([{ n: 10, K: 0 }, { n: 20, K: 3 }]).beta)).toBe(true);
  });

  test("subsamplePrefix is deterministic, nested, and ≥1", () => {
    const a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    expect(subsamplePrefix(a, 0.5)).toEqual([0, 1, 2, 3, 4]);
    expect(subsamplePrefix(a, 0.3)).toEqual([0, 1, 2]);
    expect(subsamplePrefix(a, 0.01)).toEqual([0]);              // floor of 1
    // nested: smaller fraction is a prefix of the larger
    expect(subsamplePrefix(a, 0.3)).toEqual(subsamplePrefix(a, 0.7).slice(0, 3));
  });

  test("default export equals the named export", () => {
    expect(heapsFit).toBe(fitNamed);
  });
});
