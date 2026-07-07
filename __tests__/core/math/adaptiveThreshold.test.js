"use strict";

// ============================================================================
// __tests__/core/math/adaptiveThreshold.test.js
//
// Mirrors src/core/math/adaptiveThreshold.js — adaptive cut primitives for
// sorted-DESCENDING sequences. Every count function returns k in [0, n]
// meaning "keep the first k"; k === n means "no cut".
// ============================================================================

import adaptivePruneCountDefault, {
  entropyEffectiveCount,
  gapRatioEffectiveCount,
  elbowEffectiveCount,
  scaledMmEffectiveCount,
  adaptivePruneCount,
  adaptivePruneThreshold,
  kRatioMinGap,
  kMMFactor,
  kElbowSig,
} from "../../../src/core/math/adaptiveThreshold.js";

describe("adaptiveThreshold — entropyEffectiveCount", () => {
  test("a delta returns 1", () => {
    expect(entropyEffectiveCount([10, 0, 0, 0])).toBe(1);
  });

  test("a flat head keeps everything", () => {
    expect(entropyEffectiveCount([1, 1, 1, 1])).toBe(4);
  });

  test("empty input returns 0", () => {
    expect(entropyEffectiveCount([])).toBe(0);
  });
});

describe("adaptiveThreshold — scaledMmEffectiveCount", () => {
  test("saturates to n on a flat distribution", () => {
    expect(scaledMmEffectiveCount([1, 1, 1, 1])).toBe(4);
  });

  test("empty input returns 0", () => {
    expect(scaledMmEffectiveCount([])).toBe(0);
  });
});

describe("adaptiveThreshold — elbowEffectiveCount", () => {
  test("fewer than 3 elements returns n", () => {
    expect(elbowEffectiveCount([5, 1])).toBe(2);
    expect(elbowEffectiveCount([5])).toBe(1);
  });

  test("finds the elbow at a sharp cliff", () => {
    expect(elbowEffectiveCount([100, 100, 1, 1, 1, 1, 1, 1, 1, 1])).toBe(2);
  });

  test("flat distribution has no elbow -> returns n", () => {
    expect(elbowEffectiveCount([1, 1, 1, 1])).toBe(4);
  });
});

describe("adaptiveThreshold — gapRatioEffectiveCount", () => {
  test("cuts at the sharpest qualifying ratio cliff", () => {
    expect(gapRatioEffectiveCount([100, 100, 1, 1])).toBe(2);
  });

  test("a positive-to-zero drop is an infinite gap", () => {
    expect(gapRatioEffectiveCount([5, 5, 0, 0])).toBe(2);
  });

  test("single element returns n", () => {
    expect(gapRatioEffectiveCount([7])).toBe(1);
  });
});

describe("adaptiveThreshold — adaptivePruneCount", () => {
  test("finds a sharp cliff via the elbow", () => {
    expect(adaptivePruneCount([100, 100, 1, 1, 1, 1, 1, 1, 1, 1])).toBe(2);
  });

  test("keeps all of a flat distribution", () => {
    expect(adaptivePruneCount([1, 1, 1, 1])).toBe(4);
  });

  test("default export is adaptivePruneCount", () => {
    expect(adaptivePruneCountDefault).toBe(adaptivePruneCount);
  });
});

describe("adaptiveThreshold — adaptivePruneThreshold", () => {
  test("reports the first-dropped value on a cliff", () => {
    // keeps 2 -> boundary value is sortedDesc[2] === 1
    expect(adaptivePruneThreshold([100, 100, 1, 1, 1, 1, 1, 1, 1, 1])).toBe(1);
  });

  test("returns 0 when nothing is cut", () => {
    expect(adaptivePruneThreshold([1, 1, 1, 1])).toBe(0);
  });
});

describe("adaptiveThreshold — exported defaults", () => {
  test("magic-number constants", () => {
    expect(kRatioMinGap).toBe(1.5);
    expect(kMMFactor).toBe(2.2);
    expect(kElbowSig).toBe(0.01);
  });
});
