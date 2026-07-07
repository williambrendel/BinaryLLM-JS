"use strict";

// ============================================================================
// __tests__/utilities/sparseArrayType.test.js
// ============================================================================

import { sparseArrayType } from "../../src/utilities/sparseArrayType.js";

describe("sparseArrayType", () => {
  test("Uint16Array while ids fit in 16 bits (dim <= 65536)", () => {
    expect(sparseArrayType(1)).toBe(Uint16Array);
    expect(sparseArrayType(509)).toBe(Uint16Array); // fixture dict F
    expect(sparseArrayType(33104)).toBe(Uint16Array); // dict_combined-scale band
    expect(sparseArrayType(65535)).toBe(Uint16Array);
    expect(sparseArrayType(65536)).toBe(Uint16Array); // largest id 65535 still fits
  });

  test("Uint32Array once ids exceed 16 bits (dim > 65536)", () => {
    expect(sparseArrayType(65537)).toBe(Uint32Array);
    expect(sparseArrayType(99312)).toBe(Uint32Array); // 3F of dict_combined
    expect(sparseArrayType(1_000_000)).toBe(Uint32Array);
  });

  test("the boundary is exactly 65536", () => {
    expect(sparseArrayType(65536)).toBe(Uint16Array);
    expect(sparseArrayType(65537)).toBe(Uint32Array);
  });

  test("chosen constructor actually holds the largest id", () => {
    for (const dim of [509, 65536, 65537, 99312]) {
      const Arr = sparseArrayType(dim);
      const a = Arr.from([dim - 1]);
      expect(a[0]).toBe(dim - 1); // no truncation
    }
  });
});
