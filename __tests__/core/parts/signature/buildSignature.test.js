"use strict";

// Tests for the chunk-contextual [L,C,R] signature (buildChunkSignatures).
import { buildChunkSignatures, sparseF } from "../../../../src/core/parts/signature/index.js";
import { generateParts } from "../../../../src/core/parts/generation/index.js";
import { addBigramBackstop, fuzzyEncode, bpeEncode } from "../../../../src/core/parts/encoding/index.js";
import { buildInvertedIndex } from "../../../../src/core/parts/normalization/index.js";

// A small real dictionary from an inline corpus.
const corpus = "the cat sat on the mat the dog ran running runner the the the jumping jumped";
const { dict } = generateParts(new Map(
  corpus.split(" ").reduce((m, w) => m.set(w, (m.get(w) || 0) + 1), new Map())
));
addBigramBackstop(dict);

const arr = (u) => Array.from(u);
const eq = (a, b) => arr(a).join(",") === arr(b).join(",");
const isSorted = (u) => arr(u).every((v, i) => i === 0 || u[i - 1] <= v);
const superset = (big, small) => { const s = new Set(arr(big)); return arr(small).every((v) => s.has(v)); };
const union = (...us) => [...new Set(us.flatMap(arr))].sort((a, b) => a - b);
// The sparse (peel) signature: the id-set C is built from — NOT the fuzzy F.
const sparse = (w) => [...new Set(bpeEncode(dict, w).parts.map((p) => p.id))].sort((a, b) => a - b);

describe("buildChunkSignatures — structure", () => {
  const words = ["the", "cat", "sat"];
  const sig = buildChunkSignatures(words, { dict });

  test("one entry per word, in order", () => {
    expect(sig).toHaveLength(3);
    expect(sig.map((s) => s.word)).toEqual(words);
  });

  test("C_i is the word's sparse (peel) signature — not the fuzzy F", () => {
    for (let i = 0; i < words.length; i++) expect(eq(sig[i].C, sparse(words[i]))).toBe(true);
  });

  test("every signature is a sorted Uint32Array", () => {
    for (const s of sig) for (const f of [s.C, s.L, s.R, s.LC]) { expect(f).toBeInstanceOf(Uint32Array); expect(isSorted(f)).toBe(true); }
  });

  test("boundaries: L_0 empty, R_last empty", () => {
    expect(sig[0].L.length).toBe(0);
    expect(sig[sig.length - 1].R.length).toBe(0);
  });

  test("L_i equals the previous prefix-union LC_{i-1}", () => {
    for (let i = 1; i < sig.length; i++) expect(eq(sig[i].L, sig[i - 1].LC)).toBe(true);
  });

  test("LC_i = L_i ∪ C_i and is a superset of both", () => {
    for (let i = 0; i < sig.length; i++) {
      expect(arr(sig[i].LC)).toEqual(union(sig[i].L, sig[i].C));
      expect(superset(sig[i].LC, sig[i].C)).toBe(true);
      expect(superset(sig[i].LC, sig[i].L)).toBe(true);
    }
  });

  test("prefix-union is monotonic and ends at the whole-chunk union", () => {
    for (let i = 1; i < sig.length; i++) expect(superset(sig[i].LC, sig[i - 1].LC)).toBe(true);
    expect(arr(sig[sig.length - 1].LC)).toEqual(union(...sig.map((s) => s.C)));
  });

  test("R is monotonic backward (R_i superset of R_{i+1})", () => {
    for (let i = 0; i < sig.length - 1; i++) expect(superset(sig[i].R, sig[i + 1].R)).toBe(true);
  });
});

describe("buildChunkSignatures — OR-pooling properties", () => {
  test("[L∪C] is order-invariant: 'cat dog' and 'dog cat' share the final prefix-union", () => {
    const ab = buildChunkSignatures(["cat", "dog"], { dict });
    const ba = buildChunkSignatures(["dog", "cat"], { dict });
    expect(eq(ab[1].LC, ba[1].LC)).toBe(true);
  });

  test("single-word chunk: L and R empty, LC === C", () => {
    const [s] = buildChunkSignatures(["running"], { dict });
    expect(s.L.length).toBe(0);
    expect(s.R.length).toBe(0);
    expect(eq(s.LC, s.C)).toBe(true);
  });
});

describe("buildChunkSignatures — windowed context", () => {
  const words = ["the", "cat", "sat", "on", "the", "mat", "the", "dog", "ran"];
  const rangeUnion = (C, lo, hi) => {
    const us = [];
    for (let j = Math.max(0, lo); j <= Math.min(C.length - 1, hi); j++) us.push(C[j]);
    return us.length ? union(...us) : [];
  };

  test("window >= chunk length is identical to unbounded", () => {
    const un = buildChunkSignatures(words, { dict });
    const wi = buildChunkSignatures(words, { dict, window: words.length });
    for (let i = 0; i < words.length; i++) {
      expect(eq(wi[i].LC, un[i].LC)).toBe(true);
      expect(eq(wi[i].L, un[i].L)).toBe(true);
      expect(eq(wi[i].R, un[i].R)).toBe(true);
    }
  });

  test("windowed L/R/LC are the OR over exactly the last/next `window-1` words", () => {
    const w = 5;
    const sig = buildChunkSignatures(words, { dict, window: w });
    const C = sig.map((s) => s.C);
    for (let i = 0; i < words.length; i++) {
      expect(arr(sig[i].LC)).toEqual(rangeUnion(C, i - (w - 1), i));
      expect(arr(sig[i].L)).toEqual(rangeUnion(C, i - (w - 1), i - 1));
      expect(arr(sig[i].R)).toEqual(rangeUnion(C, i + 1, i + (w - 1)));
    }
  });

  test("boundaries hold under windowing: L_0 empty, R_last empty", () => {
    const sig = buildChunkSignatures(words, { dict, window: 3 });
    expect(sig[0].L.length).toBe(0);
    expect(sig[sig.length - 1].R.length).toBe(0);
  });

  test("window caps saturation: a distant position no longer sees the whole prefix", () => {
    const w = 3;
    const un = buildChunkSignatures(words, { dict });
    const wi = buildChunkSignatures(words, { dict, window: w });
    // The last position's unbounded LC is the whole-chunk union; the windowed one
    // is only the last `w` words — strictly smaller here.
    const last = words.length - 1;
    expect(superset(un[last].LC, wi[last].LC)).toBe(true);
    expect(wi[last].LC.length).toBeLessThan(un[last].LC.length);
  });
});

describe("buildChunkSignatures — causal phase", () => {
  const words = ["the", "cat", "sat", "on", "the", "mat", "the", "dog", "ran"];
  const r = 3;
  const sig = buildChunkSignatures(words, { dict, phase: "causal", radius: r });
  const C = sig.map((s) => s.C);
  const rangeU = (lo, hi) => { const us = []; for (let j = Math.max(0, lo); j <= Math.min(C.length - 1, hi); j++) us.push(C[j]); return us.length ? union(...us) : []; };

  test("R is empty everywhere (the future is forbidden)", () => {
    for (const s of sig) expect(s.R.length).toBe(0);
  });

  test("LC = OR over the last r+1 words; L = same minus current", () => {
    for (let i = 0; i < words.length; i++) {
      expect(arr(sig[i].LC)).toEqual(rangeU(i - r, i));
      expect(arr(sig[i].L)).toEqual(rangeU(i - r, i - 1));
      expect(arr(sig[i].LC)).toEqual(union(sig[i].L, sig[i].C));
    }
  });

  test("boundary: L_0 empty, LC_0 === C_0", () => {
    expect(sig[0].L.length).toBe(0);
    expect(eq(sig[0].LC, sig[0].C)).toBe(true);
  });

  test("causal signature ignores the future (same prefix ⇒ same LC)", () => {
    const a = buildChunkSignatures(["the", "cat", "sat", "on"], { dict, phase: "causal", radius: r });
    const b = buildChunkSignatures(["the", "cat", "dog", "ran"], { dict, phase: "causal", radius: r });
    expect(eq(a[1].LC, b[1].LC)).toBe(true); // positions 0..1 identical
  });
});

describe("buildChunkSignatures — bidirectional phase (shift-pad)", () => {
  const words = ["the", "cat", "sat", "on", "the", "mat", "the", "dog", "ran"]; // 9 > 2r+1 (r=2)
  const r = 2;
  const shiftWin = (i, k) => { if (k <= 2 * r + 1) return [0, k - 1]; let lo = i - r, hi = i + r; if (lo < 0) { hi -= lo; lo = 0; } else if (hi > k - 1) { lo -= hi - (k - 1); hi = k - 1; } return [lo, hi]; };
  const sig = buildChunkSignatures(words, { dict, phase: "bidirectional", radius: r });
  const C = sig.map((s) => s.C);
  const rangeU = (lo, hi) => { const us = []; for (let j = Math.max(0, lo); j <= Math.min(C.length - 1, hi); j++) us.push(C[j]); return us.length ? union(...us) : []; };

  test("LC is empty everywhere ([L∪C] belongs to the causal phase)", () => {
    for (const s of sig) expect(s.LC.length).toBe(0);
  });

  test("L and R follow the shift-padded window", () => {
    for (let i = 0; i < words.length; i++) {
      const [lo, hi] = shiftWin(i, words.length);
      expect(arr(sig[i].L)).toEqual(rangeU(lo, i - 1));
      expect(arr(sig[i].R)).toEqual(rangeU(i + 1, hi));
    }
  });

  test("shift-pad compensates at the start: position 0 R spans 2r words", () => {
    expect(sig[0].L.length).toBe(0);
    expect(arr(sig[0].R)).toEqual(rangeU(1, 2 * r)); // window [0, 2r]
  });

  test("shiftPad:false falls back to a plain clamp (R reaches only r)", () => {
    const tr = buildChunkSignatures(words, { dict, phase: "bidirectional", radius: r, shiftPad: false });
    expect(arr(tr[0].R)).toEqual(rangeU(1, r));
  });

  test("boundaries: L_0 empty, R_last empty", () => {
    expect(sig[0].L.length).toBe(0);
    expect(sig[words.length - 1].R.length).toBe(0);
  });
});

describe("buildChunkSignatures — canonicalization", () => {
  test("encoder:'fuzzy' derives C from the fuzzy F; default/'sparse' from the peel", () => {
    const w = ["running", "dream"];
    const sp = buildChunkSignatures(w, { dict, encoder: "sparse" });
    const fz = buildChunkSignatures(w, { dict, encoder: "fuzzy" });
    const def = buildChunkSignatures(w, { dict });
    for (let i = 0; i < w.length; i++) {
      expect(eq(sp[i].C, sparse(w[i]))).toBe(true);
      expect(eq(fz[i].C, fuzzyEncode(dict, w[i]))).toBe(true);
      expect(eq(def[i].C, sp[i].C)).toBe(true); // default is sparse
    }
    // fuzzy C is strictly larger (redundant containment) than the sparse peel
    expect(fz[0].C.length).toBeGreaterThan(sp[0].C.length);
  });

  test("with an index (fuzzy F key), C is the sparse peel of the canonical word", () => {
    const words = [...new Set(corpus.split(" "))];
    // The index is keyed on the redundant FUZZY F (the NN recovery key)...
    const index = buildInvertedIndex(words, words.map((w) => fuzzyEncode(dict, w)));
    const sig = buildChunkSignatures(["runing"], { dict, index }); // typo of "running"
    expect(typeof sig[0].word).toBe("string");
    // ...but the signature content C is the SPARSE peel of the recovered word.
    expect(eq(sig[0].C, sparse(sig[0].word))).toBe(true);
  });

  test("peel cache: NN hit returns index.peels[wi], identical to re-peeling", () => {
    const words = [...new Set(corpus.split(" "))];
    const F = words.map((w) => fuzzyEncode(dict, w));
    const peels = words.map((w) => sparseF(dict, w));
    const cached = buildInvertedIndex(words, F, peels); // with peel cache
    const plain = buildInvertedIndex(words, F);         // no cache → re-peels
    const chunk = ["runing", "the", "runner"]; // typo + clean
    const a = buildChunkSignatures(chunk, { dict, index: cached });
    const b = buildChunkSignatures(chunk, { dict, index: plain });
    for (let i = 0; i < chunk.length; i++) expect(eq(a[i].C, b[i].C)).toBe(true); // cache ≡ re-peel
    // the cached C for the recovered word is exactly index.peels[wi]
    const wi = words.indexOf(a[0].word);
    expect(wi).toBeGreaterThanOrEqual(0);
    expect(eq(a[0].C, cached.peels[wi])).toBe(true);
  });
});
