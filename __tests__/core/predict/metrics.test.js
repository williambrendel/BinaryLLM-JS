"use strict";

// ============================================================================
// __tests__/core/predict/metrics.test.js
//
// The evaluation guardrail (spec §4–§5): log-loss is the decision metric, and a
// model is valid only if it clears the calibration gate (ll ≤ unigram ll). These
// tests use a synthetic dataset where the context (curWord) DETERMINES the label
// (nextWord), so a context model must beat unigram on log-loss AND accuracy.
// ============================================================================

import { buildUnigram, buildContextGram, evaluate, calibrationGate, leafProb, flooredLogProb } from "../../../src/core/predict/index.js";
import { NO_LABEL } from "../../../src/core/signatures/signatureDataset.js";

const V = 5;
const rec = (cur, next) => ({ scale: 0, curWord: cur, nextWord: next, L: [], C: [], R: [] });
const labelOf = (r) => r.nextWord;

// contexts 0,1,2: next = context (deterministic), plus noise on context 0.
const data = [];
for (let i = 0; i < 300; i++) data.push(rec(i % 3, i % 3));
for (let i = 0; i < 30; i++) data.push(rec(0, 1)); // context 0 is 100:30 → 0

describe("predict/metrics — calibration gate", () => {
  test("unigram passes the gate against itself (equal log-loss)", () => {
    const uni = buildUnigram(data, labelOf);
    const m = evaluate(uni, data, { V, labelOf });
    expect(calibrationGate(m.ll, m.ll)).toBe(true);
    expect(m.n).toBe(data.length);
  });

  test("an above-unigram model is rejected", () => {
    expect(calibrationGate(6.2, 6.1)).toBe(false);
    expect(calibrationGate(6.0, 6.1)).toBe(true);
  });
});

describe("predict/metrics — informative context beats unigram", () => {
  const uni = buildUnigram(data, labelOf);
  const bi = buildContextGram(data, labelOf, (r) => r.curWord);
  const u = evaluate(uni, data, { V, labelOf });
  const b = evaluate(bi, data, { V, labelOf });

  test("context model has strictly lower log-loss (the decision metric)", () => {
    expect(b.ll).toBeLessThan(u.ll);
    expect(calibrationGate(b.ll, u.ll)).toBe(true);
  });

  test("and higher accuracy (diagnostic)", () => {
    expect(b.acc).toBeGreaterThan(u.acc);
    expect(b.acc).toBeGreaterThan(0.85); // near-deterministic contexts
  });
});

describe("predict/metrics — invariants", () => {
  test("NO_LABEL records are skipped", () => {
    const withNL = data.concat([rec(0, NO_LABEL)]);
    const m = evaluate(buildUnigram(withNL, labelOf), withNL, { V, labelOf });
    expect(m.n).toBe(data.length); // the unlabeled record dropped
  });

  test("interpolation floor: P_out(w) >= beta * P_unigram(w) for all w", () => {
    const bi = buildContextGram(data, labelOf, (r) => r.curWord);
    const leaf = bi.route(rec(1, 1)); // a sharp deterministic context
    const beta = 0.1, alpha = 0.1;
    for (let w = 0; w < V; w++) {
      const pOut = Math.exp(flooredLogProb(leaf, bi.unigram, w, V, alpha, beta));
      expect(pOut).toBeGreaterThanOrEqual(beta * leafProb(bi.unigram, w, V, alpha) - 1e-12);
    }
  });

  test("floored log-loss never worse than a blow-up would be (finite, bounded)", () => {
    const bi = buildContextGram(data, labelOf, (r) => r.curWord);
    const m = evaluate(bi, data, { V, labelOf });
    expect(Number.isFinite(m.ll)).toBe(true);
    expect(Number.isFinite(m.llFloored)).toBe(true);
  });
});
