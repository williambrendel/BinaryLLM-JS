"use strict";

/**
 * @file heaps.js
 * @brief §9.2 Heaps analysis — does the discovered part count K scale with the
 * number of contexts |A| as a power law `K ≈ c·|A|^β` (finding parts), or stay
 * flat (collapsing to senses)?
 *
 * `heapsFit` is a log-log least-squares fit: `ln K = ln c + β·ln|A|`. The slope
 * β is the Heaps exponent — β ∈ (0,1] with good r² supports "finding parts";
 * β ≈ 0 (flat K) says the operator is collapsing to a fixed sense count (§9.2,
 * §11). Read β in the UNSATURATED regime: a bend at large |A| is exhaustion of a
 * finite part inventory (success), not sense-collapse (design-review caveat).
 */

/**
 * @function heapsFit
 * @param {Array<{n:number, K:number}>} points - (|A|, part-count) observations.
 * @returns {{beta:number, c:number, r2:number, n:number}} power-law exponent β,
 *   prefactor c (so `K ≈ c·n^β`), r² of the log-log fit, and #points used.
 *   `beta=NaN` when fewer than two positive-K points exist.
 */
export const heapsFit = (points) => {
  const pts = points.filter((p) => p.n > 0 && p.K > 0).map((p) => [Math.log(p.n), Math.log(p.K)]);
  const n = pts.length;
  if (n < 2) return { beta: NaN, c: NaN, r2: NaN, n };
  let sx = 0, sy = 0;
  for (const [x, y] of pts) { sx += x; sy += y; }
  const mx = sx / n, my = sy / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (const [x, y] of pts) { sxy += (x - mx) * (y - my); sxx += (x - mx) * (x - mx); syy += (y - my) * (y - my); }
  const beta = sxx > 0 ? sxy / sxx : NaN;
  const c = Math.exp(my - beta * mx);
  const r2 = sxx > 0 && syy > 0 ? (sxy * sxy) / (sxx * syy) : NaN;
  return { beta, c, r2, n };
};

/**
 * @function subsamplePrefix
 * @description Deterministic subsample: the first `round(frac·|arr|)` elements
 * (≥1). Prefix, not random, so the sweep is reproducible and nested (larger
 * fractions contain smaller ones).
 * @param {Array} arr @param {number} frac - in (0,1].
 * @returns {Array}
 */
export const subsamplePrefix = (arr, frac) => arr.slice(0, Math.max(1, Math.round(arr.length * frac)));

export default heapsFit;
