"use strict";

/**
 * @file adaptiveThreshold.js
 * @brief Port of core/adaptive_threshold — adaptive cut primitives for
 * sorted-DESCENDING sequences.
 *
 * The extractor uses {@link adaptivePruneCount} (the elbow → scaled-MM cascade)
 * to pick how many candidates to keep from each frequency-sorted bin without
 * any external cap.
 *
 * Every count function takes an array sorted DESCENDING and returns `k` in
 * `[0, n]` meaning "keep the first k". `k === n` means "no cut"; `k === 0` only
 * for empty / all-non-positive input.
 */

// Defaults — referenced by name so call sites carry no magic numbers.
export const kRatioMinGap = 1.5;
export const kRatioEps = 1e-10;
export const kElbowSig = 0.01;
export const kMMFactor = 2.2;
export const kNoMaxCutIndex = Number.POSITIVE_INFINITY;

/**
 * Perplexity-based effective count: k = ceil(exp(H)) where H is the Shannon
 * entropy of the (unnormalized) head mass. Stops at the first non-positive.
 * @param {number[]} sortedDesc
 * @param {number} [maxCutIndex=kNoMaxCutIndex]
 * @returns {number}
 *
 * @example
 * entropyEffectiveCount([10, 0, 0, 0])   // → 1   (a delta: one dominant mass)
 * entropyEffectiveCount([1, 1, 1, 1])    // → 4   (flat head: keep everything)
 */
export const entropyEffectiveCount = (sortedDesc, maxCutIndex = kNoMaxCutIndex) => {
  const n = sortedDesc.length;
  if (n === 0) return 0;
  const cap = Math.min(Math.max(maxCutIndex, 1), n);

  let s = 0,hUn = 0,l = 0;
  for (let i = 0, v; i < cap; ++i) {
    v = sortedDesc[i];
    if (!(v > 0)) break;
    s += v;
    hUn -= v * Math.log(v);
    ++l;
  }
  if (l === 0) return 0;

  const H = Math.log(s) + hUn / s;
  const expH = Math.exp(H);
  if (!(expH > 1)) return 1;
  const k = Math.ceil(expH);
  return k > l && l || k;
};

/**
 * Sharpest qualifying log-ratio cliff. Walks consecutive pairs, tracks the
 * steepest prev/curr ratio clearing `minGap`, and reports where it sits.
 * @param {number[]} sortedDesc
 * @param {number} [minGap=kRatioMinGap]
 * @param {number} [eps=kRatioEps]
 * @param {number} [maxCutIndex=kNoMaxCutIndex]
 * @returns {number}
 *
 * @example
 * gapRatioEffectiveCount([100, 100, 1, 1])   // → 2   (sharp drop after the 100s)
 * gapRatioEffectiveCount([5, 5, 0, 0])       // → 2   (positive→zero = infinite gap)
 */
export const gapRatioEffectiveCount = (
  sortedDesc,
  minGap = kRatioMinGap,
  eps = kRatioEps,
  maxCutIndex = kNoMaxCutIndex,
) => {
  const n = sortedDesc.length;
  if (n === 0) return 0;
  if (n < 2) return n;
  if (!(eps >= 0)) eps = 0;
  const cap = Math.min(Math.max(maxCutIndex, 1), n);

  let cutIdx = cap;
  let maxGap = 0;
  let ac = sortedDesc[0];
  let i = 1;
  for (; i < cap; ++i) {
    if (!(ac > eps)) break;
    const ap = ac;
    ac = sortedDesc[i];
    if (!(ac > 0)) {
      // Cliff from positive to <= 0: treat as +infinity gap.
      if (Infinity > maxGap) {
        maxGap = Infinity;
        if (maxGap >= minGap) cutIdx = i;
      }
      ++i;
      break;
    }
    const g = ap / ac;
    if (g > maxGap) {
      maxGap = g;
      if (g >= minGap) cutIdx = i;
    }
  }

  let result = Math.min(cutIdx, i);
  if (result < 1) result = 1;
  return result;
};

/**
 * Geometric kneedle: the interior position maximizing chord-to-curve
 * distance, gated by `sig * median(v)`. Returns n when there is no
 * significant elbow.
 * @param {number[]} sortedDesc
 * @param {number} [sig=kElbowSig]
 * @returns {number}
 *
 * @example
 * elbowEffectiveCount([100, 100, 1, 1, 1, 1, 1, 1, 1, 1])   // → 2   (elbow at the cliff)
 * elbowEffectiveCount([1, 1, 1, 1])                         // → 4   (no significant elbow)
 */
export const elbowEffectiveCount = (sortedDesc, sig = kElbowSig) => {
  const n = sortedDesc.length;
  if (n < 3) return n;

  const y0 = sortedDesc[0], yN = sortedDesc[n - 1], yRange = y0 - yN;
  if (!(yRange > 1e-12)) return n;

  let med;
  if (n % 2 === 1) {
    med = sortedDesc[Math.floor(n / 2)];
  } else {
    med = 0.5 * (sortedDesc[n / 2 - 1] + sortedDesc[n / 2]);
  }
  const sigThreshold = sig * med;

  const xN = n - 1;
  let bestDiff = 0, bestI = n;
  for (let i = 1, l = n - 1; i < l; ++i) {
    const t = i / xN, chord = y0 + t * (yN - y0), diff = chord - sortedDesc[i];
    if (diff > bestDiff) {
      bestDiff = diff;
      bestI = i;
    }
  }

  if (bestDiff >= sigThreshold && bestI < n) return bestI;
  return n;
};

/**
 * Slope-aware fallback: min(ceil(factor * medianMass), n), where medianMass
 * is the smallest k whose cumulative head sum reaches half the total.
 * @param {number[]} sortedDesc
 * @param {number} [factor=kMMFactor]
 * @returns {number}
 *
 * @example
 * scaledMmEffectiveCount([1, 1, 1, 1])   // → 4   (saturates to n on a flat mass)
 */
export const scaledMmEffectiveCount = (sortedDesc, factor = kMMFactor) => {
  const n = sortedDesc.length;
  if (n === 0) return 0;

  let total = 0;
  for (let i = 0; i < n; i++) total += sortedDesc[i];
  if (!(total > 0)) return n;

  const half = total * 0.5;
  let cum = 0, mm = n;
  for (let i = 0; i < n; ++i) {
    cum += sortedDesc[i];
    if (cum >= half) {
      mm = i + 1;
      break;
    }
  }

  const scaled = Math.ceil(factor * mm);
  if (!(scaled > 0)) return 1;
  if (scaled >= n) return n;
  return scaled;
};

/**
 * Canonical adaptive prune: elbow first, scaled-MM fallback.
 * @param {number[]} sortedDesc
 * @param {number} [sig=kElbowSig]
 * @param {number} [factor=kMMFactor]
 * @returns {number} number of head elements to keep.
 *
 * @example
 * adaptivePruneCount([100, 100, 1, 1, 1, 1, 1, 1, 1, 1])   // → 2   (elbow wins)
 * adaptivePruneCount([1, 1, 1, 1])                         // → 4   (scaled-MM fallback)
 */
export const adaptivePruneCount = (sortedDesc, sig = kElbowSig, factor = kMMFactor) => {
  const n = sortedDesc.length;
  const kElbow = elbowEffectiveCount(sortedDesc, sig);
  if (kElbow < n) return kElbow;
  return scaledMmEffectiveCount(sortedDesc, factor);
};

/**
 * Value form of {@link adaptivePruneCount}: the boundary (first dropped)
 * value, or 0 when nothing is cut.
 * @param {number[]} sortedDesc
 * @param {number} [sig=kElbowSig]
 * @param {number} [factor=kMMFactor]
 * @returns {number}
 *
 * @example
 * adaptivePruneThreshold([100, 100, 1, 1, 1, 1, 1, 1, 1, 1])   // → 1   (first dropped value)
 * adaptivePruneThreshold([1, 1, 1, 1])                         // → 0   (nothing cut)
 */
export const adaptivePruneThreshold = (sortedDesc, sig = kElbowSig, factor = kMMFactor) => {
  const k = adaptivePruneCount(sortedDesc, sig, factor);
  if (k >= sortedDesc.length) return 0;
  return sortedDesc[k];
};

/**
 * @ignore
 */
export default adaptivePruneCount;
