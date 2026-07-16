"use strict";

/**
 * @file growPart.js
 * @brief Grow one coverage-fired part for a node split (the `best_split` primitive).
 *
 * A part `p` is a set of bits; a sample `x` fires it iff `popcount(x ∧ p) ≥ k` — it
 * contains at least `k` of the part's bits. Coverage lives in integer `k`-space and
 * `|p|` never enters scoring, so a wider part is capacity, never reward. The split
 * objective is the per-class binary-entropy gain: with per-class weights
 * `w_l ∝ n_l^{−α}` (Σ w_l = 1) and Left-fractions `f_l = a_l / n_l`,
 *
 *   P_L = Σ_l w_l f_l,   Gain = H_b(P_L) − Σ_l w_l H_b(f_l)   ∈ [0, 1]
 *
 * Gain is 1 for a balanced bipartition of the label set with every class kept
 * intact, 0 for no split or no separation. `α=0` weights classes equally (raw
 * balance); `α=1 → w_l ∝ 1/n_l` (per-label balance). Bits accrete greedily: each
 * step adds the candidate (from a frequency shortlist of width `M`) with the
 * highest joint Gain, stopping when the marginal gain `≤ λ` (an MDL floor that
 * prevents memorizing `p`). The threshold `k*` is chosen jointly by a suffix-sum
 * sweep over `k = |p|…1`, both sides held to the `s_min` floor.
 */

// Binary entropy (nats).
const Hb = (p) => (p <= 0 || p >= 1 ? 0 : -p * Math.log(p) - (1 - p) * Math.log(1 - p));

/**
 * @function growPart
 * @description Grows one coverage-fired part and its threshold `k*`; returns the split.
 * @param {number[]} idx - Node sample indices.
 * @param {number[][]} featArr - Per-sample sorted bit arrays.
 * @param {Set<number>[]} featSet - Per-sample bit sets.
 * @param {number[]} labels - Per-sample labels.
 * @param {Object} [opts]
 * @param {number} [opts.D=4] - Max part width (bit-accretion cap).
 * @param {number} [opts.sMin=20] - Support floor per side (counts).
 * @param {number} [opts.alpha=1] - Class-weight exponent `w_l ∝ n_l^{−α}`.
 * @param {number} [opts.lambda=1e-9] - MDL floor: stop when marginal gain ≤ λ.
 * @param {number} [opts.M=48] - Candidate bits per accretion step (frequency shortlist).
 * @returns {{bits:number[], k:number, threshold:number, fire:number[], notFire:number[], gain:number}|null}
 */
export const growPart = (idx, featArr, featSet, labels, opts = {}) => {
  const { D = 4, sMin = 20, alpha = 0, lambda = 1e-9, M = 48, gainType = "ig" } = opts;
  const P = idx.length;

  // Node-local class index: counts n_l and weights w_l ∝ n_l^{−α} (Σ = 1).
  const classIdx = new Map(); const nCount = []; const labPos = new Int32Array(P);
  for (let i = 0; i < P; i++) {
    const l = labels[idx[i]];
    let c = classIdx.get(l);
    if (c === undefined) { c = nCount.length; classIdx.set(l, c); nCount.push(0); }
    labPos[i] = c; nCount[c]++;
  }
  const C = nCount.length;
  if (C <= 1) return null; // pure node
  const w = new Float64Array(C); let wsum = 0;
  for (let c = 0; c < C; c++) { const wc = alpha === 0 ? 1 : nCount[c] ** -alpha; w[c] = wc; wsum += wc; }
  for (let c = 0; c < C; c++) w[c] /= wsum;

  // Gain from a Left-count vector a[c] against fixed totals nCount[c]. Two objectives:
  //   'hb' — balanced label-set bipartition (H_b(P_L) − Σ w_l H_b(f_l)).
  //   'ig' — purity-aligned multiclass entropy reduction (weighted information gain),
  //          the reduction in leaf label-entropy the split buys. Matches log-loss.
  const gainHb = (a) => {
    let PL = 0, spread = 0;
    for (let c = 0; c < C; c++) { const f = a[c] / nCount[c]; PL += w[c] * f; spread += w[c] * Hb(f); }
    return Hb(PL) - spread;
  };
  // Weighted precompute for the impurity criteria: per-sample class weight
  // ω_c = n_c^{−α}, node weighted mass, node entropy and Gini.
  const weighted = gainType === "ig" || gainType === "gini";
  const omega = new Float64Array(C); let Wnode = 0;
  if (weighted) { for (let c = 0; c < C; c++) { omega[c] = alpha === 0 ? 1 : nCount[c] ** -alpha; Wnode += nCount[c] * omega[c]; } }
  let Hnode = 0, Gnode = 0;
  if (weighted) for (let c = 0; c < C; c++) { const p = nCount[c] * omega[c] / Wnode; if (p > 0) { Hnode -= p * Math.log(p); Gnode += p * p; } }
  Gnode = 1 - Gnode;
  // IG — entropy reduction (rewards average class separation → balanced bisects).
  const gainIg = (a) => {
    let WL = 0; for (let c = 0; c < C; c++) WL += a[c] * omega[c];
    const WR = Wnode - WL; if (WL <= 0 || WR <= 0) return 0;
    let HL = 0, HR = 0;
    for (let c = 0; c < C; c++) { const wl = a[c] * omega[c]; if (wl > 0) { const p = wl / WL; HL -= p * Math.log(p); } const wr = (nCount[c] - a[c]) * omega[c]; if (wr > 0) { const p = wr / WR; HR -= p * Math.log(p); } }
    return Hnode - (WL / Wnode) * HL - (WR / Wnode) * HR;
  };
  // Gini reduction — rewards isolating the dominant class (→ lopsided pure peels).
  const gainGini = (a) => {
    let WL = 0; for (let c = 0; c < C; c++) WL += a[c] * omega[c];
    const WR = Wnode - WL; if (WL <= 0 || WR <= 0) return 0;
    let sL = 0, sR = 0;
    for (let c = 0; c < C; c++) { const wl = a[c] * omega[c]; sL += wl * wl; const wr = (nCount[c] - a[c]) * omega[c]; sR += wr * wr; }
    const GL = 1 - sL / (WL * WL), GR = 1 - sR / (WR * WR);
    return Gnode - (WL / Wnode) * GL - (WR / Wnode) * GR;
  };
  const gainOf = gainType === "ig" ? gainIg : gainType === "gini" ? gainGini : gainHb;

  // Threshold sweep: Left = {coverage kArr[i] ≥ k}, k = s…1 via suffix sum over k.
  // Returns the best (gain, k) with both sides ≥ sMin. Cost O(P + s·C).
  const sweep = (kArr, s) => {
    const H = new Int32Array(C * (s + 1)); // Hlab[c][k]
    for (let i = 0; i < P; i++) H[labPos[i] * (s + 1) + kArr[i]]++;
    const a = new Int32Array(C);
    let best = { g: 0, k: 0 }, mLeft = 0;
    for (let kk = s; kk >= 1; kk--) {
      let added = 0;
      for (let c = 0; c < C; c++) { const h = H[c * (s + 1) + kk]; a[c] += h; added += h; }
      mLeft += added;
      if (mLeft >= sMin && P - mLeft >= sMin) { const g = gainOf(a); if (g > best.g) best = { g, k: kk }; }
    }
    return best;
  };

  // Frequency shortlist of candidate bits not yet in the part.
  const pSet = new Set();
  const shortlist = () => {
    const freq = new Map();
    for (let i = 0; i < P; i++) { const arr = featArr[idx[i]]; for (const b of arr) if (!pSet.has(b)) freq.set(b, (freq.get(b) || 0) + 1); }
    let arr = [...freq.keys()];
    if (arr.length > M) arr = [...freq.entries()].sort((x, y) => y[1] - x[1]).slice(0, M).map((e) => e[0]);
    return arr;
  };

  // Greedy accretion. kArr = coverage counts of the current part (persist across steps).
  const part = [];
  const kArr = new Int32Array(P), trial = new Int32Array(P);
  let gPrev = 0, kStar = 0;
  for (let step = 0; step < D; step++) {
    const cand = shortlist();
    let bestB = -1, bestG = gPrev, bestK = kStar;
    for (const b of cand) {
      for (let i = 0; i < P; i++) trial[i] = kArr[i] + (featSet[idx[i]].has(b) ? 1 : 0);
      const r = sweep(trial, step + 1);
      if (r.g > bestG) { bestG = r.g; bestB = b; bestK = r.k; }
    }
    if (bestB < 0 || bestG - gPrev <= lambda) break; // MDL floor / no improvement
    part.push(bestB); pSet.add(bestB);
    for (let i = 0; i < P; i++) if (featSet[idx[i]].has(bestB)) kArr[i]++;
    gPrev = bestG; kStar = bestK;
  }
  if (part.length === 0) return null;

  const fire = [], notFire = [];
  for (let i = 0; i < P; i++) (kArr[i] >= kStar ? fire : notFire).push(idx[i]);
  if (fire.length < sMin || notFire.length < sMin) return null;
  return { bits: part, k: kStar, threshold: kStar / part.length, fire, notFire, gain: gPrev };
};

export default growPart;
