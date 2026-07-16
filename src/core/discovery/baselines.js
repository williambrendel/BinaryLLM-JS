"use strict";

/**
 * @file baselines.js
 * @brief §6c naive baselines — the floor the discovery methods must beat. All
 * produce the SAME model `⟨D_τ; [(Q_k,p⁻_k,t_k)]⟩` and use the same gate (§4),
 * peel (§5.3), and max-pool readout (§7); only how the positive sets `Q_k` are
 * produced is dumb. They isolate how much of any win is DISCOVERY vs the
 * gate + negative channel + max-pool machinery alone.
 *
 *   6c-i  per-example       : K=|A|, one memorized template per positive.
 *   6c-ii random-chunk c=1  : disjoint random chunks of p⁺_glob∧M_τ (recall collapse).
 *   6c-ii random-chunk c=2  : capped-overlap random chunks (recall-viable control).
 *   6c-ii-a SA-balance      : anneal the c-partition on BALANCE only (no gain, no Ā_h).
 *
 * Negatives are peeled against full `Ā_h` (no residualization — it's a baseline).
 */

import peelNegative from "./peelNegative.js";
import fitThreshold from "./thresholdFit.js";
import { adaptedGateScore } from "../gate/adaptedGate.js";
import andCount from "../math/sparse/andCount.js";

const lcg = (seed) => () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 0x100000000;
const shuffle = (arr, rnd) => { const c = arr.slice(); for (let i = c.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [c[i], c[j]] = [c[j], c[i]]; } return c; };

const finishGate = (Q, A, AbarH, Dtau, kappa, m) => {
  const P = A.filter((y) => andCount(y, Q) >= m);
  const { pMinus } = peelNegative(Q, AbarH, P.length ? P : A, { Dtau, kappa, m });
  const { t } = fitThreshold((x) => adaptedGateScore(x, { pPlusAdapted: Q, pMinus, dropSet: Dtau }), P.length ? P : A, AbarH);
  return { Q, pMinus, t, size: P.length };
};

/** p⁺_glob ∧ M_τ = OR(A) ∖ D_τ, as a sorted array. */
const adaptedPositive = (A, Dtau) => {
  const Dset = new Set(Dtau), s = new Set();
  for (const y of A) for (const b of y) if (!Dset.has(b)) s.add(b);
  return Uint32Array.from(s).sort();
};

/**
 * @function baselinePerExample  (6c-i)
 * @returns {{Dtau:number[], gates:Array, algo:string}} K=|A| memorized templates.
 */
export const baselinePerExample = (A, AbarH, Dtau, opts = {}) => {
  const { kappa = 2, m = 2 } = opts, Dset = new Set(Dtau);
  const gates = A.map((y) => finishGate([...y].filter((b) => !Dset.has(b)), A, AbarH, Dtau, kappa, m))
    .filter((g) => g.Q.length > 0);
  return { Dtau, gates, algo: "per_example" };
};

/**
 * Partition `P⁺` into K chunks (c=1 disjoint; c=2 adds capped random 2nd
 * memberships up to `shareBudget` extra assignments), returned as sorted arrays.
 */
const chunkPartition = (Pplus, K, c, shareBudget, rnd) => {
  const bits = shuffle([...Pplus], rnd);
  const chunks = Array.from({ length: K }, () => []);
  bits.forEach((b, i) => chunks[i % K].push(b));
  if (c === 2 && shareBudget > 0) {
    const extra = shuffle(bits, rnd).slice(0, shareBudget);
    for (const b of extra) { let k = Math.floor(rnd() * K); if (!chunks[k].includes(b)) chunks[k].push(b); }
  }
  return chunks.map((ch) => Uint32Array.from(new Set(ch)).sort());
};

/**
 * @function baselineRandomChunk  (6c-ii)
 * @param {number} K @param {object} opts
 * @param {1|2} [opts.c=2] @param {number} [opts.seed=1] @param {number} [opts.shareBudget=0]
 * @returns {{Dtau:number[], gates:Array, algo:string}}
 */
export const baselineRandomChunk = (A, AbarH, Dtau, K, opts = {}) => {
  const { c = 2, seed = 1, shareBudget = 0, kappa = 2, m = 2, sMin = 3 } = opts;
  const Pplus = adaptedPositive(A, Dtau);
  const chunks = chunkPartition(Pplus, K, c, shareBudget, lcg(seed >>> 0 || 1));
  const gates = chunks.map((Q) => finishGate([...Q], A, AbarH, Dtau, kappa, m)).filter((g) => g.size >= sMin);
  return { Dtau, gates, algo: c === 1 ? "random_chunk_c1" : "random_chunk_c2" };
};

/**
 * @function baselineRandomChunkSA  (6c-ii-a) — anneal on BALANCE only.
 * Objective is coherence-free: chunk bit-mass imbalance + uncovered-bit penalty
 * + sharing-budget violation. NO gain, NO Ā_h (that would make it a method, §6d).
 */
export const baselineRandomChunkSA = (A, AbarH, Dtau, K, opts = {}) => {
  const { c = 2, seed = 1, shareBudget = 0, kappa = 2, m = 2, sMin = 3, iters = 2000, t0 = 1, cool = 0.999 } = opts;
  const rnd = lcg(seed >>> 0 || 1);
  const Pplus = [...adaptedPositive(A, Dtau)];
  // membership: bit → set of chunk indices
  let owner = Pplus.map((_, i) => [i % K]);
  const balanceCost = (own) => {
    const sizes = new Array(K).fill(0); let shares = 0, uncovered = 0;
    own.forEach((ks) => { if (ks.length === 0) uncovered++; else { ks.forEach((k) => sizes[k]++); shares += ks.length - 1; } });
    const mean = Pplus.length / K; let imb = 0; for (const s of sizes) imb += (s - mean) * (s - mean);
    return imb + 5 * uncovered + 10 * Math.max(0, shares - shareBudget);
  };
  let cur = balanceCost(owner), T = t0;
  for (let it = 0; it < iters; it++) {
    const i = Math.floor(rnd() * Pplus.length);
    const prev = owner[i].slice();
    const move = rnd();
    if (move < 0.5 || prev.length === 0) owner[i] = [Math.floor(rnd() * K)];               // reassign
    else if (move < 0.8 && c === 2) owner[i] = [...new Set([...prev, Math.floor(rnd() * K)])]; // add share
    else owner[i] = prev.slice(0, Math.max(1, prev.length - 1));                            // drop share
    const next = balanceCost(owner);
    if (next <= cur || rnd() < Math.exp(-(next - cur) / T)) cur = next; else owner[i] = prev;
    T *= cool;
  }
  const chunks = Array.from({ length: K }, () => []);
  owner.forEach((ks, i) => ks.forEach((k) => chunks[k].push(Pplus[i])));
  const gates = chunks.map((ch) => finishGate(Uint32Array.from(new Set(ch)).sort(), A, AbarH, Dtau, kappa, m)).filter((g) => g.size >= sMin);
  return { Dtau, gates, algo: "random_chunk_sa" };
};

export default { baselinePerExample, baselineRandomChunk, baselineRandomChunkSA };
