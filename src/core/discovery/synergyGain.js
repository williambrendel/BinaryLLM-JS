"use strict";

/**
 * @file synergyGain.js
 * @brief Discriminative-clique gain: the m-of-n information-gain objective and
 * its synergy increment, over sparse bit-set contexts. §5.1 of the multi-part
 * discovery spec.
 *
 * A "context" is a sorted, strictly-increasing array of bit ids (the signature
 * of one context). A pool is an array of contexts. The objective a clique `Q`
 * is scored on is the best m-of-n decision stump `|Q ∧ x| ≥ s` separating a
 * positive slice `P` from residual negatives `N` — the SAME inner gate the
 * scorer uses (§4/§7), so discovery and inference optimize the same thing.
 *
 * There is no co-occurrence / lift graph here (§11): `d(b) > κ` marginal
 * enrichment prunes the search, and synergy = the m-of-n gain improvement from
 * adding a bit. Density never selects.
 */

const LOG2 = Math.LN2;
const log2 = (x) => Math.log(x) / LOG2;

/**
 * Binary entropy of a group holding `a` positives and `b` negatives, in bits.
 * @returns {number} H = −Σ p·log₂p; 0 for an empty or pure group.
 */
export const ent2 = (a, b) => {
  const n = a + b;
  if (n === 0) return 0;
  const pa = a / n, pb = b / n;
  return -(pa > 0 ? pa * log2(pa) : 0) - (pb > 0 ? pb * log2(pb) : 0);
};

/**
 * Information gain of the boolean split (fire vs not-fire) that puts `firP`
 * positives and `firN` negatives in the fire group, out of `nPos`/`nNeg` total.
 * @returns {number} parent entropy − weighted child entropy (bits).
 */
export const splitGain = (firP, firN, nPos, nNeg) => {
  const total = nPos + nNeg;
  if (total === 0) return 0;
  const parent = ent2(nPos, nNeg);
  const w1 = (firP + firN) / total;
  const child = w1 * ent2(firP, firN) + (1 - w1) * ent2(nPos - firP, nNeg - firN);
  return parent - child;
};

/**
 * Build the per-bit hit-list index for a pool: `bit → ascending context ids`.
 * @param {Array<number[]|Uint16Array|Uint32Array>} contexts
 * @returns {Map<number, number[]>}
 */
export const buildHitLists = (contexts) => {
  const h = new Map();
  for (let i = 0; i < contexts.length; i++) {
    const y = contexts[i];
    for (let t = 0; t < y.length; t++) {
      const b = y[t];
      let l = h.get(b);
      if (!l) { l = []; h.set(b, l); }
      l.push(i);
    }
  }
  return h;
};

/**
 * A reusable per-pool counter: `count(Q)` returns, for each context in the
 * pool, `|Q ∧ context|` — the m-of-n match count — in O(Σ_{b∈Q} |hits(b)|).
 * @param {Map<number, number[]>} hitLists
 * @param {number} n - pool size
 */
export const makeCounter = (hitLists, n) => ({
  count: (Q) => {
    const c = new Int32Array(n);
    for (let t = 0; t < Q.length; t++) {
      const l = hitLists.get(Q[t]);
      if (!l) continue;
      for (let u = 0; u < l.length; u++) c[l[u]]++;
    }
    return c;
  },
});

/**
 * Best m-of-n gain of clique `Q` separating `P` from `N`, and the support
 * threshold that achieves it. `GAIN(Q ; P, Ā_res)` of §5.1.
 * @param {number[]} Q - sorted clique bit ids
 * @param {{count:Function}} posCounter - counter over P
 * @param {{count:Function}} negCounter - counter over N
 * @param {number} nPos
 * @param {number} nNeg
 * @param {number} [sFloor=2] - coherence floor: the m-of-n support is searched
 *   over `s ∈ [min(sFloor,|Q|) .. |Q|]`, never below `sFloor`. `s=1` is an OR
 *   (any-bit-fires) and lets unions of cliques blend (§11 length detector), so
 *   a clique must fire on ≥2 CO-OCCURRING bits to score. Pass `1` only to rank
 *   single bits (seeding heuristics).
 * @returns {{gain:number, support:number}} best gain (bits) and its threshold s.
 */
export const gainSupport = (Q, posCounter, negCounter, nPos, nNeg, sFloor = 2) => {
  if (Q.length === 0) return { gain: 0, support: 0 };
  const cP = posCounter.count(Q), cN = negCounter.count(Q);
  const maxc = Q.length;
  const low = Math.min(sFloor, maxc);           // sub-floor cliques scored at full presence
  // fireP[s] = #{cP ≥ s}, fireN[s] likewise, for s = 1..maxc (histogram → suffix sum)
  const hP = new Int32Array(maxc + 2), hN = new Int32Array(maxc + 2);
  for (let i = 0; i < cP.length; i++) hP[cP[i]]++;
  for (let j = 0; j < cN.length; j++) hN[cN[j]]++;
  let firP = 0, firN = 0, bestGain = 0, bestS = low;
  for (let s = maxc; s >= 1; s--) {
    firP += hP[s]; firN += hN[s];               // suffix count of "≥ s"
    if (s < low) continue;                       // enforce the support floor
    // One-sided: a clique is a POSITIVE detector, so only score a support whose
    // fire group is positive-enriched (lift > 1). Without this, a mixed pair
    // {s1,s2} that fires only on cross-sense NEGATIVES scores a high (pure but
    // inverted) info gain — an anti-clique. Require firP·nNeg > firN·nPos.
    if (firP * nNeg <= firN * nPos) continue;
    const g = splitGain(firP, firN, nPos, nNeg);
    if (g > bestGain) { bestGain = g; bestS = s; }
  }
  return { gain: bestGain, support: bestS };
};

/**
 * Per-bit lift `d(b) = rate_P(b) / rate_N(b)` with additive-ε smoothing on the
 * rates, so an unseen bit lands at d≈1 rather than exploding.
 * @returns {number}
 */
export const bitLift = (b, hitP, nPos, hitN, nNeg) => {
  const eps = 1 / (nNeg + 1);
  const rP = (hitP.get(b)?.length || 0) / nPos;
  const rN = (hitN.get(b)?.length || 0) / nNeg;
  return (rP + eps) / (rN + eps);
};

export default gainSupport;
