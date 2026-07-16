"use strict";

/**
 * @file discriminativeTree.js
 * @brief Track 3 of part generation — the balanced-split discrimination tree.
 *
 * Greedily grows a set of Start/Mid/End parts that best *separate* the word
 * population. For each length `L` in `[lMin, lMax]` it repeatedly adds the
 * candidate part whose presence splits the current partition most — scored by
 * the **unweighted average per-subset binary-entropy balance**,
 * `(Σ_g H₂(p_g)) / D` over the `D` splittable subsets — until no split clears
 * `minGain`. A balanced (≈50/50) split scores 1; a non-split scores 0.
 *
 * These are the decorrelated, word-separating parts; combined with the frequency
 * head ({@link generativeAffixes}) they cover both discrimination and the common
 * morphemes the tree alone skips.
 */

import { Kind } from "../kind.js";

const h2 = (p) => (p <= 0 || p >= 1 ? 0 : -p * Math.log2(p) - (1 - p) * Math.log2(1 - p));

/** @description Length-L Start/End/Mid candidate → word indices (lists < 2 dropped). @ignore */
const buildCandidates = (words, alive, L) => {
  const occ = new Map();
  const add = (key, wi) => { let a = occ.get(key); if (!a) occ.set(key, (a = [])); a.push(wi); };
  for (const wi of alive) {
    const w = words[wi], len = w.length;
    if (len > L) { add("S|" + w.slice(0, L), wi); add("E|" + w.slice(len - L), wi); }
    const seen = new Set();
    for (let p = 1; p + L <= len - 1; p++) { const key = "M|" + w.slice(p, p + L); if (!seen.has(key)) { seen.add(key); add(key, wi); } }
  }
  for (const [key, list] of occ) if (list.length < 2) occ.delete(key);
  return occ;
};

const kindOfTag = (t) => (t === "S" ? Kind.Start : t === "E" ? Kind.End : Kind.Mid);

/**
 * @function discriminativeTree
 * @description Adds balanced-split Start/Mid/End parts to `dict`.
 *
 * @param {string[]} words - Vocabulary.
 * @param {Float64Array|number[]} freq - Token frequency per word index.
 * @param {number[]} alive - Indices of words in play (not taken as wholes).
 * @param {import("../dictionary.js").PartDictionary} dict - Dictionary to add to.
 * @param {Set<string>} addedSet - Receives the `"kind|value"` keys added, mutated.
 * @param {Object} [opts]
 * @param {number} [opts.lMax=7] @param {number} [opts.lMin=3]
 * @param {number} [opts.minGain=0.0008] - Min average-balance gain to keep splitting.
 * @param {number} [opts.maxTree=20000] - Safety fuse on total parts added.
 * @returns {number} Count of parts added.
 *
 * @example
 * const disc = new Set();
 * discriminativeTree(words, freq, alive, dict, disc);   // adds word-separating parts
 */
export const discriminativeTree = (words, freq, alive, dict, addedSet, opts = {}) => {
  const { lMax = 7, lMin = 3, minGain = 0.0008, maxTree = 20000 } = opts;
  const EPS = 1e-12;
  const groupId = new Int32Array(words.length);
  const groupMass = [alive.reduce((s, wi) => s + freq[wi], 0)];
  const groupCount = [alive.length];
  let added = 0;
  for (let L = lMax; L >= lMin && added < maxTree; L--) {
    const occ = buildCandidates(words, alive, L);
    for (;;) {
      let D = 0; for (const c of groupCount) if (c >= 2) D++;
      const yes = new Map();
      let best = null;
      for (const [key, list] of occ) {
        yes.clear();
        let freqSum = 0;
        for (const wi of list) { const g = groupId[wi], f = freq[wi]; freqSum += f; yes.set(g, (yes.get(g) || 0) + f); }
        let sumH = 0;
        for (const [g, y] of yes) { const m = groupMass[g]; if (y > 0 && y < m) sumH += h2(y / m); }
        const gain = D > 0 ? sumH / D : 0;
        if (gain < minGain) continue;
        const vlen = key.length - 2;
        if (best === null || gain > best.gain + EPS ||
            (Math.abs(gain - best.gain) <= EPS && (vlen > best.vlen || (vlen === best.vlen && (freqSum > best.fs || (freqSum === best.fs && key < best.key)))))) {
          best = { key, list, gain, vlen, fs: freqSum };
        }
      }
      if (!best) break;
      const kind = kindOfTag(best.key[0]), val = best.key.slice(2);
      dict.add(kind, val); addedSet.add(kind + "|" + val); added++;
      const byG = new Map();
      for (const wi of best.list) { const g = groupId[wi]; let e = byG.get(g); if (!e) byG.set(g, (e = { m: [], y: 0 })); e.m.push(wi); e.y += freq[wi]; }
      for (const [g, e] of byG) if (e.y > 0 && e.y < groupMass[g]) {
        const nid = groupMass.length;
        groupMass.push(e.y); groupMass[g] -= e.y;
        groupCount.push(e.m.length); groupCount[g] -= e.m.length;
        for (const wi of e.m) groupId[wi] = nid;
      }
      if (added >= maxTree) break;
    }
  }
  return added;
};

export default discriminativeTree;
