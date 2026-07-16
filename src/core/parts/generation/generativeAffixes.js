"use strict";

/**
 * @file generativeAffixes.js
 * @brief Track 2 of part generation — the frequency-head affixes.
 *
 * For each length `L` in `[lMin, lMax]` and each affix position
 * (Start/Mid/End), collects the candidate substrings with their token-weighted
 * frequency, then takes only the clean frequency **head** (gated by
 * {@link elbowEffectiveCount}, skipped when flat). This deliberately captures a
 * small set of *obvious* common morphemes (`ing`, `tion`, `un`, `pre`, `form`,
 * `graph`, …) — the coverage the discriminative tree tends to skip — without
 * saturating the vocabulary.
 */

import { Kind } from "../kind.js";
import { elbowEffectiveCount } from "../../math/adaptiveThreshold.js";

/**
 * @function generativeAffixes
 * @description Adds the frequency-head Start/Mid/End parts to `dict`.
 *
 * @param {string[]} words - Vocabulary.
 * @param {Float64Array|number[]} freq - Token frequency per word index.
 * @param {number[]} alive - Indices of words still in play (not taken as wholes).
 * @param {import("../dictionary.js").PartDictionary} dict - Dictionary to add to.
 * @param {Set<string>} addedSet - Receives the `"kind|value"` keys added (for
 *   track tagging), mutated in place.
 * @param {Object} [opts]
 * @param {number} [opts.lMax=7] @param {number} [opts.lMin=3]
 * @param {number} [opts.elbowSig=0.01]
 * @returns {number} Count of parts added.
 *
 * @example
 * const gen = new Set();
 * generativeAffixes(words, freq, alive, dict, gen);   // adds -ing, -tion, un-, form…
 */
export const generativeAffixes = (words, freq, alive, dict, addedSet, opts = {}) => {
  const { lMax = 7, lMin = 3, elbowSig = 0.01 } = opts;
  let n = 0;
  for (let L = lMax; L >= lMin; L--) {
    for (const kind of [Kind.Start, Kind.Mid, Kind.End]) {
      const cand = new Map();
      for (const wi of alive) {
        const w = words[wi], len = w.length, f = freq[wi];
        if (kind === Kind.Start) { if (len > L) { const s = w.slice(0, L); cand.set(s, (cand.get(s) || 0) + f); } }
        else if (kind === Kind.End) { if (len > L) { const s = w.slice(len - L, len); cand.set(s, (cand.get(s) || 0) + f); } }
        else { const seen = new Set(); for (let p = 1; p + L <= len - 1; p++) { const s = w.slice(p, p + L); if (!seen.has(s)) { seen.add(s); cand.set(s, (cand.get(s) || 0) + f); } } }
      }
      if (!cand.size) continue;
      const sorted = [...cand.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
      const k = elbowEffectiveCount(sorted.map((e) => e[1]), elbowSig);
      if (k < sorted.length) for (let i = 0; i < k; i++) { dict.add(kind, sorted[i][0]); addedSet.add(kind + "|" + sorted[i][0]); n++; }
    }
  }
  return n;
};

export default generativeAffixes;
